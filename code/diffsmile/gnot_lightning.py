from __future__ import annotations

import argparse
import datetime as dt
import gc
import math
from typing import TYPE_CHECKING, Any, Literal, cast, override

import lightning as L  # noqa: N812
import optuna
import polars as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from optuna.integration import PyTorchLightningPruningCallback
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from diffsmile.config import ConditioningScalarIndex, TrainingConfigConditionalDiffusion, config, dataset_config
from diffsmile.model.dataset import TimeSeriesDataset
from diffsmile.model.gnot_diffusion import DiffusionGNOT
from diffsmile.model.loss import ButterflyArbitrageLoss, CalendarArbitrageLoss, SmoothnessLoss
from diffsmile.model.scheduler_rad import RegionAwareScheduler
from diffsmile.model.validation import run_gnot_inpaint_kernelized_trace

if TYPE_CHECKING:
    from pathlib import Path

    from jaxtyping import Float


DATA_DIR = dataset_config.conditional_data_path
BatchSample = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def scale_tte_mesh(tte: torch.Tensor) -> torch.Tensor:
    return torch.log1p(tte / 7.0) / math.log1p(730 / 7.0)


class VolSurfacePointCloudDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        batch_size: int = 64,
        num_workers: int = 4,
        config: TrainingConfigConditionalDiffusion = config,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.config = config

        self.scaler = StandardScaler()
        self.train_dataset: TimeSeriesDataset | None = None
        self.val_dataset: TimeSeriesDataset | None = None

        self.data_mean: torch.Tensor | None = None
        self.data_std: torch.Tensor | None = None

        self.log_moneyness_grid: torch.Tensor | None = None
        self.ttm_grid: torch.Tensor | None = None
        self.cut_off_train_idx: int | None = None

    def _normalize_dates(self, dates: list[dt.date | dt.datetime]) -> list[dt.date]:
        normalized_dates: list[dt.date] = []
        for value in dates:
            if isinstance(value, dt.datetime):
                normalized_dates.append(value.date())
            else:
                normalized_dates.append(value)
        return normalized_dates

    def _load_and_preprocess(
        self,
        surfaces_path: Path,
        scalars_path: Path,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[dt.date],
        list[int],
        list[int],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        all_surfaces = torch.load(surfaces_path, weights_only=False)
        merged_scalars = torch.as_tensor(torch.load(scalars_path, weights_only=False)).float()

        iv_surfaces = all_surfaces["iv_surfaces"]
        dates = self._normalize_dates(all_surfaces["dates"])
        log_moneyness_grid = all_surfaces["log_moneyness_grid"]
        ttm_grid = all_surfaces["ttm_grid"]

        if iv_surfaces.shape[0] != len(dates):
            msg = "Merged surfaces must have the same number of surface rows as dates."
            raise ValueError(msg)
        if merged_scalars.ndim != 2:  # noqa: PLR2004
            msg = "Merged conditioning scalars must be a 2D tensor-like object."
            raise ValueError(msg)
        if merged_scalars.shape[0] != len(dates):
            msg = "Merged surfaces and conditioning scalars must have the same number of rows."
            raise ValueError(msg)

        train_indices = [idx for idx, date in enumerate(dates) if dataset_config.train_start <= date <= dataset_config.train_end]
        val_indices = [idx for idx, date in enumerate(dates) if dataset_config.val_start <= date <= dataset_config.val_end]

        if not train_indices:
            msg = "Merged dataset produced an empty train split for the configured date range."
            raise ValueError(msg)
        if not val_indices:
            msg = "Merged dataset produced an empty val split for the configured date range."
            raise ValueError(msg)

        all_scalars = merged_scalars[:, 1:]
        if all_scalars.shape[1] != self.config.SCALAR_COUNT:
            msg = f"Expected {self.config.SCALAR_COUNT} conditioning scalars after dropping the first column, got {all_scalars.shape[1]}."  # noqa: E501
            raise ValueError(msg)
        train_scalars = all_scalars[train_indices]

        self.scaler.fit(train_scalars.numpy())
        all_scalars_scaled = self.scaler.transform(all_scalars.numpy())

        surfaces_tensor = iv_surfaces.unsqueeze(1).float()
        all_scalars_tensor = torch.from_numpy(all_scalars_scaled).float()

        log_surfaces = surfaces_tensor.log()
        mean = log_surfaces[train_indices].mean()
        std = log_surfaces[train_indices].std()
        surfaces_norm = (log_surfaces - mean) / std

        flat_np = iv_surfaces.reshape(len(iv_surfaces), -1).numpy()
        df_pl = pl.DataFrame(flat_np)

        ewm_5d = (
            df_pl.with_columns(pl.all().ewm_mean(span=5))
            .to_torch()
            .reshape(len(iv_surfaces), self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH)  # ty:ignore[unresolved-attribute]
        )
        ewm_20d = (
            df_pl.with_columns(pl.all().ewm_mean(span=20))
            .to_torch()
            .reshape(len(iv_surfaces), self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH)  # ty:ignore[unresolved-attribute]
        )

        ewm_5d_norm = ((ewm_5d.log() - mean) / std).float()
        ewm_20d_norm = ((ewm_20d.log() - mean) / std).float()

        return (
            surfaces_norm,
            ewm_5d_norm,
            ewm_20d_norm,
            all_scalars_tensor,
            dates,
            train_indices,
            val_indices,
            mean,
            std,
            log_moneyness_grid,
            ttm_grid,
        )

    def _reshape_surface(self, surface: torch.Tensor) -> torch.Tensor:
        return surface.reshape(-1, 1, self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH)

    def _build_dataset(
        self,
        *,
        surfaces: torch.Tensor,
        ema_short: torch.Tensor,
        ema_long: torch.Tensor,
        scalars: torch.Tensor,
        dates: list[dt.date],
    ) -> TimeSeriesDataset:
        return TimeSeriesDataset(
            surfaces=surfaces,
            surfaces_ema_short=self._reshape_surface(ema_short),
            surfaces_ema_long=self._reshape_surface(ema_long),
            scalars=scalars,
            dates=dates,
        )

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        train_surfaces_path = dataset_config.merged_surfaces_path
        train_scalars_path = dataset_config.merged_conditioning_scalars_path

        (
            surfaces_norm,
            ewm_5d_norm,
            ewm_20d_norm,
            all_scalars_tensor,
            dates,
            train_indices,
            val_indices,
            self.data_mean,
            self.data_std,
            self.log_moneyness_grid,
            self.ttm_grid,
        ) = self._load_and_preprocess(train_surfaces_path, train_scalars_path)

        self.train_dataset = self._build_dataset(
            surfaces=surfaces_norm[train_indices],
            ema_short=ewm_5d_norm[train_indices],
            ema_long=ewm_20d_norm[train_indices],
            scalars=all_scalars_tensor[train_indices],
            dates=[dates[idx] for idx in train_indices],
        )

        self.val_dataset = self._build_dataset(
            surfaces=surfaces_norm[val_indices],
            ema_short=ewm_5d_norm[val_indices],
            ema_long=ewm_20d_norm[val_indices],
            scalars=all_scalars_tensor[val_indices],
            dates=[dates[idx] for idx in val_indices],
        )

    def _make_dataloader(self, dataset: TimeSeriesDataset, *, shuffle: bool, drop_last: bool = False) -> DataLoader[BatchSample]:
        dataloader_kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "persistent_workers": self.num_workers > 0,
            "pin_memory": True,
            "drop_last": drop_last,
        }
        if self.num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = 4

        return DataLoader(dataset, **dataloader_kwargs)

    def train_dataloader(self) -> DataLoader[BatchSample]:
        assert self.train_dataset is not None
        return self._make_dataloader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader[BatchSample]:
        assert self.val_dataset is not None
        return self._make_dataloader(self.val_dataset, shuffle=False)


class GNOTDiffusionLightningModule(L.LightningModule):
    fixed_coords: Float[torch.Tensor, "N 2"]

    def __init__(  # noqa: PLR0913
        self,
        embed_dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 4,
        n_experts: int = 4,
        mlp_layers: int = 2,
        lr: float = 0.0002,
        weight_decay: float = 1e-5,
        T_1: int = 500,
        T_2: int = 500,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        dropout: float = 0.1,
        butterfly_loss_weight: float = 0.00005,
        calendar_loss_weight: float = 0.00005,
        smoothness_loss_weight: float = 0.01,
        smoothness_weight_mode: Literal["direct", "inverse_sqrt", "uniform"] = "direct",
        *,
        smoothness_penalize_moneyness: bool = True,
        smoothness_penalize_maturity: bool = True,
        use_arbitrage_loss: bool = False,
        config: TrainingConfigConditionalDiffusion = config,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config

        self.model = DiffusionGNOT(
            space_dim=2,
            value_dim=1,
            scalar_dim=config.SCALAR_COUNT,
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_experts=n_experts,
            mlp_layers=mlp_layers,
            dropout=dropout,
        )

        self.mse_loss_fn = nn.MSELoss(reduction="none")
        self.butterfly_loss: ButterflyArbitrageLoss | None = None
        self.calendar_loss: CalendarArbitrageLoss | None = None

        self.lr = lr
        self.weight_decay = weight_decay
        self.T_total = T_1 + T_2
        self.use_arbitrage_loss = use_arbitrage_loss
        self.butterfly_loss_weight = butterfly_loss_weight
        self.calendar_loss_weight = calendar_loss_weight

        self.scheduler = RegionAwareScheduler(T_1=T_1, T_2=T_2, beta_start=beta_start, beta_end=beta_end)

        self.log_moneyness_grid: torch.Tensor | None = None
        self.ttm_grid: torch.Tensor | None = None
        self.data_mean: torch.Tensor | None = None
        self.data_std: torch.Tensor | None = None

        self.smoothness_loss_weight = smoothness_loss_weight
        self.smoothness_penalize_moneyness = smoothness_penalize_moneyness
        self.smoothness_penalize_maturity = smoothness_penalize_maturity
        self.smoothness_loss_fn: SmoothnessLoss | None = None
        self.smoothness_weight_mode: Literal["direct", "inverse_sqrt", "uniform"] = smoothness_weight_mode

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None and hasattr(dm, "log_moneyness_grid"):
            self.log_moneyness_grid = dm.log_moneyness_grid.to(self.device)
            self.ttm_grid = dm.ttm_grid.to(self.device)
            assert self.log_moneyness_grid is not None
            assert self.ttm_grid is not None

            logk_mesh, tte_mesh = torch.meshgrid(self.log_moneyness_grid, self.ttm_grid, indexing="ij")

            tte_mesh = scale_tte_mesh(tte_mesh)
            coords = torch.stack([tte_mesh.flatten(), logk_mesh.flatten()], dim=1)

            self.fixed_coords = coords

            self.butterfly_loss = ButterflyArbitrageLoss(
                log_moneyness=dm.log_moneyness_grid.to(self.device),
                maturities=dm.ttm_grid.to(self.device),
            )
            self.smoothness_loss_fn = SmoothnessLoss(
                maturities=dm.ttm_grid.to(self.device),
                penalize_moneyness=self.smoothness_penalize_moneyness,
                penalize_maturity=self.smoothness_penalize_maturity,
                weight_mode=self.smoothness_weight_mode,
                min_weight=0.5,
                max_weight=3.0,
            )

            self.calendar_loss = CalendarArbitrageLoss(maturities=dm.ttm_grid.to(self.device))

            self.data_mean = dm.data_mean.to(self.device)
            self.data_std = dm.data_std.to(self.device)

    def _shared_step(  # noqa: PLR0915
        self,
        batch: tuple[torch.Tensor, ...],
        split: Literal["train", "val"] = "train",
    ) -> torch.Tensor:
        assert self.smoothness_loss_fn is not None
        surf, ema_short, ema_long, scalars, next_surf = batch
        B = surf.size(0)
        H, W = self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH
        N = H * W

        assert hasattr(self, "fixed_coords")
        all_coords = self.fixed_coords.unsqueeze(0).expand(B, -1, -1)

        # use bernoulli mask instead
        mask = torch.bernoulli(torch.full((B, 1, H, W), 0.5, device=self.device))

        t = torch.randint(0, self.T_total, (B, 1), device=self.device)
        signal_scale, noise_scale = self.scheduler.get_sampling_scales(t, mask)

        noise = torch.randn_like(next_surf, device=self.device)
        noisy_image = signal_scale * next_surf + noise_scale * noise

        context_coords = all_coords
        noisy_values = noisy_image.permute(0, 2, 3, 1).reshape(B, N, 1)
        noise_scale_flat = noise_scale.permute(0, 2, 3, 1).reshape(B, N, 1)

        # Context: yesterday's surface + EMA short + EMA long (B, N, 3)
        surf_flat = surf.permute(0, 2, 3, 1).reshape(B, N, 1)
        ema_short_flat = ema_short.permute(0, 2, 3, 1).reshape(B, N, 1)
        ema_long_flat = ema_long.permute(0, 2, 3, 1).reshape(B, N, 1)
        context_values = torch.cat([surf_flat, ema_short_flat, ema_long_flat], dim=-1)

        noise_pred = self.model(
            query_coords=all_coords,
            noisy_values=noisy_values,
            noise_scale=noise_scale_flat,
            scalars=scalars,
            context_coords=context_coords,
            context_values=context_values,
        )
        noise_pred_img = noise_pred.view(B, H, W, 1).permute(0, 3, 1, 2)
        x_0_pred = (noisy_image - noise_scale * noise_pred_img) / signal_scale.clamp(min=1e-5)

        # to point cloud
        noise_flat = noise.permute(0, 2, 3, 1).reshape(B, N, 1)

        # Loss computation with region-aware masking and inverse time weighting
        loss_per_point = self.mse_loss_fn(noise_pred, noise_flat).squeeze(-1)  # (B, N)
        loss_mask = (noise_scale_flat.squeeze(-1) > 0.0).float()

        maturities = all_coords[..., 0]  # (B, N)
        time_weights = 1.0 / torch.sqrt(maturities.clamp(min=1e-4))
        time_weights = time_weights / time_weights.mean(dim=-1, keepdim=True)

        mse_loss = (loss_per_point * loss_mask * time_weights).sum() / loss_mask.sum().clamp(min=1.0)

        smoothness_penalty = self.smoothness_loss_fn(x_0_pred)
        smooth_term = self.smoothness_loss_weight * smoothness_penalty
        if self.use_arbitrage_loss and self.butterfly_loss is not None and self.calendar_loss is not None:
            assert self.data_mean is not None
            assert self.data_std is not None
            stats_mean, stats_std = self.data_mean, self.data_std

            x_0_log = torch.clamp(x_0_pred, -4.0, 4.0) * stats_std + stats_mean
            next_surf_log = torch.clamp(next_surf, -4.0, 4.0) * stats_std + stats_mean

            x_0_real = torch.clamp(torch.exp(x_0_log), min=1e-6, max=5.0)
            next_surf_real = torch.clamp(torch.exp(next_surf_log), min=1e-6, max=5.0)

            butterfly_term = self.butterfly_loss(x_0_real)
            calendar_term = self.calendar_loss(x_0_real)
            butterfly_next = self.butterfly_loss(next_surf_real)
            calendar_next = self.calendar_loss(next_surf_real)

            snr = (signal_scale**2) / (noise_scale**2 + 1e-8)
            weights = torch.clamp(
                snr.sum(dim=(-1, -2, -3)) / loss_mask.sum(dim=-1).clamp(min=1),
                max=5.0,
            )
            butterfly_diff = (weights * (butterfly_term - butterfly_next)).mean()
            calendar_diff = (weights * (calendar_term - calendar_next)).mean()
            arb_term = (self.butterfly_loss_weight * butterfly_diff) + (self.calendar_loss_weight * calendar_diff)
        else:
            butterfly_diff = torch.zeros((), device=self.device, dtype=torch.float32)
            calendar_diff = torch.zeros((), device=self.device, dtype=torch.float32)
            arb_term = torch.zeros((), device=self.device, dtype=torch.float32)

        loss = mse_loss + smooth_term + arb_term
        denom = (mse_loss.abs() + smooth_term.abs() + arb_term.abs()).clamp(min=1e-8)

        self.log_dict(
            {
                f"{split}/loss_total": loss.detach(),
                f"{split}/loss_mse": mse_loss.detach(),
                f"{split}/loss_smooth_raw": smoothness_penalty.detach(),
                f"{split}/loss_smooth_term": smooth_term.detach(),
                f"{split}/loss_butterfly_raw": butterfly_diff.detach(),
                f"{split}/loss_calendar_raw": calendar_diff.detach(),
                f"{split}/loss_arb_term": arb_term.detach(),
                f"{split}/loss_pct_mse": 100.0 * mse_loss.detach().abs() / denom,
                f"{split}/loss_pct_smooth": 100.0 * smooth_term.detach().abs() / denom,
                f"{split}/loss_pct_arb": 100.0 * arb_term.detach().abs() / denom,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        return loss

    @override
    def training_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, split="train")
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    @override
    def validation_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, split="val")
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    @override
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def sample_and_evaluate(
        self, dataset: TimeSeriesDataset, day_idx: int, batch_size: int = 10, num_steps: int = 500
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()

        surf_sample, ema_short, ema_long, scalars_sample, next_surf_sample = dataset[day_idx]

        H, W = self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH
        N = H * W
        B = batch_size

        assert self.log_moneyness_grid is not None
        assert self.ttm_grid is not None

        logk_mesh, tte_mesh = torch.meshgrid(self.log_moneyness_grid, self.ttm_grid, indexing="ij")
        tte_mesh = scale_tte_mesh(tte_mesh)
        coords = torch.stack([tte_mesh.flatten(), logk_mesh.flatten()], dim=1)
        coords = coords.unsqueeze(0).expand(B, -1, -1).to(self.device)

        surf_sample = surf_sample.view(1, 1, H, W).repeat(B, 1, 1, 1).to(self.device)
        ema_short_sample = ema_short.view(1, 1, H, W).repeat(B, 1, 1, 1).to(self.device)
        ema_long_sample = ema_long.view(1, 1, H, W).repeat(B, 1, 1, 1).to(self.device)
        scalars_sample = scalars_sample.view(1, -1).repeat(B, 1).to(self.device)

        mask = torch.ones((B, 1, H, W), device=self.device)

        _, initial_noise_scale = self.scheduler.get_sampling_scales(
            torch.tensor([[num_steps - 1]], device=self.device).repeat(B, 1), mask
        )
        x_t = initial_noise_scale * torch.randn((B, 1, H, W), device=self.device)

        surf_flat = surf_sample.permute(0, 2, 3, 1).reshape(B, N, 1)
        ema_short_flat = ema_short_sample.permute(0, 2, 3, 1).reshape(B, N, 1)
        ema_long_flat = ema_long_sample.permute(0, 2, 3, 1).reshape(B, N, 1)
        context_values = torch.cat([surf_flat, ema_short_flat, ema_long_flat], dim=-1)

        with torch.no_grad():
            for t_ in reversed(range(num_steps)):
                t = torch.tensor([[t_]], device=self.device).repeat(B, 1)
                _, noise_scale = self.scheduler.get_sampling_scales(t, mask)

                noisy_values = x_t.permute(0, 2, 3, 1).reshape(B, N, 1)
                noise_scale_flat = noise_scale.permute(0, 2, 3, 1).reshape(B, N, 1)

                noise_pred = self.model(
                    query_coords=coords,
                    noisy_values=noisy_values,
                    noise_scale=noise_scale_flat,
                    scalars=scalars_sample,
                    context_coords=coords,
                    context_values=context_values,
                )

                noise_pred_img = noise_pred.reshape(B, H, W, 1).permute(0, 3, 1, 2)
                x_t = self.scheduler.backward_eps(x_t=x_t, pred_eps=noise_pred_img, t=t, mask=mask)

        return x_t, next_surf_sample

    def sample_from_checkpoint(
        self, day_idx: int, batch_size: int = 10, num_steps: int = 500
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dm = VolSurfacePointCloudDataModule(config=self.config)
        dm.setup()

        assert dm.log_moneyness_grid is not None
        assert dm.ttm_grid is not None
        assert dm.val_dataset is not None

        self.log_moneyness_grid = dm.log_moneyness_grid.to(self.device)
        self.ttm_grid = dm.ttm_grid.to(self.device)

        return self.sample_and_evaluate(dm.val_dataset, day_idx, batch_size, num_steps)

    def inpaint_trace_from_checkpoint(
        self,
        day_idx: int,
        prob: float = 0.3,
        capture_ts: tuple[int, ...] | list[int] | None = None,
        *,
        scalars_override_raw: dict[ConditioningScalarIndex, float] | None = None,
        random_seed: int | None = None,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        if capture_ts is None:
            capture_ts = (0, 1, 5, 10, 20, 50, 100, 150, 200, 300, 400, 499)

        dm = VolSurfacePointCloudDataModule(config=self.config)
        dm.setup()

        assert dm.log_moneyness_grid is not None
        assert dm.ttm_grid is not None
        assert dm.val_dataset is not None
        assert dm.data_mean is not None
        assert dm.data_std is not None

        self.log_moneyness_grid = dm.log_moneyness_grid.to(self.device)
        self.ttm_grid = dm.ttm_grid.to(self.device)

        snapshots, target_img, mask_img, coords = run_gnot_inpaint_kernelized_trace(
            model=self,
            day_idx=day_idx,
            prob=prob,
            scheduler=self.scheduler,
            dataset_val=dm.val_dataset,
            config=self.config,
            device=self.device,
            log_moneyness_grid=self.log_moneyness_grid,
            ttm_grid=self.ttm_grid,
            capture_ts=capture_ts,
            scalars_override_raw=scalars_override_raw,
            scaler=dm.scaler,
            random_seed=random_seed,
        )
        return snapshots, target_img, mask_img, coords, dm.data_mean, dm.data_std


OPTUNA_BATCH_SIZE = 16
GRADIENT_CLIP_VAL = 1.0


def _parse_bool(value: str | bool) -> bool:  # noqa: FBT001
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    msg = f"Invalid boolean value: {value!r}. Use one of: true/false, yes/no, 1/0."
    raise argparse.ArgumentTypeError(msg)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--batch_size", type=int, default=OPTUNA_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_experts", type=int, default=4)
    parser.add_argument("--mlp_layers", type=int, default=2)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--butterfly_loss_weight", type=float, default=5e-5)
    parser.add_argument("--calendar_loss_weight", type=float, default=5e-5)
    parser.add_argument("--smoothness_loss_weight", type=float, default=0.01)
    parser.add_argument("--smoothness_penalize_moneyness", type=_parse_bool, default=True)
    parser.add_argument("--smoothness_penalize_maturity", type=_parse_bool, default=True)
    parser.add_argument(
        "--smoothness_weight_mode", type=str, default="inverse_sqrt", choices=["uniform", "direct", "inverse_sqrt"]
    )

    # Optuna args
    parser.add_argument("--optimize", action="store_true", help="Run Optuna hyperparameter optimization")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--storage", type=str, default="sqlite:///gnot_optuna.db", help="Optuna storage URL")
    parser.add_argument("--study_name", type=str, default="gnot_study", help="Optuna study name")
    return parser


def _create_datamodule(*, batch_size: int) -> VolSurfacePointCloudDataModule:
    return VolSurfacePointCloudDataModule(batch_size=batch_size, config=config)


def _create_model(args: argparse.Namespace, **overrides: float) -> GNOTDiffusionLightningModule:
    model_kwargs: dict[str, Any] = {
        "embed_dim": args.embed_dim,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "n_experts": args.n_experts,
        "mlp_layers": args.mlp_layers,
        "lr": args.lr,
        "use_arbitrage_loss": config.USE_ARBITRAGE_LOSS,
        "butterfly_loss_weight": args.butterfly_loss_weight,
        "calendar_loss_weight": args.calendar_loss_weight,
        "smoothness_loss_weight": args.smoothness_loss_weight,
        "smoothness_penalize_moneyness": args.smoothness_penalize_moneyness,
        "smoothness_penalize_maturity": args.smoothness_penalize_maturity,
        "smoothness_weight_mode": args.smoothness_weight_mode,
    }
    model_kwargs.update(overrides)
    return GNOTDiffusionLightningModule(**model_kwargs)


def _create_trainer(*, args: argparse.Namespace, callbacks: list[Any], for_optuna: bool) -> L.Trainer:
    trainer_kwargs: dict[str, Any] = {
        "max_epochs": args.epochs,
        "callbacks": callbacks,
        "gradient_clip_val": GRADIENT_CLIP_VAL,
        "accelerator": "auto",
    }
    if for_optuna:
        trainer_kwargs.update(
            {
                "devices": 1,
                "precision": "bf16-mixed",
                "enable_progress_bar": True,
                "logger": False,
            }
        )
    else:
        trainer_kwargs.update(
            {
                "fast_dev_run": args.fast_dev_run,
                "devices": "auto",
                "precision": "32-true",
            }
        )

    return L.Trainer(**trainer_kwargs)


def _suggest_hyperparameters(trial: optuna.trial.Trial) -> dict[str, float | int]:
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "embed_dim": trial.suggest_categorical("embed_dim", [128, 256]),
        "n_layers": trial.suggest_int("n_layers", 4, 16),
        "n_heads": trial.suggest_categorical("n_heads", [4, 8]),
        "n_experts": trial.suggest_int("n_experts", 4, 8),
        "mlp_layers": trial.suggest_int("mlp_layers", 2, 4),
    }


def objective(trial: optuna.trial.Trial, args: argparse.Namespace) -> float:
    torch.cuda.empty_cache()

    dm = _create_datamodule(batch_size=OPTUNA_BATCH_SIZE)
    model = _create_model(args, **_suggest_hyperparameters(trial))

    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    trainer = _create_trainer(args=args, callbacks=[pruning_callback], for_optuna=True)

    trainer.fit(model, dm)
    val_loss = trainer.callback_metrics["val_loss"]

    del model
    del dm
    gc.collect()

    if isinstance(val_loss, torch.Tensor):
        return val_loss.item()
    return float(val_loss)


def _run_optuna(args: argparse.Namespace) -> None:
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        storage=args.storage,
        study_name=args.study_name,
        load_if_exists=True,
    )

    print(f"Starting Optuna optimization with {args.n_trials} trials...")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


def _run_training(args: argparse.Namespace) -> None:
    dm = _create_datamodule(batch_size=args.batch_size)
    model = _create_model(args)

    if args.compile:
        model = cast("GNOTDiffusionLightningModule", torch.compile(model, mode="reduce-overhead"))

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="gnot-best-{epoch:02d}-{val_loss:.6f}",
    )
    trainer = _create_trainer(args=args, callbacks=[checkpoint_callback], for_optuna=False)
    trainer.fit(model, dm)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    torch.set_float32_matmul_precision("high")

    if args.optimize:
        _run_optuna(args)
    else:
        _run_training(args)


if __name__ == "__main__":
    main()

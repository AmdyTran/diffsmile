from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from diffsmile.config import ConditioningScalarIndex, TrainingConfigConditionalDiffusion
from diffsmile.eval.data import build_evaluation_data
from diffsmile.eval.model import load_checkpoint_model
from diffsmile.model.kernelized_evaluation import build_kernelized_slice_evaluation
from diffsmile.model.validation import build_fullgrid_prediction_buffers, run_gnot_inpaint_kernelized_trace

if TYPE_CHECKING:
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    from diffsmile.gnot_lightning import GNOTDiffusionLightningModule, VolSurfacePointCloudDataModule
    from diffsmile.model.dataset import TimeSeriesDataset
    from diffsmile.model.scheduler_rad import RegionAwareScheduler


@dataclass(slots=True)
class InpaintTraceResult:
    snapshots: dict[int, torch.Tensor]
    target_img: torch.Tensor
    mask_img: torch.Tensor
    coords: torch.Tensor
    data_mean: torch.Tensor
    data_std: torch.Tensor


@dataclass(slots=True)
class FullGridPredictionResult:
    preds_buffer: list[np.ndarray]
    gt_buffer: list[np.ndarray]
    day_indices: list[int]


@dataclass(slots=True)
class KernelizedSliceEvaluationResult:
    mean_pred: np.ndarray
    mean_true: np.ndarray
    abs_diff: np.ndarray
    signed_diff: np.ndarray
    mape: np.ndarray
    mask_reference: np.ndarray
    mask_frequency: np.ndarray
    logk_axis: np.ndarray
    ttm_days: np.ndarray
    eval_day_indices: np.ndarray
    mask_prob: np.ndarray


@dataclass(slots=True)
class EvaluationRunner:
    model: GNOTDiffusionLightningModule
    scheduler: RegionAwareScheduler
    datasets: dict[str, TimeSeriesDataset]
    scaler: StandardScaler
    data_mean: torch.Tensor
    data_std: torch.Tensor
    log_moneyness_grid: torch.Tensor
    ttm_grid: torch.Tensor
    device: torch.device
    datamodule: VolSurfacePointCloudDataModule
    val_prediction_dates: list[dt.date]
    config: TrainingConfigConditionalDiffusion = field(default_factory=TrainingConfigConditionalDiffusion)

    def __post_init__(self) -> None:
        self.model = self.model.to(self.device)
        self.log_moneyness_grid = self.log_moneyness_grid.to(self.device)
        self.ttm_grid = self.ttm_grid.to(self.device)
        self.scheduler = self.scheduler.to(self.device)

        self.model.log_moneyness_grid = self.log_moneyness_grid
        self.model.ttm_grid = self.ttm_grid
        self.model.scheduler = self.scheduler
        self.model.eval()

    @classmethod
    def from_checkpoint(  # noqa: PLR0913
        cls,
        checkpoint_path: str,
        *,
        device: torch.device | str | None = None,
        strict: bool = False,
        batch_size: int = 64,
        num_workers: int = 4,
        config: TrainingConfigConditionalDiffusion | None = None,
        scheduler: RegionAwareScheduler | None = None,
    ) -> EvaluationRunner:
        resolved_config = config if config is not None else TrainingConfigConditionalDiffusion()

        eval_data = build_evaluation_data(batch_size=batch_size, num_workers=num_workers, config=resolved_config)
        model, resolved_device = load_checkpoint_model(checkpoint_path, device=device, strict=strict)

        resolved_scheduler = scheduler if scheduler is not None else model.scheduler

        return cls(
            model=model,
            scheduler=resolved_scheduler,
            datasets={"train": eval_data.train_dataset, "val": eval_data.val_dataset},
            scaler=eval_data.scaler,
            data_mean=eval_data.data_mean,
            data_std=eval_data.data_std,
            log_moneyness_grid=eval_data.log_moneyness_grid,
            ttm_grid=eval_data.ttm_grid,
            device=resolved_device,
            datamodule=eval_data.datamodule,
            val_prediction_dates=eval_data.val_prediction_dates,
            config=resolved_config,
        )

    @property
    def train_dataset(self) -> TimeSeriesDataset:
        return self.datasets["train"]

    @property
    def val_dataset(self) -> TimeSeriesDataset:
        return self.datasets["val"]

    @property
    def val_dates(self) -> list[dt.date]:
        return list(self.val_prediction_dates)

    @property
    def raw_train_scalars(self) -> torch.Tensor:
        return self._inverse_transform_scalars(self.train_dataset.scalars)

    @property
    def raw_val_scalars(self) -> torch.Tensor:
        return self._inverse_transform_scalars(self.val_dataset.scalars)

    def _inverse_transform_scalars(self, scalars: torch.Tensor) -> torch.Tensor:
        raw_scalars = self.scaler.inverse_transform(scalars.detach().cpu().numpy())
        return torch.from_numpy(raw_scalars).float()

    @staticmethod
    def _normalize_date(value: dt.date | dt.datetime) -> dt.date:
        if isinstance(value, dt.datetime):
            return value.date()
        return value

    def val_date(self, day_idx: int) -> dt.date:
        return self.val_dates[day_idx]

    def val_index_for_date(self, date: dt.date | dt.datetime) -> int:
        normalized = self._normalize_date(date)
        for idx, current in enumerate(self.val_dates):
            if current == normalized:
                return idx
        msg = f"Validation date {normalized} not found in dataset."
        raise ValueError(msg)

    def sample(
        self,
        *,
        day_idx: int,
        batch_size: int = 10,
        num_steps: int = 500,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.sample_and_evaluate(self.val_dataset, day_idx=day_idx, batch_size=batch_size, num_steps=num_steps)

    def sample_from_checkpoint(
        self,
        day_idx: int,
        batch_size: int = 10,
        num_steps: int = 500,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sample(day_idx=day_idx, batch_size=batch_size, num_steps=num_steps)

    def inpaint_trace(
        self,
        *,
        day_idx: int,
        prob: float = 0.3,
        capture_ts: tuple[int, ...] | list[int] | None = None,
        scalars_override_raw: dict[ConditioningScalarIndex, float] | None = None,
        random_seed: int | None = None,
    ) -> InpaintTraceResult:
        if capture_ts is None:
            capture_ts = (0, 1, 5, 10, 20, 50, 100, 150, 200, 300, 400, 499)

        snapshots, target_img, mask_img, coords = run_gnot_inpaint_kernelized_trace(
            model=self.model,
            day_idx=day_idx,
            prob=prob,
            scheduler=self.scheduler,
            dataset_val=self.val_dataset,
            config=self.config,
            device=self.device,
            log_moneyness_grid=self.log_moneyness_grid,
            ttm_grid=self.ttm_grid,
            capture_ts=capture_ts,
            scalars_override_raw=scalars_override_raw,
            scaler=self.scaler,
            random_seed=random_seed,
        )
        return InpaintTraceResult(
            snapshots=snapshots,
            target_img=target_img,
            mask_img=mask_img,
            coords=coords,
            data_mean=self.data_mean,
            data_std=self.data_std,
        )

    def fullgrid_predictions(
        self,
        *,
        days_per_batch: int = 4,
        samples_per_day: int = 8,
        pin_memory: bool = True,
        progress_desc: str = "Full-grid validation",
    ) -> FullGridPredictionResult:
        preds_buffer, gt_buffer, day_indices = build_fullgrid_prediction_buffers(
            model=self.model,
            scheduler=self.scheduler,
            dataset_val=self.val_dataset,
            config=self.config,
            device=self.device,
            log_moneyness_grid=self.log_moneyness_grid,
            ttm_grid=self.ttm_grid,
            mean=self.data_mean,
            std=self.data_std,
            days_per_batch=days_per_batch,
            samples_per_day=samples_per_day,
            pin_memory=pin_memory,
            progress_desc=progress_desc,
        )
        return FullGridPredictionResult(preds_buffer=preds_buffer, gt_buffer=gt_buffer, day_indices=day_indices)

    def kernelized_slice_evaluation(
        self,
        *,
        mask_prob: float = 0.50,
        eval_day_indices: list[int] | tuple[int, ...] | None = None,
        n_days: int = 12,
        n_mc: int = 1,
        base_seed: int = 20260320,
    ) -> KernelizedSliceEvaluationResult:
        result = build_kernelized_slice_evaluation(
            model=self.model,
            scheduler=self.scheduler,
            dataset_val=self.val_dataset,
            config=self.config,
            device=self.device,
            log_moneyness_grid=self.log_moneyness_grid,
            ttm_grid=self.ttm_grid,
            scaler=self.scaler,
            mean=self.data_mean,
            std=self.data_std,
            mask_prob=mask_prob,
            eval_day_indices=eval_day_indices,
            n_days=n_days,
            n_mc=n_mc,
            base_seed=base_seed,
        )
        return KernelizedSliceEvaluationResult(
            mean_pred=result["mean_pred"],
            mean_true=result["mean_true"],
            abs_diff=result["abs_diff"],
            signed_diff=result["signed_diff"],
            mape=result["mape"],
            mask_reference=result["mask_reference"],
            mask_frequency=result["mask_frequency"],
            logk_axis=result["logk_axis"],
            ttm_days=result["ttm_days"],
            eval_day_indices=result["eval_day_indices"],
            mask_prob=result["mask_prob"],
        )

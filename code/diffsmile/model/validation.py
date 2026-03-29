from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import tqdm
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from sklearn.preprocessing import StandardScaler

    from diffsmile.config import ConditioningScalarIndex, TrainingConfigConditionalDiffusion
    from diffsmile.gnot_lightning import GNOTDiffusionLightningModule
    from diffsmile.model.dataset import TimeSeriesDataset
    from diffsmile.model.scheduler_rad import RegionAwareScheduler


def _scale_tte_mesh(tte: torch.Tensor) -> torch.Tensor:
    return torch.log1p(tte / 7.0) / math.log1p(730 / 7.0)


def run_gnot_inpaint_kernelized_trace(  # noqa: PLR0913
    model: GNOTDiffusionLightningModule,
    day_idx: int,
    prob: float,
    scheduler: RegionAwareScheduler,
    dataset_val: TimeSeriesDataset,
    config: TrainingConfigConditionalDiffusion,
    device: torch.device,
    log_moneyness_grid: torch.Tensor,
    ttm_grid: torch.Tensor,
    capture_ts: Sequence[int] = (0, 1, 5, 10, 20, 50, 100, 150, 200, 300, 400, 499),
    *,
    scalars_override_raw: dict[ConditioningScalarIndex, float] | None = None,
    scaler: StandardScaler | None = None,
    random_seed: int | None = None,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a region-aware diffusion inpainting trace with optional scalar overrides."""
    h, w = config.IMAGE_HEIGHT, config.IMAGE_WIDTH
    n, batch_size = h * w, 1

    if random_seed is not None:
        torch.manual_seed(random_seed)

    surf_sample, ema_short, ema_long, scalars_sample, next_surf = dataset_val[day_idx]

    target_img = next_surf.to(device).view(batch_size, 1, h, w)
    scalars_sample = scalars_sample.view(batch_size, -1).to(device)

    if scalars_override_raw:
        if scaler is None:
            msg = "A scaler is required for raw scalar overrides."
            raise ValueError(msg)

        if scaler.mean_ is None or scaler.scale_ is None:
            msg = "Scaler must be fitted before raw scalar overrides."
            raise ValueError(msg)

        scalars_np = scalars_sample.cpu().numpy().copy()
        mean_arr = torch.as_tensor(scaler.mean_, dtype=torch.float64).cpu().numpy()
        scale_arr = torch.as_tensor(scaler.scale_, dtype=torch.float64).cpu().numpy()
        for idx, raw_val in scalars_override_raw.items():
            scalars_np[0, idx] = (raw_val - mean_arr[idx]) / scale_arr[idx]
        scalars_sample = torch.from_numpy(scalars_np).to(device, dtype=torch.float32)

    logk_mesh, tte_mesh = torch.meshgrid(log_moneyness_grid, ttm_grid, indexing="ij")
    scaled_tte = _scale_tte_mesh(tte_mesh)
    coords = torch.stack([scaled_tte.flatten(), logk_mesh.flatten()], dim=1).unsqueeze(0).to(device)

    def prep_context(s: torch.Tensor) -> torch.Tensor:
        return s.to(device).view(batch_size, 1, h, w).permute(0, 2, 3, 1).reshape(batch_size, n, 1)

    context_values = torch.cat([prep_context(surf_sample), prep_context(ema_short), prep_context(ema_long)], dim=-1)
    mask_img = (torch.rand((batch_size, 1, h, w), device=device) < prob).float()

    t_start = torch.tensor([config.T1 - 1], device=device, dtype=torch.long)
    sig, noise_std = scheduler.get_sampling_scales(t_start, mask_img)
    x_t_img = sig * target_img + noise_std * torch.randn_like(target_img)

    capture_set = {int(t) for t in capture_ts if 0 <= int(t) < config.T1}
    snapshots: dict[int, torch.Tensor] = {}

    model.eval()
    with torch.no_grad():
        for t_idx in tqdm.tqdm(reversed(range(config.T1)), desc=f"Day {day_idx} Trace"):
            t_torch = torch.tensor([t_idx], device=device, dtype=torch.long)
            _, noise_scale_img = scheduler.get_sampling_scales(t_torch, mask_img)

            noise_pred = model.model(
                query_coords=coords,
                noisy_values=x_t_img.permute(0, 2, 3, 1).reshape(batch_size, n, 1),
                noise_scale=noise_scale_img.permute(0, 2, 3, 1).reshape(batch_size, n, 1),
                scalars=scalars_sample,
                context_coords=coords,
                context_values=context_values,
            )

            pred_eps = noise_pred.view(batch_size, h, w, 1).permute(0, 3, 1, 2)
            x_t_img = scheduler.backward_eps(x_t=x_t_img, pred_eps=pred_eps, t=t_torch, mask=mask_img)

            if t_idx in capture_set:
                snapshots[t_idx] = x_t_img.permute(0, 2, 3, 1).reshape(n, 1).cpu().clone()

    return snapshots, target_img.reshape(n, 1).cpu(), mask_img.reshape(n, 1).cpu(), coords[0].cpu()


def build_fullgrid_prediction_buffers(  # noqa: PLR0913, PLR0915
    *,
    model: GNOTDiffusionLightningModule,
    scheduler: RegionAwareScheduler,
    dataset_val: TimeSeriesDataset,
    config: TrainingConfigConditionalDiffusion,
    device: torch.device,
    log_moneyness_grid: torch.Tensor,
    ttm_grid: torch.Tensor,
    mean: float | torch.Tensor,
    std: float | torch.Tensor,
    days_per_batch: int = 4,
    samples_per_day: int = 8,
    pin_memory: bool = True,
    progress_desc: str = "Full-grid validation",
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Run full-grid RAD sampling on validation data and return buffers for downstream analysis."""
    h, w = config.IMAGE_HEIGHT, config.IMAGE_WIDTH
    n = h * w

    logk_mesh, tte_mesh = torch.meshgrid(log_moneyness_grid, ttm_grid, indexing="ij")
    scaled_tte = _scale_tte_mesh(tte_mesh)
    base_coords = torch.stack([scaled_tte.flatten(), logk_mesh.flatten()], dim=1).to(device)

    day_loader = DataLoader(dataset_val, batch_size=days_per_batch, shuffle=False, pin_memory=pin_memory)

    std_gpu = torch.as_tensor(std, device=device, dtype=torch.float32)
    mean_gpu = torch.as_tensor(mean, device=device, dtype=torch.float32)

    preds_buffer: list[np.ndarray] = []
    gt_buffer: list[np.ndarray] = []
    day_indices_buffer: list[int] = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (surf_chunk, ema_short_chunk, ema_long_chunk, scalars_chunk, next_surf_chunk) in enumerate(
            tqdm.tqdm(day_loader, desc=progress_desc)
        ):
            surf_chunk = surf_chunk.to(device, non_blocking=True)  # noqa: PLW2901
            ema_short_chunk = ema_short_chunk.to(device, non_blocking=True)  # noqa: PLW2901
            ema_long_chunk = ema_long_chunk.to(device, non_blocking=True)  # noqa: PLW2901
            scalars_chunk = scalars_chunk.to(device, non_blocking=True)  # noqa: PLW2901
            next_surf_chunk = next_surf_chunk.to(device, non_blocking=True)  # noqa: PLW2901

            current_days = surf_chunk.size(0)
            total_items = current_days * samples_per_day

            surf_batch = surf_chunk.repeat_interleave(samples_per_day, dim=0)
            ema_short_batch = ema_short_chunk.repeat_interleave(samples_per_day, dim=0)
            ema_long_batch = ema_long_chunk.repeat_interleave(samples_per_day, dim=0)
            scalars_batch = scalars_chunk.repeat_interleave(samples_per_day, dim=0)
            next_surf_batch = next_surf_chunk.repeat_interleave(samples_per_day, dim=0)

            batch_coords = base_coords.unsqueeze(0).expand(total_items, -1, -1)

            surf_flat = surf_batch.permute(0, 2, 3, 1).reshape(total_items, n, 1)
            ema_short_flat = ema_short_batch.permute(0, 2, 3, 1).reshape(total_items, n, 1)
            ema_long_flat = ema_long_batch.permute(0, 2, 3, 1).reshape(total_items, n, 1)
            context_values = torch.cat([surf_flat, ema_short_flat, ema_long_flat], dim=-1)

            mask = torch.ones((total_items, 1, h, w), device=device)
            t_start = torch.full((total_items,), config.T1 - 1, device=device, dtype=torch.long)

            initial_signal, initial_noise = scheduler.get_sampling_scales(t_start, mask)
            x_t = initial_signal * next_surf_batch + initial_noise * torch.randn_like(next_surf_batch)

            for t_idx in reversed(range(config.T1)):
                t = torch.full((total_items,), t_idx, device=device, dtype=torch.long)
                _, noise_scale_img = scheduler.get_sampling_scales(t, mask)

                noisy_values_flat = x_t.permute(0, 2, 3, 1).reshape(total_items, n, 1)
                noise_scale_flat = noise_scale_img.permute(0, 2, 3, 1).reshape(total_items, n, 1)

                noise_pred_flat = model.model(
                    query_coords=batch_coords,
                    noisy_values=noisy_values_flat,
                    noise_scale=noise_scale_flat,
                    scalars=scalars_batch,
                    context_coords=batch_coords,
                    context_values=context_values,
                )

                noise_pred_img = noise_pred_flat.view(total_items, h, w, 1).permute(0, 3, 1, 2)
                x_t = scheduler.backward_eps(x_t=x_t, pred_eps=noise_pred_img, t=t, mask=mask)

            x_t = x_t.mul(std_gpu).add(mean_gpu).exp()
            next_surf_denorm = next_surf_batch.mul(std_gpu).add(mean_gpu).exp()

            generated_flat = x_t.detach().cpu().numpy().squeeze(1)
            gt_flat = next_surf_denorm.detach().cpu().numpy().squeeze(1)

            generated_structured = generated_flat.reshape(current_days, samples_per_day, h, w)
            gt_structured = gt_flat.reshape(current_days, samples_per_day, h, w)[:, 0, :, :]

            preds_buffer.extend([generated_structured[i] for i in range(current_days)])
            gt_buffer.extend([gt_structured[i] for i in range(current_days)])

            start_day = batch_idx * days_per_batch
            day_indices_buffer.extend(range(start_day, start_day + current_days))

    return preds_buffer, gt_buffer, day_indices_buffer

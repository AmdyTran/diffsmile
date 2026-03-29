from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import tqdm

from diffsmile.gnot_lightning import GNOTDiffusionLightningModule, scale_tte_mesh
from diffsmile.helpers.evaluation import _denorm_iv
from diffsmile.helpers.evaluation_metrics import _ttm_to_days
from diffsmile.model.validation import run_gnot_inpaint_kernelized_trace

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl
    from sklearn.preprocessing import StandardScaler

    from diffsmile.config import TrainingConfigConditionalDiffusion
    from diffsmile.model.dataset import TimeSeriesDataset
    from diffsmile.model.scheduler_rad import RegionAwareScheduler


def run_gnot_inpaint_slice(  # noqa: PLR0913
    model: GNOTDiffusionLightningModule,
    df_day: pl.DataFrame,
    day_idx: int,
    prob: float,
    scheduler: RegionAwareScheduler,
    dataset_val: TimeSeriesDataset,
    config: TrainingConfigConditionalDiffusion,
    mean: torch.Tensor | float,
    std: torch.Tensor | float,
    device: torch.device,
    log_moneyness_grid: torch.Tensor,
    ttm_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    # 1. Extract and Prep Target Slice
    n_points = len(df_day)
    log_moneyness_obs = df_day["log_moneyness"].to_torch()
    iv_obs = df_day["iv"].to_torch()

    tau_obs = df_day["days_to_expiry"].to_torch()
    tau_obs = scale_tte_mesh(tau_obs)

    query_coords = torch.stack(tensors=[tau_obs, log_moneyness_obs], dim=1).to(device).float()
    iv_grid = iv_obs.float().unsqueeze(1).to(device)

    m, s = mean.item() if torch.is_tensor(mean) else mean, std.item() if torch.is_tensor(std) else std
    iv_grid_norm = (iv_grid.log() - m) / s

    mask = torch.bernoulli(torch.full(iv_grid_norm.shape, prob)).to(device)

    # 2. Extract Context History
    surf_sample, ema_short, ema_long, scalars_sample, _surf_next = dataset_val[day_idx]

    H, W = config.IMAGE_HEIGHT, config.IMAGE_WIDTH
    N = H * W
    B = 1

    logk_mesh, tte_mesh = torch.meshgrid(log_moneyness_grid, ttm_grid, indexing="ij")

    tte_mesh = scale_tte_mesh(tte_mesh)
    coords = torch.stack([tte_mesh.flatten(), logk_mesh.flatten()], dim=1).unsqueeze(0).to(device)

    surf_flat = surf_sample.to(device).view(1, 1, H, W).permute(0, 2, 3, 1).reshape(B, N, 1)
    ema_short_flat = ema_short.to(device).view(1, 1, H, W).permute(0, 2, 3, 1).reshape(B, N, 1)
    ema_long_flat = ema_long.to(device).view(1, 1, H, W).permute(0, 2, 3, 1).reshape(B, N, 1)
    context_values = torch.cat([surf_flat, ema_short_flat, ema_long_flat], dim=-1)
    scalars_sample = scalars_sample.view(1, -1).to(device)

    initial_signal, initial_noise_scale = scheduler.get_sampling_scales(torch.tensor([[config.T1 - 1]], device=device), mask)
    x_t = initial_signal * iv_grid_norm + initial_noise_scale * torch.randn_like(iv_grid_norm)

    model.eval()
    with torch.no_grad():
        for t_ in tqdm.tqdm(reversed(range(config.T1))):
            t = torch.tensor([[t_]], device=device)
            _, noise_scale = scheduler.get_sampling_scales(t, mask)

            noisy_values = x_t.view(B, n_points, 1)
            noise_scale_flat = noise_scale.reshape(B, n_points, 1)

            noise_pred = model.model(
                query_coords=query_coords,
                noisy_values=noisy_values,
                noise_scale=noise_scale_flat,
                scalars=scalars_sample,
                context_coords=coords,
                context_values=context_values,
            )
            noise_pred_img = noise_pred.view(B, n_points, 1)
            x_t = scheduler.backward_eps(x_t=x_t, pred_eps=noise_pred_img, t=t, mask=mask)

    return x_t, iv_grid_norm, mask, query_coords


def _kernelized_recon_surface_t0(  # noqa: PLR0913
    *,
    model: GNOTDiffusionLightningModule,
    scheduler: RegionAwareScheduler,
    dataset_val: TimeSeriesDataset,
    config: TrainingConfigConditionalDiffusion,
    device: torch.device,
    log_moneyness_grid: torch.Tensor,
    ttm_grid: torch.Tensor,
    scaler: StandardScaler,
    mean: float | torch.Tensor,
    std: float | torch.Tensor,
    day_idx: int,
    mask_prob: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run one kernelized RAD reconstruction at t=0 and return denormalized surfaces + mask + axes."""
    snapshots, target_cpu, mask_cpu, _coords_cpu = run_gnot_inpaint_kernelized_trace(
        model=model,
        day_idx=int(day_idx),
        prob=float(mask_prob),
        scheduler=scheduler,
        dataset_val=dataset_val,
        config=config,
        device=device,
        log_moneyness_grid=log_moneyness_grid,
        ttm_grid=ttm_grid,
        capture_ts=[0],
        scalars_override_raw=None,
        scaler=scaler,
        random_seed=int(random_seed),
    )

    if 0 not in snapshots:
        msg = f"Missing t=0 snapshot for day_idx={day_idx}"
        raise ValueError(msg)

    h, w = config.IMAGE_HEIGHT, config.IMAGE_WIDTH
    pred_surface = _denorm_iv(snapshots[0], mean, std).reshape(h, w)
    true_surface = _denorm_iv(target_cpu, mean, std).reshape(h, w)
    mask_surface = mask_cpu.reshape(h, w).detach().cpu().numpy()

    logk_axis = log_moneyness_grid.detach().cpu().numpy()
    ttm_axis = ttm_grid.detach().cpu().numpy()
    ttm_days = _ttm_to_days(ttm_axis)

    return pred_surface, true_surface, mask_surface, logk_axis, ttm_days


def build_kernelized_slice_evaluation(  # noqa: PLR0913
    *,
    model: GNOTDiffusionLightningModule,
    scheduler: RegionAwareScheduler,
    dataset_val: TimeSeriesDataset,
    config: TrainingConfigConditionalDiffusion,
    device: torch.device,
    log_moneyness_grid: torch.Tensor,
    ttm_grid: torch.Tensor,
    scaler: StandardScaler,
    mean: float | torch.Tensor,
    std: float | torch.Tensor,
    mask_prob: float = 0.50,
    eval_day_indices: Sequence[int] | None = None,
    n_days: int = 12,
    n_mc: int = 1,
    base_seed: int = 20260320,
) -> dict[str, np.ndarray]:
    """Aggregate kernelized reconstructions across validation days for a fixed masking rate."""
    if eval_day_indices is None:
        eval_day_indices = np.linspace(0, len(dataset_val) - 1, n_days, dtype=int).tolist()

    pred_accum: list[np.ndarray] = []
    true_accum: list[np.ndarray] = []
    mask_accum: list[np.ndarray] = []

    for day_idx in tqdm.tqdm(eval_day_indices, desc=f"Kernelized eval p={mask_prob:.0%}"):
        for mc_idx in range(n_mc):
            seed = int(base_seed + day_idx * 1000 + mc_idx)
            pred_surface, true_surface, mask_surface, logk_axis, ttm_days = _kernelized_recon_surface_t0(
                model=model,
                scheduler=scheduler,
                dataset_val=dataset_val,
                config=config,
                device=device,
                log_moneyness_grid=log_moneyness_grid,
                ttm_grid=ttm_grid,
                scaler=scaler,
                mean=mean,
                std=std,
                day_idx=int(day_idx),
                mask_prob=float(mask_prob),
                random_seed=seed,
            )
            pred_accum.append(pred_surface)
            true_accum.append(true_surface)
            mask_accum.append(mask_surface)

    pred_stack = np.stack(pred_accum, axis=0)  # (R, H, W)
    true_stack = np.stack(true_accum, axis=0)  # (R, H, W)
    mask_stack = np.stack(mask_accum, axis=0)  # (R, H, W), 1 = masked, 0 = retained

    mean_pred = pred_stack.mean(axis=0)
    mean_true = true_stack.mean(axis=0)

    abs_diff = np.abs(mean_pred - mean_true)
    signed_diff = mean_pred - mean_true
    mape = abs_diff / np.maximum(np.abs(mean_true), 1e-8)

    return {
        "mean_pred": mean_pred,
        "mean_true": mean_true,
        "abs_diff": abs_diff,
        "signed_diff": signed_diff,
        "mape": mape,
        "mask_reference": mask_stack[0],
        "mask_frequency": mask_stack.mean(axis=0),
        "logk_axis": logk_axis,
        "ttm_days": ttm_days,
        "eval_day_indices": np.asarray(eval_day_indices),
        "mask_prob": np.asarray([mask_prob]),
    }


def plot_kernelized_ttm_slice_grid(  # noqa: PLR0913
    results: dict[str, np.ndarray],
    *,
    n_cols: int = 4,
    ttm_indices: Sequence[int] | None = None,
    show_mask_points: bool = True,
    mask_source: Literal["reference", "frequency"] = "reference",
    mask_threshold: float = 0.5,
) -> None:
    """Grid of smiles by TTM: market vs RAD reconstruction on kernelized validation data."""
    mean_pred = results["mean_pred"]
    mean_true = results["mean_true"]
    logk_axis = results["logk_axis"]
    ttm_days = results["ttm_days"]
    mask_prob = float(results["mask_prob"][0])

    if ttm_indices is None:
        ttm_indices = list(range(len(ttm_days)))

    ttm_indices = [int(i) for i in ttm_indices if 0 <= int(i) < len(ttm_days)]
    if len(ttm_indices) == 0:
        msg = "No valid TTM indices selected."
        raise ValueError(msg)

    if show_mask_points:
        match mask_source:
            case "reference":
                mask_map = results["mask_reference"]
            case "frequency":
                mask_map = results["mask_frequency"]
            case _:
                msg = "mask_source must be 'reference' or 'frequency'"
                raise ValueError(msg)
    else:
        mask_map = None

    n_rows = math.ceil(len(ttm_indices) / n_cols)

    y_all = np.concatenate(
        [
            mean_true[:, ttm_indices].ravel(),
            mean_pred[:, ttm_indices].ravel(),
        ]
    )
    y_lo, y_hi = np.quantile(y_all, [0.01, 0.99])

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.8 * n_cols, 3.4 * n_rows),
        sharex=True,
        sharey=True,
    )

    axes_flat = [axes] if n_rows * n_cols == 1 else np.asarray(axes).reshape(-1)

    for panel_idx, t_idx in enumerate(ttm_indices):
        ax = axes_flat[panel_idx]

        ax.plot(
            logk_axis,
            mean_true[:, t_idx],
            color="#111111",
            lw=2.3,
            label="Kernelized market (mean)" if panel_idx == 0 else None,
        )
        ax.plot(
            logk_axis,
            mean_pred[:, t_idx],
            color="#1f78b4",
            lw=2.1,
            ls="--",
            label="RAD reconstruction (mean)" if panel_idx == 0 else None,
        )

        if show_mask_points and mask_map is not None:
            masked_idx = mask_map[:, t_idx] >= float(mask_threshold)
            retained_idx = ~masked_idx

            ax.scatter(
                logk_axis[retained_idx],
                mean_true[retained_idx, t_idx],
                color="limegreen",
                s=18,
                alpha=0.9,
                label="Retained (mask=0)" if panel_idx == 0 else None,
                zorder=4,
            )
            ax.scatter(
                logk_axis[masked_idx],
                mean_true[masked_idx, t_idx],
                facecolors="none",
                edgecolors="red",
                linewidths=1.0,
                s=24,
                alpha=0.95,
                label="Masked (mask=1)" if panel_idx == 0 else None,
                zorder=4,
            )

        local_mae = float(np.mean(np.abs(mean_pred[:, t_idx] - mean_true[:, t_idx])))
        ax.set_title(f"TTM={ttm_days[t_idx]:.1f}d | MAE={local_mae:.4f}")
        ax.set_ylim(float(y_lo), float(y_hi))
        ax.grid(alpha=0.25)

    for panel_idx in range(len(ttm_indices), len(axes_flat)):
        axes_flat[panel_idx].axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=True)

    fig.supxlabel("Log-Moneyness (k)")
    fig.supylabel(r"Implied Volatility ($\sigma$)")
    fig.suptitle(f"Kernelized Validation Smile Grid at Masking={100.0 * mask_prob:.0f}%", y=1.035)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])  # ty:ignore[invalid-argument-type]
    plt.show()


def plot_kernelized_mask_map(results: dict[str, np.ndarray]) -> None:
    """Visualize where masking happens on the kernelized grid."""
    logk_axis = results["logk_axis"]
    ttm_days = results["ttm_days"]
    mask_ref = results["mask_reference"]
    mask_freq = results["mask_frequency"]

    extent = [
        float(logk_axis.min()),
        float(logk_axis.max()),
        float(ttm_days.min()),
        float(ttm_days.max()),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True, sharey=True)

    im0 = axes[0].imshow(
        mask_ref.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].set_title("Reference Mask (single run)")

    im1 = axes[1].imshow(
        mask_freq.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title("Mask Frequency (across runs)")

    axes[0].set_xlabel("Log-Moneyness (k)")
    axes[1].set_xlabel("Log-Moneyness (k)")
    axes[0].set_ylabel("TTM (Days)")
    axes[0].grid(False)  # noqa: FBT003
    axes[1].grid(False)  # noqa: FBT003

    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
    cbar0.set_label("Mask State")

    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
    cbar1.set_label("P(masked)")

    fig.suptitle("Kernelized Mask Diagnostics", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_kernelized_surface_difference(results: dict[str, np.ndarray]) -> None:
    """Surface-level view of model-vs-kernelized target differences."""
    mean_pred = results["mean_pred"]
    mean_true = results["mean_true"]
    signed_diff = results["signed_diff"]
    mape = results["mape"] * 100.0
    logk_axis = results["logk_axis"]
    ttm_days = results["ttm_days"]
    mask_prob = float(results["mask_prob"][0])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharex=True, sharey=True)

    y_extent = [float(ttm_days.min()), float(ttm_days.max())]
    x_extent = [float(logk_axis.min()), float(logk_axis.max())]
    extent = [x_extent[0], x_extent[1], y_extent[0], y_extent[1]]

    v_true_lo, v_true_hi = np.quantile(np.concatenate([mean_true.ravel(), mean_pred.ravel()]), [0.01, 0.99])
    vmax_diff = float(np.quantile(np.abs(signed_diff), 0.99))

    im0 = axes[0].imshow(
        mean_true.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
        vmin=float(v_true_lo),
        vmax=float(v_true_hi),
    )
    axes[0].set_title("Kernelized Market Mean")

    im1 = axes[1].imshow(
        mean_pred.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
        vmin=float(v_true_lo),
        vmax=float(v_true_hi),
    )
    axes[1].set_title("RAD Reconstruction Mean")

    im2 = axes[2].imshow(
        signed_diff.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="RdBu_r",
        vmin=-vmax_diff,
        vmax=vmax_diff,
    )
    axes[2].set_title("Signed Error (RAD - Market)")

    for ax in axes:
        ax.set_xlabel("Log-Moneyness (k)")
        ax.grid(False)  # noqa: FBT003
    axes[0].set_ylabel("TTM (Days)")

    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
    cbar0.set_label(r"Implied Volatility ($\sigma$)")

    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
    cbar1.set_label(r"Implied Volatility ($\sigma$)")

    cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.03)
    cbar2.set_label("Implied Volatility Error")

    global_mape = float(np.mean(mape))
    fig.suptitle(
        f"Kernelized Validation Surface Comparison at Masking={100.0 * mask_prob:.0f}% | Global MAPE={global_mape:.2f}%",
        y=1.03,
    )
    plt.tight_layout()
    plt.show()

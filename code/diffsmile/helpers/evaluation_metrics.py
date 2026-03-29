from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tqdm

from diffsmile.helpers.evaluation import _denorm_iv
from diffsmile.model.validation import run_gnot_inpaint_kernelized_trace

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch
    from sklearn.preprocessing import StandardScaler

    from diffsmile.config import TrainingConfigConditionalDiffusion
    from diffsmile.gnot_lightning import GNOTDiffusionLightningModule
    from diffsmile.model.dataset import TimeSeriesDataset
    from diffsmile.model.scheduler_rad import RegionAwareScheduler


def _ttm_to_days(ttm_axis: np.ndarray) -> np.ndarray:
    ttm_axis = np.asarray(ttm_axis, dtype=float)
    # If maturities are in years, convert to day scale for thesis buckets.
    return ttm_axis * 365.0 if float(np.nanmax(ttm_axis)) < 20.0 else ttm_axis  # noqa: PLR2004


def _safe_region_mean(values: np.ndarray, region_mask: np.ndarray) -> float:
    if not np.any(region_mask):
        return float("nan")
    return float(np.mean(values[region_mask]))


def _build_surface_region_masks(  # noqa: PLR0913
    logk_axis: np.ndarray,
    ttm_days_axis: np.ndarray,
    *,
    short_days: float = 30.0,
    long_days: float = 150.0,
    atm_width: float = 0.05,
    wing_cutoff: float = 0.15,
) -> dict[str, np.ndarray]:
    k2d = np.broadcast_to(logk_axis[:, None], (len(logk_axis), len(ttm_days_axis)))
    t2d = np.broadcast_to(ttm_days_axis[None, :], (len(logk_axis), len(ttm_days_axis)))

    return {
        "short_tte": t2d < short_days,
        "mid_tte": (t2d >= short_days) & (t2d <= long_days),
        "long_tte": t2d > long_days,
        "deep_otm_puts": k2d <= -wing_cutoff,
        "atm_near_money": np.abs(k2d) <= atm_width,
        "deep_otm_calls": k2d >= wing_cutoff,
    }


def evaluate_rad_sparsity(  # noqa: PLR0913
    *,
    model: GNOTDiffusionLightningModule,
    scheduler: RegionAwareScheduler,
    dataset_val: TimeSeriesDataset,
    config: TrainingConfigConditionalDiffusion,
    mean: float | torch.Tensor,
    std: float | torch.Tensor,
    scaler: StandardScaler,
    device: torch.device,
    log_moneyness_grid: torch.Tensor,
    ttm_grid: torch.Tensor,
    mask_probs: Sequence[float],
    eval_day_indices: Sequence[int],
    base_seed: int = 1234,
    eps: float = 1e-8,
) -> tuple[pd.DataFrame, dict[float, np.ndarray]]:

    h, w = config.IMAGE_HEIGHT, config.IMAGE_WIDTH
    logk_axis = log_moneyness_grid.detach().cpu().numpy()
    ttm_axis = ttm_grid.detach().cpu().numpy()
    ttm_days_axis = _ttm_to_days(ttm_axis)

    region_masks = _build_surface_region_masks(logk_axis, ttm_days_axis)
    records: list[dict[str, float]] = []
    ape_maps_accumulator: dict[float, list[np.ndarray]] = {float(p): [] for p in mask_probs}

    for p in mask_probs:
        p = float(p)  # noqa: PLW2901
        for day_idx in tqdm.tqdm(eval_day_indices, desc=f"mask={p:.0%}"):
            run_seed = int(base_seed + round(p * 1000) * 10000 + day_idx)

            snapshots, target_cpu, _mask_cpu, _coords_cpu = run_gnot_inpaint_kernelized_trace(
                model=model,
                day_idx=int(day_idx),
                prob=p,
                scheduler=scheduler,
                dataset_val=dataset_val,
                config=config,
                device=device,
                capture_ts=[0],
                scalars_override_raw=None,
                scaler=scaler,
                random_seed=run_seed,
                log_moneyness_grid=log_moneyness_grid,
                ttm_grid=ttm_grid,
            )

            if 0 not in snapshots:
                msg = f"Missing t=0 snapshot for day_idx={day_idx}, mask={p}."
                raise ValueError(msg)

            pred_surface = _denorm_iv(snapshots[0], mean, std).reshape(h, w)
            true_surface = _denorm_iv(target_cpu, mean, std).reshape(h, w)

            abs_pct_error = np.abs(pred_surface - true_surface) / np.maximum(np.abs(true_surface), eps)
            sq_error = (pred_surface - true_surface) ** 2

            ape_maps_accumulator[p].append(abs_pct_error)

            records.append(
                {
                    "day_idx": int(day_idx),
                    "mask_prob": p,
                    "mask_pct": 100.0 * p,
                    "global_mape": 100.0 * float(np.mean(abs_pct_error)),
                    "global_mse": float(np.mean(sq_error)),
                    "short_tte_mape": 100.0 * _safe_region_mean(abs_pct_error, region_masks["short_tte"]),
                    "mid_tte_mape": 100.0 * _safe_region_mean(abs_pct_error, region_masks["mid_tte"]),
                    "long_tte_mape": 100.0 * _safe_region_mean(abs_pct_error, region_masks["long_tte"]),
                    "deep_otm_puts_mape": 100.0 * _safe_region_mean(abs_pct_error, region_masks["deep_otm_puts"]),
                    "atm_near_money_mape": 100.0 * _safe_region_mean(abs_pct_error, region_masks["atm_near_money"]),
                    "deep_otm_calls_mape": 100.0 * _safe_region_mean(abs_pct_error, region_masks["deep_otm_calls"]),
                }
            )

    metrics_df = pd.DataFrame.from_records(records).sort_values(["mask_prob", "day_idx"]).reset_index(drop=True)
    mean_ape_maps = {p: np.mean(np.stack(maps, axis=0), axis=0) for p, maps in ape_maps_accumulator.items() if len(maps) > 0}
    return metrics_df, mean_ape_maps


def build_sparsity_table(
    metrics_df: pd.DataFrame,
    mask_probs_curve: Sequence[float],
) -> pd.DataFrame:
    curve_pcts = [100.0 * float(p) for p in mask_probs_curve]
    table_df = (
        metrics_df[metrics_df["mask_pct"].isin(curve_pcts)]
        .groupby("mask_pct", as_index=False)
        .agg(
            global_mape=("global_mape", "mean"),
            short_tte=("short_tte_mape", "mean"),
            mid_tte=("mid_tte_mape", "mean"),
            long_tte=("long_tte_mape", "mean"),
            deep_otm_puts=("deep_otm_puts_mape", "mean"),
            atm_near_money=("atm_near_money_mape", "mean"),
            deep_otm_calls=("deep_otm_calls_mape", "mean"),
        )
        .sort_values("mask_pct")
        .reset_index(drop=True)
    )

    table_df["Masking Ratio"] = table_df["mask_pct"].map(lambda x: f"{x:.0f}%")
    table_df = table_df.rename(
        columns={
            "global_mape": "Global MAPE",
            "short_tte": "Short TTE (<30d)",
            "mid_tte": "Mid TTE (30-150d)",
            "long_tte": "Long TTE (>150d)",
            "deep_otm_puts": "Deep OTM Puts",
            "atm_near_money": "ATM / Near-The-Money",
            "deep_otm_calls": "Deep OTM Calls",
        }
    )

    return table_df[
        [
            "Masking Ratio",
            "Global MAPE",
            "Short TTE (<30d)",
            "Mid TTE (30-150d)",
            "Long TTE (>150d)",
            "Deep OTM Puts",
            "ATM / Near-The-Money",
            "Deep OTM Calls",
        ]
    ]

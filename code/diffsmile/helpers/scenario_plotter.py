from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from diffsmile.helpers.evaluation_metrics import _ttm_to_days

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Sequence

SCENARIO_ORDER: Final[list[str]] = ["base", "+1std", "+3std", "max", "-1std", "-3std", "min"]

COLORS: Final[dict[str, str]] = {
    "base": "#111111",
    "+1std": "#66c2a4",
    "+3std": "#2ca25f",
    "max": "#006d2c",
    "-1std": "#fc9272",
    "-3std": "#de2d26",
    "min": "#a50f15",
}


def plot_scenario_slices(
    scenario_surfaces: dict[str, np.ndarray],
    logk_grid: np.ndarray,
    ttm_grid: np.ndarray,
    n_cols: int = 4,
    y_label_x: float = 0.02,
) -> None:
    active_scenarios = [s for s in SCENARIO_ORDER if s in scenario_surfaces]
    if len(active_scenarios) < 2:  # noqa: PLR2004
        msg = "Need at least 2 scenarios in 'scenario_surfaces'."
        raise ValueError(msg)

    # Robust y-limits
    all_vals = np.concatenate([scenario_surfaces[s].ravel() for s in active_scenarios])
    ymin, ymax = np.quantile(all_vals, [0.01, 0.99])

    n_mats = len(ttm_grid)
    n_rows = math.ceil(n_mats / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 3.4 * n_rows), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for i, ttm in enumerate(ttm_grid):
        ax = axes_flat[i]
        for s in active_scenarios:
            ls = "--" if s in {"max", "min"} else "-"
            lw = 2.2 if s in {"base", "max", "min"} else 1.8

            ax.plot(
                logk_grid, scenario_surfaces[s][:, i], color=COLORS.get(s), lw=lw, ls=ls, alpha=0.95, label=s if i == 0 else None
            )

        ax.set_title(f"TTM={ttm:.2f}")
        ax.set_ylim(ymin, ymax)
        ax.grid(alpha=0.25)

    for i in range(n_mats, len(axes_flat)):
        axes_flat[i].axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(active_scenarios), bbox_to_anchor=(0.5, 1.01), frameon=True)

    fig.supxlabel("Log-Moneyness (k)")
    fig.supylabel(r"Implied Volatility ($\sigma$)", x=y_label_x)
    fig.suptitle("Implied Volatility: Scenario Analyses", y=1.03)

    plt.tight_layout(rect=(0.02, 0, 1, 0.98))
    plt.show()


def plot_global_and_maturity_curves(
    metrics_df: pd.DataFrame,
    mask_probs_curve: Sequence[float],
) -> None:
    curve_pcts = [100.0 * float(p) for p in mask_probs_curve]
    plot_df = metrics_df[metrics_df["mask_pct"].isin(curve_pcts)].copy()

    agg_global = (
        plot_df.groupby("mask_pct", as_index=False)
        .agg(global_mape_mean=("global_mape", "mean"), global_mape_std=("global_mape", "std"))
        .sort_values("mask_pct")
    )

    agg_tte = (
        plot_df.groupby("mask_pct", as_index=False)
        .agg(
            short=("short_tte_mape", "mean"),
            mid=("mid_tte_mape", "mean"),
            long=("long_tte_mape", "mean"),
        )
        .sort_values("mask_pct")
    )

    tte_long_df = agg_tte.melt(
        id_vars="mask_pct",
        value_vars=["short", "mid", "long"],
        var_name="Regime",
        value_name="MAPE",
    )

    regime_label_map = {
        "short": "Short-Term (<30d)",
        "mid": "Mid-Term (30-150d)",
        "long": "Long-Term (>150d)",
    }
    tte_long_df["Regime"] = tte_long_df["Regime"].map(regime_label_map)

    _fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    ax0 = axes[0]
    sns.lineplot(data=agg_global, x="mask_pct", y="global_mape_mean", marker="o", lw=2.5, color="#111111", ax=ax0)
    ax0.fill_between(
        agg_global["mask_pct"],
        agg_global["global_mape_mean"] - agg_global["global_mape_std"].fillna(0.0),
        agg_global["global_mape_mean"] + agg_global["global_mape_std"].fillna(0.0),
        alpha=0.18,
        color="#111111",
    )
    ax0.set_title("Plot A: Global Robustness Curve")
    ax0.set_xlabel("Percentage Masked (%)")
    ax0.set_ylabel("Global MAPE (%)")
    ax0.grid(alpha=0.25)

    ax1 = axes[1]
    sns.lineplot(
        data=tte_long_df,
        x="mask_pct",
        y="MAPE",
        hue="Regime",
        marker="o",
        lw=2.2,
        palette={
            "Short-Term (<30d)": "#d7301f",
            "Mid-Term (30-150d)": "#3182bd",
            "Long-Term (>150d)": "#31a354",
        },
        ax=ax1,
    )
    ax1.set_title("Plot B: Maturity Sensitivity")
    ax1.set_xlabel("Percentage Masked (%)")
    ax1.set_ylabel("MAPE (%)")
    ax1.grid(alpha=0.25)
    ax1.legend(title="TTE Regime")

    plt.tight_layout()
    plt.show()


def plot_spatial_error_progression(
    ape_maps: dict[float, np.ndarray],
    logk_axis: np.ndarray,
    ttm_axis: np.ndarray,
    *,
    target_mask_probs: Sequence[float] = (0.25, 0.50, 0.85),
) -> None:
    if len(ape_maps) == 0:
        msg = "ape_maps is empty. Run evaluate_rad_sparsity first."
        raise ValueError(msg)

    available = np.array(sorted(ape_maps.keys()), dtype=float)
    selected = [float(available[np.argmin(np.abs(available - p))]) for p in target_mask_probs]

    ttm_days_axis = _ttm_to_days(ttm_axis)
    maps_for_scale = [ape_maps[p] * 100.0 for p in selected]
    vmin = min(float(m.min()) for m in maps_for_scale)
    vmax = max(float(m.max()) for m in maps_for_scale)

    fig, axes = plt.subplots(1, len(selected), figsize=(5.6 * len(selected), 4.6), sharex=True, sharey=True)
    if len(selected) == 1:
        axes = np.array([axes])

    im = None
    for ax, p in zip(axes, selected, strict=True):
        mape_map_pct = ape_maps[p] * 100.0
        im = ax.imshow(
            mape_map_pct.T,
            origin="lower",
            aspect="auto",
            extent=[
                float(logk_axis.min()),
                float(logk_axis.max()),
                float(ttm_days_axis.min()),
                float(ttm_days_axis.max()),
            ],
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Mask={100.0 * p:.0f}%")
        ax.set_xlabel("Log-Moneyness (k)")
        ax.grid(False)  # noqa: FBT003

    axes[0].set_ylabel("TTM (Days)")
    if im is None:
        msg = "No spatial error maps were rendered."
        raise RuntimeError(msg)

    # Reserve room on the right for a dedicated colorbar axis to avoid overlap.
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.14, top=0.86, wspace=0.06)
    cax = fig.add_axes((0.915, 0.14, 0.012, 0.72))
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("MAPE (%)")
    fig.suptitle("Plot C: Spatial Error Progression", y=0.98)
    plt.show()


def aggregate_and_plot(  # noqa: PLR0913, PLR0915
    preds_buffer: list[np.ndarray],
    gt_buffer: list[np.ndarray],
    day_indices_buffer: list[int],
    config_indices: dict[str, int],
    day_to_date: dict[int, dt.date | dt.datetime | np.datetime64 | pd.Timestamp | str] | None = None,
    date_fmt: str = "%Y-%m-%d",
    date_tick_interval_months: int = 6,
) -> None:
    """Aggregate generated/market ATM vols and plot short/mid/long maturity curves.

    If ``day_to_date`` is provided, the x-axis is rendered as calendar dates.
    """
    all_preds = np.stack(preds_buffer, axis=0)
    all_gt = np.stack(gt_buffer, axis=0)

    if isinstance(day_indices_buffer[0], (list, range, np.ndarray)):
        all_days = np.concatenate([np.array(x) for x in day_indices_buffer])
    else:
        all_days = np.array(day_indices_buffer)

    short_idx, mid_idx, long_idx = config_indices["SHORT"], config_indices["MID"], config_indices["LONG"]
    atm_idx = config_indices["ATM"]

    pred_short = all_preds[:, :, short_idx, atm_idx]
    pred_mid = all_preds[:, :, mid_idx, atm_idx]
    pred_long = all_preds[:, :, long_idx, atm_idx]

    gt_short = all_gt[:, short_idx, atm_idx]
    gt_mid = all_gt[:, mid_idx, atm_idx]
    gt_long = all_gt[:, long_idx, atm_idx]

    _, batch_size = pred_short.shape

    days_expanded = np.repeat(all_days, batch_size)

    x_col = "Day"
    x_label = "Day Index"
    x_values_pred: np.ndarray | pd.DatetimeIndex = days_expanded
    x_values_gt: np.ndarray | pd.DatetimeIndex = all_days

    if day_to_date is not None:
        mapped_dates = [day_to_date.get(int(day)) for day in all_days]
        missing_days = [int(day) for day, date_val in zip(all_days, mapped_dates, strict=False) if date_val is None]
        if missing_days:
            unique_missing = sorted(set(missing_days))
            preview = ", ".join(str(d) for d in unique_missing[:10])
            msg = f"Missing date mapping for day indices: {preview}"
            raise ValueError(msg)

        mapped_dates_ts = pd.to_datetime(mapped_dates)
        day_to_timestamp = {int(day): ts for day, ts in zip(all_days, mapped_dates_ts, strict=False)}

        x_col = "Date"
        x_label = "Date"
        x_values_gt = mapped_dates_ts
        x_values_pred = pd.to_datetime([day_to_timestamp[int(day)] for day in days_expanded])

    df_preds = pd.DataFrame(
        {
            x_col: x_values_pred,
            "Short-Term": pred_short.ravel(),
            "Mid-Term": pred_mid.ravel(),
            "Long-Term": pred_long.ravel(),
            "Source": "Predicted",
        }
    ).melt(
        id_vars=[x_col, "Source"], value_vars=["Short-Term", "Mid-Term", "Long-Term"], var_name="Type", value_name="Volatility"
    )

    df_gt = pd.DataFrame(
        {x_col: x_values_gt, "Short-Term": gt_short, "Mid-Term": gt_mid, "Long-Term": gt_long, "Source": "Market"}
    ).melt(
        id_vars=[x_col, "Source"], value_vars=["Short-Term", "Mid-Term", "Long-Term"], var_name="Type", value_name="Volatility"
    )

    df_results = pd.concat([df_preds, df_gt], ignore_index=True)

    _, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    sns.set_theme(style="whitegrid")

    maturities = ["Short-Term", "Mid-Term", "Long-Term"]
    colors = ["tab:blue", "tab:green", "tab:red"]

    for i, (mat, color) in enumerate(zip(maturities, colors, strict=True)):
        ax = axes[i]
        df_subset = df_results[df_results["Type"] == mat].sort_values(x_col)

        sns.lineplot(
            data=df_subset[df_subset["Source"] == "Predicted"],
            x=x_col,
            y="Volatility",
            color=color,
            label=f"{mat} Prediction (95% CI)",
            ax=ax,
        )

        sns.lineplot(
            data=df_subset[df_subset["Source"] == "Market"],
            x=x_col,
            y="Volatility",
            color="black",
            linestyle="--",
            label="Market Actual",
            ax=ax,
            linewidth=2,
        )

        if x_col == "Date":
            locator = mdates.MonthLocator(interval=date_tick_interval_months)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))

        ax.set_ylabel(r"ATM Implied Volatility ($\sigma$)")
        ax.set_title(f"{mat} ATM Implied Volatility")
        ax.legend(loc="upper right")

    if x_col == "Date":
        plt.setp(axes[-1].get_xticklabels(), rotation=30, ha="right")

    plt.xlabel(x_label)
    plt.tight_layout()
    plt.show()

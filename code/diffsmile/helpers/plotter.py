from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Callable


def plot_day_surface_analysis(  # noqa: PLR0915
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    log_moneyness: torch.Tensor,
    ttm_grid: torch.Tensor,
    *,
    separate_plots: bool = False,
) -> None:
    # Standardize inputs to Numpy
    preds_np = predictions.squeeze(1).cpu().numpy() if predictions.ndim == 4 else predictions.cpu().numpy()  # noqa: PLR2004

    gt_np = ground_truth.squeeze(0).cpu().numpy() if ground_truth.ndim == 3 else ground_truth.cpu().numpy()  # noqa: PLR2004

    k_grid = log_moneyness.cpu().numpy()
    tau_grid = ttm_grid.cpu().numpy()

    # Auto-correct shapes to (Maturity, Strike)
    expected_strikes = len(k_grid)
    expected_maturities = len(tau_grid)

    if gt_np.shape[0] == expected_strikes and gt_np.shape[1] == expected_maturities:
        gt_np = gt_np.T
        preds_np = np.transpose(preds_np, (0, 2, 1))
    elif gt_np.shape[0] != expected_maturities or gt_np.shape[1] != expected_strikes:
        msg = f"Surface shape {gt_np.shape} does not match grids ({expected_strikes}, {expected_maturities})"
        raise ValueError(msg)

    sns.set_theme(style="whitegrid")

    if separate_plots:
        # Figure 1: Volatility Smile
        _, ax1 = plt.subplots(figsize=(6, 6))
    else:
        # Plotting setup
        fig = plt.figure(figsize=(18, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2])
        ax1 = fig.add_subplot(gs[0])

    mat_indices = [1, 8, min(18, expected_maturities - 1)]
    colors = ["tab:blue", "tab:green", "tab:red"]
    labels = ["Short-Term", "Mid-Term", "Long-Term"]

    for idx, color, label in zip(mat_indices, colors, labels, strict=True):
        ax1.plot(k_grid, gt_np[idx, :], color=color, linestyle="--", linewidth=2, alpha=0.6)

        mean_smile = np.mean(preds_np[:, idx, :], axis=0)
        ax1.plot(k_grid, mean_smile, color=color, label=label, linewidth=2)

        lower = np.percentile(preds_np[:, idx, :], 5, axis=0)
        upper = np.percentile(preds_np[:, idx, :], 95, axis=0)
        ax1.fill_between(k_grid, lower, upper, color=color, alpha=0.15)

    ax1.set_title("Volatility Smile (Fukasawa Check)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Log-Moneyness (k)")
    ax1.set_ylabel(r"Implied Volatility ($\sigma$)")
    ax1.legend()
    ax1.grid(visible=True, which="both", linestyle="--", linewidth=0.5)

    if separate_plots:
        plt.tight_layout()
        plt.show()

    if separate_plots:
        # Figure 2: Term Structure
        _fig2, ax2 = plt.subplots(figsize=(6, 6))
    else:
        ax2 = fig.add_subplot(gs[1])

    atm_idx = np.abs(k_grid - 0.0).argmin()

    ax2.plot(tau_grid, gt_np[:, atm_idx], color="black", linestyle="--", label="Market Actual", linewidth=2)

    mean_ts = np.mean(preds_np[:, :, atm_idx], axis=0)
    ax2.plot(tau_grid, mean_ts, color="royalblue", label="Diffusion Mean", linewidth=2)

    lower_ts = np.percentile(preds_np[:, :, atm_idx], 5, axis=0)
    upper_ts = np.percentile(preds_np[:, :, atm_idx], 95, axis=0)
    ax2.fill_between(tau_grid, lower_ts, upper_ts, color="royalblue", alpha=0.2, label="90% CI")

    ax2.set_title("ATM Term Structure (Calendar Check)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("TTM (Days)")
    ax2.set_ylabel(r"ATM Implied Volatility ($\sigma$)")
    ax2.legend()

    if separate_plots:
        plt.tight_layout()
        plt.show()

    if separate_plots:
        # Figure 3: Relative Error Heatmap
        _fig3, ax3 = plt.subplots(figsize=(7.2, 6))
    else:
        ax3 = fig.add_subplot(gs[2])

    mean_pred_surface = np.mean(preds_np, axis=0)
    relative_error = np.abs(mean_pred_surface - gt_np) / (gt_np + 1e-8)

    sns.heatmap(relative_error, ax=ax3, cmap="magma", cbar_kws={"label": r"Relative Error ($\sigma$)"})

    ax3.set_xticks(np.arange(0, len(k_grid), 4))
    ax3.set_xticklabels(np.round(k_grid[::4], 2), rotation=45)
    ax3.set_yticks(np.arange(0, len(tau_grid), 4))
    ax3.set_yticklabels(np.round(tau_grid[::4], 0).astype(int), rotation=0)

    ax3.set_title("Reconstruction Error Heatmap", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Log-Moneyness (k)")
    ax3.set_ylabel("Maturity Index")

    plt.tight_layout()
    plt.show()


def plot_comparison_analysis(  # noqa: C901, PLR0915
    predictions_main: torch.Tensor,
    predictions_alt: torch.Tensor,
    ground_truth: torch.Tensor,
    log_moneyness: torch.Tensor,
    ttm_grid: torch.Tensor,
) -> None:
    # Convert to Numpy
    # Handle (Batch, 1, T, K) -> (Batch, T, K) or (1, T, K) -> (T, K)
    p1_np = predictions_main.squeeze(1).cpu().numpy() if predictions_main.ndim == 4 else predictions_main.cpu().numpy()  # noqa: PLR2004
    p2_np = predictions_alt.squeeze(1).cpu().numpy() if predictions_alt.ndim == 4 else predictions_alt.cpu().numpy()  # noqa: PLR2004
    gt_np = ground_truth.squeeze(0).cpu().numpy() if ground_truth.ndim == 3 else ground_truth.cpu().numpy()  # noqa: PLR2004

    k_grid = log_moneyness.cpu().numpy()
    tau_grid = ttm_grid.cpu().numpy()

    exp_k = len(k_grid)
    exp_t = len(tau_grid)

    # Shape correction
    def _ensure_shape(arr: np.ndarray) -> np.ndarray:
        # Expects (Batch, T, K) or (T, K)  # noqa: ERA001
        if arr.ndim == 3:  # noqa: PLR2004
            if arr.shape[1] == exp_k and arr.shape[2] == exp_t:
                return np.transpose(arr, (0, 2, 1))
            if arr.shape[1] != exp_t or arr.shape[2] != exp_k:
                msg = f"Shape mismatch: {arr.shape} vs ({exp_t}, {exp_k})"
                raise ValueError(msg)
        elif arr.ndim == 2:  # noqa: PLR2004
            if arr.shape[0] == exp_k and arr.shape[1] == exp_t:
                return arr.T
            if arr.shape[0] != exp_t or arr.shape[1] != exp_k:
                msg = f"Shape mismatch: {arr.shape} vs ({exp_t}, {exp_k})"
                raise ValueError(msg)
        return arr

    p1_np = _ensure_shape(p1_np)
    p2_np = _ensure_shape(p2_np)

    # Ground truth is always 2D (T, K)
    if gt_np.shape[0] == exp_k and gt_np.shape[1] == exp_t:
        gt_np = gt_np.T

    # Setup Grid
    _fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    sns.set_theme(style="whitegrid")

    # Calculate errors
    mean_p1 = np.mean(p1_np, axis=0) if p1_np.ndim == 3 else p1_np  # noqa: PLR2004
    mean_p2 = np.mean(p2_np, axis=0) if p2_np.ndim == 3 else p2_np  # noqa: PLR2004

    err1 = np.abs(mean_p1 - gt_np) / (gt_np + 1e-8)
    err2 = np.abs(mean_p2 - gt_np) / (gt_np + 1e-8)

    # Helper for Smile Plotting
    def _plot_smile(ax: plt.Axes, preds: np.ndarray, title: str) -> None:
        mat_indices = [1, 8, min(18, exp_t - 1)]
        colors = ["tab:blue", "tab:green", "tab:red"]
        labels = ["Short-Term", "Mid-Term", "Long-Term"]

        has_batch = preds.ndim == 3  # noqa: PLR2004

        for idx, color, label in zip(mat_indices, colors, labels, strict=True):
            ax.plot(k_grid, gt_np[idx, :], color=color, linestyle="--", linewidth=2, alpha=0.5)

            if has_batch:
                mu = np.mean(preds[:, idx, :], axis=0)
                lower = np.percentile(preds[:, idx, :], 5, axis=0)
                upper = np.percentile(preds[:, idx, :], 95, axis=0)
                ax.fill_between(k_grid, lower, upper, color=color, alpha=0.15)
            else:
                mu = preds[idx, :]

            ax.plot(k_grid, mu, color=color, label=label, linewidth=2)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Log-Moneyness (k)")
        ax.set_ylabel(r"Implied Volatility ($\sigma$)")
        subplot_spec = ax.get_subplotspec()
        if subplot_spec is not None and subplot_spec.is_first_row():
            ax.legend(loc="upper right")
        ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)

    # Helper for Heatmap
    def _plot_heatmap(ax: plt.Axes, err_data: np.ndarray, title: str) -> None:
        sns.heatmap(err_data, ax=ax, cmap="magma", cbar_kws={"label": r"Rel. Error ($\sigma$)"})
        ax.set_xticks(np.arange(0, len(k_grid), 4))
        ax.set_xticklabels(np.round(k_grid[::4], 2), rotation=45)
        ax.set_yticks(np.arange(0, len(tau_grid), 4))
        ax.set_yticklabels(np.round(tau_grid[::4], 0).astype(int), rotation=0)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Maturity Index")

    # Row 1: Main Prediction
    _plot_smile(axes[0, 0], p1_np, "Volatility Smile (Main)")
    _plot_heatmap(axes[0, 1], err1, "Reconstruction Error (Main)")

    # Row 2: Alt Prediction
    _plot_smile(axes[1, 0], p2_np, "Volatility Smile (Alternative)")
    _plot_heatmap(axes[1, 1], err2, "Reconstruction Error (Alternative)")

    plt.show()


def calculate_and_plot_physics_metrics(  # noqa: C901, PLR0912, PLR0913, PLR0915
    preds_buffer: list[np.ndarray],
    gt_buffer: list[np.ndarray],
    day_indices_buffer: list[int],
    arb_loss_fn: Callable[[torch.Tensor], torch.Tensor],
    butterfly_loss_fn: Callable[[torch.Tensor], torch.Tensor],
    device: str = "cpu",
    *,
    day_to_date: dict[int, dt.date | dt.datetime | np.datetime64 | pd.Timestamp | str] | None = None,
    date_fmt: str = "%Y-%m-%d",
    date_tick_interval_months: int = 6,
    separate_plots: bool = False,
) -> None:
    preds_np = np.stack(preds_buffer, axis=0)
    gt_np = np.stack(gt_buffer, axis=0)

    n_days, batch_size, h, w = preds_np.shape
    preds_tensor = torch.from_numpy(preds_np).float().to(device).view(-1, 1, h, w)
    gt_tensor = torch.from_numpy(gt_np).float().to(device).view(n_days, 1, h, w)

    gt_expanded = gt_tensor.repeat_interleave(batch_size, dim=0)  # 2. Calculate Metrics
    mse_scores = torch.mean((preds_tensor - gt_expanded) ** 2, dim=(1, 2, 3)).cpu().numpy()

    # calculate loss to plot
    pred_cal_loss = arb_loss_fn(preds_tensor)
    pred_but_loss = butterfly_loss_fn(preds_tensor)

    if pred_cal_loss.ndim == 0:
        pred_cal_loss = pred_cal_loss.expand(preds_tensor.size(0))
    if pred_but_loss.ndim == 0:
        pred_but_loss = pred_but_loss.expand(preds_tensor.size(0))

    pred_cal_np = pred_cal_loss.detach().cpu().numpy()
    pred_but_np = pred_but_loss.detach().cpu().numpy()

    gt_cal_loss = arb_loss_fn(gt_tensor)
    gt_but_loss = butterfly_loss_fn(gt_tensor)

    gt_cal_np = gt_cal_loss.detach().cpu().numpy()
    gt_but_np = gt_but_loss.detach().cpu().numpy()

    if isinstance(day_indices_buffer[0], (list, range, np.ndarray)):
        flat_days = np.concatenate([np.array(x) for x in day_indices_buffer])
    else:
        flat_days = np.array(day_indices_buffer)

    days_pred_expanded = np.repeat(flat_days, batch_size)

    x_col = "Day"
    x_values_pred: np.ndarray | pd.DatetimeIndex = days_pred_expanded
    x_values_gt: np.ndarray | pd.DatetimeIndex = flat_days

    if day_to_date is not None:
        mapped_dates = [day_to_date.get(int(day)) for day in flat_days]
        missing_days = [int(day) for day, date_val in zip(flat_days, mapped_dates, strict=False) if date_val is None]
        if missing_days:
            unique_missing = sorted(set(missing_days))
            preview = ", ".join(str(d) for d in unique_missing[:10])
            msg = f"Missing date mapping for day indices: {preview}"
            raise ValueError(msg)

        mapped_dates_ts = pd.to_datetime(mapped_dates)
        day_to_timestamp = {int(day): ts for day, ts in zip(flat_days, mapped_dates_ts, strict=False)}

        x_col = "Date"
        x_values_gt = mapped_dates_ts
        x_values_pred = pd.to_datetime([day_to_timestamp[int(day)] for day in days_pred_expanded])

    df_preds = pd.DataFrame(
        {
            x_col: x_values_pred,
            "Type": "Prediction",
            "MSE": mse_scores,
            "Calendar Loss": pred_cal_np,
            "Butterfly Loss": pred_but_np,
        }
    )

    df_gt = pd.DataFrame(
        {
            x_col: x_values_gt,
            "Type": "Market",
            "MSE": np.zeros(len(flat_days), dtype=np.float32),
            "Calendar Loss": gt_cal_np,
            "Butterfly Loss": gt_but_np,
        }
    )

    df_metrics = pd.concat([df_preds, df_gt], ignore_index=True)

    x_label = "Date" if x_col == "Date" else "Day Index"

    def _style_time_axis(ax: plt.Axes) -> None:
        ax.set_xlabel(x_label)
        if x_col == "Date":
            locator = mdates.MonthLocator(interval=date_tick_interval_months)
            formatter = mdates.DateFormatter(date_fmt)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    sns.set_theme(style="whitegrid")

    if separate_plots:
        _fig1, ax1 = plt.subplots(figsize=(8, 6))
    else:
        _fig1, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True)
        ax1 = axes[0]

    # Plot 1: MSE (Predictions Only)
    sns.lineplot(
        data=df_metrics[df_metrics["Type"] == "Prediction"],
        x=x_col,
        y="MSE",
        ax=ax1,
        color="tab:blue",
        label="Model Reconstruction MSE",
    )
    ax1.set_title("Reconstruction Error (MSE) over Time")
    ax1.set_ylabel("Mean Squared Error (Vol Points²)")

    if separate_plots:
        _style_time_axis(ax1)
        plt.tight_layout()
        plt.show()
        _fig2, ax2 = plt.subplots(figsize=(8, 6))
    else:
        ax2 = axes[1]

    # Plot 2: Calendar Arbitrage
    sns.lineplot(
        data=df_metrics[df_metrics["Type"] == "Prediction"],
        x=x_col,
        y="Calendar Loss",
        ax=ax2,
        color="tab:orange",
        label="Model Mean + 95% CI",
    )
    sns.scatterplot(
        data=df_metrics[df_metrics["Type"] == "Market"],
        x=x_col,
        y="Calendar Loss",
        ax=ax2,
        color="black",
        alpha=0.6,
        s=15,
        label="Market Actual",
        zorder=5,
    )
    ax2.set_title("Calendar Arbitrage Violation Check")
    ax2.set_ylabel("Calendar Loss Penalty")
    ax2.set_yscale("symlog", linthresh=1e-5)

    if separate_plots:
        _style_time_axis(ax2)
        plt.tight_layout()
        plt.show()
        _, ax3 = plt.subplots(figsize=(8, 6))
    else:
        ax3 = axes[2]

    # Plot 3: Butterfly Arbitrage
    sns.lineplot(
        data=df_metrics[df_metrics["Type"] == "Prediction"],
        x=x_col,
        y="Butterfly Loss",
        ax=ax3,
        color="tab:green",
        label="Model Mean + 95% CI",
    )
    sns.scatterplot(
        data=df_metrics[df_metrics["Type"] == "Market"],
        x=x_col,
        y="Butterfly Loss",
        ax=ax3,
        color="black",
        alpha=0.6,
        s=15,
        label="Market Actual",
        zorder=5,
    )
    ax3.set_title("Butterfly Arbitrage Violation Check")
    ax3.set_ylabel("Butterfly Loss Penalty")
    ax3.set_yscale("symlog", linthresh=1e-5)

    if separate_plots:
        _style_time_axis(ax3)
    else:
        _style_time_axis(ax3)

    plt.tight_layout()
    plt.show()


def plot_surface_comparison(  # noqa: PLR0913
    ground_truth: torch.Tensor,
    mask: torch.Tensor,
    model_main: torch.Tensor,
    model_alt: torch.Tensor,
    k_grid: torch.Tensor,
    tau_grid: torch.Tensor,
) -> None:
    gt_np = ground_truth.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    m1_np = model_main.detach().cpu().numpy()
    m2_np = model_alt.detach().cpu().numpy()

    k = k_grid.detach().cpu().numpy()
    t = tau_grid.detach().cpu().numpy()

    K_mesh, T_mesh = np.meshgrid(k, t)

    # Ensure orientation matches meshgrid (Maturity x Strike)
    if gt_np.shape != K_mesh.shape:
        gt_np = gt_np.T
        mask_np = mask_np.T
        m1_np = m1_np.T
        m2_np = m2_np.T

    masked_view = gt_np.copy()
    masked_view[mask_np < 0.5] = np.nan  # 0 is observed, 1 is masked and noised till T1  # noqa: PLR2004

    z_min = np.nanmin(gt_np)
    z_max = np.nanmax(gt_np)

    fig = plt.figure(figsize=(16, 12))

    plots = [
        (gt_np, "Ground Truth"),
        (masked_view, "Observed Data (Masked)"),
        (m1_np, "Model Main Prediction"),
        (m2_np, "Model Alternative Prediction"),
    ]

    for i, (data, title) in enumerate(plots, 1):
        ax = fig.add_subplot(2, 2, i, projection="3d")

        surf = ax.plot_surface(
            K_mesh,
            T_mesh,
            data,
            cmap="plasma",
            edgecolor="none",
            alpha=0.9,
            vmin=z_min,
            vmax=z_max,
        )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Log-Moneyness (k)")
        ax.set_ylabel("TTM (Days)")
        ax.set_zlabel(r"Implied Volatility ($\sigma$)")

        ax.set_zlim(z_min, z_max)
        ax.view_init(elev=30, azim=-45)

        if i == 2:  # Only add colorbar once or for the sparse plot  # noqa: PLR2004
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    plt.tight_layout()
    plt.show()

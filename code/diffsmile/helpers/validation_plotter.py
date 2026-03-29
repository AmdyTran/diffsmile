from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from diffsmile.helpers.evaluation import _denorm_iv, _extract_coords

if TYPE_CHECKING:
    import torch
    from mpl_toolkits.mplot3d.axes3d import Axes3D


def _available_and_missing_timesteps(
    snapshots: dict[int, torch.Tensor],
    timesteps: list[int],
) -> tuple[list[int], list[int]]:
    # Preserve the caller's requested order so panel order matches expectation.
    available = [int(t) for t in timesteps if int(t) in snapshots]
    missing = [int(t) for t in timesteps if int(t) not in snapshots]
    return available, missing


def plot_3d_surface(smoothed_tensor: torch.Tensor, k_grid: torch.Tensor, tau_grid: torch.Tensor, title: str | None) -> None:
    K, T = np.meshgrid(k_grid.cpu().numpy(), tau_grid.cpu().numpy())
    Z = smoothed_tensor.detach().cpu().numpy().T

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(K, T, Z, cmap="plasma", edgecolor="none", alpha=0.9)

    ax.set_xlabel("Log-Moneyness (k)")
    ax.set_ylabel("TTM (Days)")
    ax.set_zlabel(r"Implied Volatility ($\sigma$)")
    if title:
        ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()


def _surface_on_grid(  # noqa: PLR0913
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grid_n_x: int = 80,
    grid_n_y: int = 80,
    *,
    prefer_rect_grid: bool = True,
    allow_interpolation: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if prefer_rect_grid:
        x_r = np.round(x, 8)
        y_r = np.round(y, 8)
        x_u = np.unique(x_r)
        y_u = np.unique(y_r)

        if x_u.size * y_u.size == z.size:
            x_idx = {v: i for i, v in enumerate(x_u)}
            y_idx = {v: i for i, v in enumerate(y_u)}
            z_grid = np.full((y_u.size, x_u.size), np.nan, dtype=float)

            for xx, yy, zz in zip(x_r, y_r, z, strict=False):
                z_grid[y_idx[yy], x_idx[xx]] = zz

            if not np.isnan(z_grid).any():
                x_grid, y_grid = np.meshgrid(x_u, y_u)
                return x_grid, y_grid, z_grid

    if not allow_interpolation:
        msg = (
            "Grid reshape failed. For kernelized dataset_val this should be a full rectangular grid. "
            "Set allow_interpolation=True only for sparse/irregular point clouds."
        )
        raise ValueError(msg)

    x_lin = np.linspace(x.min(), x.max(), grid_n_x)
    y_lin = np.linspace(y.min(), y.max(), grid_n_y)
    x_grid, y_grid = np.meshgrid(x_lin, y_lin)
    points = np.column_stack([x, y])

    z_grid = griddata(points, z, (x_grid, y_grid), method="linear")
    if np.isnan(z_grid).any():
        z_nn = griddata(points, z, (x_grid, y_grid), method="nearest")
        z_grid = np.where(np.isnan(z_grid), z_nn, z_grid)
    return x_grid, y_grid, z_grid


def _plot_surface_fullres(  # noqa: PLR0913
    ax: Axes3D,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    cmap: str = "plasma",
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float = 0.95,
) -> None:
    return ax.plot_surface(
        x,
        y,
        z,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolor="none",
        alpha=alpha,
        rcount=z.shape[0],
        ccount=z.shape[1],
        antialiased=True,
    )


def get_shared_limits(  # noqa: PLR0913
    snapshots: dict[int, torch.Tensor],
    query_coords: torch.Tensor,
    mean: float | torch.Tensor,
    std: float | torch.Tensor,
    timesteps: list[int] | None = None,
    *,
    robust: bool = False,
) -> dict[str, tuple[float, float]]:
    k, _, tau = _extract_coords(query_coords)

    show_ts = sorted(snapshots.keys()) if timesteps is None else [t for t in timesteps if t in snapshots]
    if not show_ts:
        msg = "No valid timesteps found for shared limits."
        raise ValueError(msg)

    z_all = np.concatenate([_denorm_iv(snapshots[t], mean, std) for t in show_ts])

    if robust:
        z_min, z_max = np.quantile(z_all, [0.01, 0.99])
    else:
        z_min, z_max = float(z_all.min()), float(z_all.max())

    return {
        "x": (float(np.min(tau)), float(np.max(tau))),
        "y": (float(np.max(k)), float(np.min(k))),
        "z": (float(z_min), float(z_max)),
    }


def plot_phase1_mask_reference(  # noqa: PLR0913
    snapshots: dict[int, torch.Tensor],
    query_coords: torch.Tensor,
    mask: torch.Tensor,
    mean: float | torch.Tensor,
    std: float | torch.Tensor,
    t_ref: int | None = None,
    axis_limits: dict[str, tuple[float, float]] | None = None,
) -> plt.Figure:
    if not snapshots:
        msg = "snapshots is empty."
        raise ValueError(msg)
    if t_ref is None:
        t_ref = 0 if 0 in snapshots else min(snapshots.keys())

    k, _, tau = _extract_coords(query_coords)
    z = _denorm_iv(snapshots[t_ref], mean, std)
    mask_np = mask.reshape(-1).detach().cpu().numpy() > 0.5  # noqa: PLR2004
    ctx_np = ~mask_np

    if axis_limits is None:
        axis_limits = get_shared_limits(
            snapshots=snapshots,
            query_coords=query_coords,
            mean=mean,
            std=std,
            timesteps=[t_ref],
            robust=False,
        )

    x, y, z_grid = _surface_on_grid(tau, k, z, allow_interpolation=False)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    _plot_surface_fullres(
        ax,
        x,
        y,
        z_grid,
        cmap="plasma",
        vmin=axis_limits["z"][0],
        vmax=axis_limits["z"][1],
        alpha=0.93,
    )

    ax.scatter(tau[ctx_np], k[ctx_np], z[ctx_np], c="limegreen", s=22, alpha=0.85)
    ax.scatter(tau[mask_np], k[mask_np], z[mask_np], facecolors="none", edgecolors="red", s=34, linewidths=1.0)

    ax.set_title(f"Mask diagnostic at t={t_ref}")
    ax.set_xlabel("TTM (Days)")
    ax.set_ylabel("Log-Moneyness (k)")
    ax.set_zlabel(r"Implied Volatility ($\sigma$)")
    ax.view_init(elev=22, azim=-130)
    ax.set_xlim(*axis_limits["x"])
    ax.set_ylim(*axis_limits["y"])
    ax.set_zlim(*axis_limits["z"])
    ax.set_box_aspect((1.3, 1.0, 0.8))

    plt.tight_layout()
    return fig


def plot_phase1_surface_progress(  # noqa: PLR0913
    snapshots: dict[int, torch.Tensor],
    query_coords: torch.Tensor,
    mean: float | torch.Tensor,
    std: float | torch.Tensor,
    timesteps: list[int],
    ncols: int = 4,
    axis_limits: dict[str, tuple[float, float]] | None = None,
) -> plt.Figure:
    show_ts, missing_ts = _available_and_missing_timesteps(snapshots, timesteps)
    if not show_ts:
        msg = "No requested timesteps found in snapshots."
        raise ValueError(msg)

    if missing_ts:
        warnings.warn(
            (
                "Requested timesteps were not found in snapshots and were skipped: "
                f"{missing_ts}. "
                "Pass capture_ts=timesteps when calling inpaint_trace_from_checkpoint if you need them captured."
            ),
            stacklevel=2,
        )

    k, _, tau = _extract_coords(query_coords)

    if axis_limits is None:
        axis_limits = get_shared_limits(
            snapshots=snapshots,
            query_coords=query_coords,
            mean=mean,
            std=std,
            timesteps=show_ts,
            robust=True,
        )

    nrows = math.ceil(len(show_ts) / ncols)
    fig = plt.figure(figsize=(5.2 * ncols, 4.2 * nrows), layout=None)

    for i, t in enumerate(show_ts, start=1):
        ax = fig.add_subplot(nrows, ncols, i, projection="3d")
        z = _denorm_iv(snapshots[t], mean, std)
        x, y, z_grid = _surface_on_grid(tau, k, z, allow_interpolation=False)

        _plot_surface_fullres(
            ax,
            x,
            y,
            z_grid,
            cmap="plasma",
            vmin=axis_limits["z"][0],
            vmax=axis_limits["z"][1],
            alpha=0.95,
        )
        ax.set_title(f"t={t}")
        ax.set_xlabel("TTM (Days)", labelpad=8)
        ax.set_ylabel("Log-Moneyness (k)", labelpad=4)
        ax.set_zlabel(r"Implied Volatility ($\sigma$)", labelpad=8)
        ax.view_init(elev=22, azim=-130)
        ax.set_xlim(*axis_limits["x"])
        ax.set_ylim(*axis_limits["y"])
        ax.set_zlim(*axis_limits["z"])
        ax.set_box_aspect((1.3, 1.0, 0.8))

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.06, top=0.95, wspace=0.08, hspace=0.16)
    return fig


def plot_phase1_surface_progress_with_mask(  # noqa: PLR0913
    snapshots: dict[int, torch.Tensor],
    query_coords: torch.Tensor,
    mask: torch.Tensor,
    mean: float | torch.Tensor,
    std: float | torch.Tensor,
    timesteps: list[int],
    ncols: int = 4,
    *,
    show_points: Literal["context", "masked", "both"] = "context",
    axis_limits: dict[str, tuple[float, float]] | None = None,
) -> plt.Figure:
    show_ts, missing_ts = _available_and_missing_timesteps(snapshots, timesteps)
    if not show_ts:
        msg = "No requested timesteps found in snapshots."
        raise ValueError(msg)

    if missing_ts:
        warnings.warn(
            (
                "Requested timesteps were not found in snapshots and were skipped: "
                f"{missing_ts}. "
                "Pass capture_ts=timesteps when calling inpaint_trace_from_checkpoint if you need them captured."
            ),
            stacklevel=2,
        )

    k, _, tau = _extract_coords(query_coords)
    mask_np = mask.reshape(-1).detach().cpu().numpy() > 0.5  # noqa: PLR2004
    ctx_np = ~mask_np

    if show_points not in {"context", "masked", "both"}:
        msg = "show_points must be one of: 'context', 'masked', 'both'"
        raise ValueError(msg)

    if axis_limits is None:
        axis_limits = get_shared_limits(
            snapshots=snapshots,
            query_coords=query_coords,
            mean=mean,
            std=std,
            timesteps=show_ts,
            robust=True,
        )

    nrows = math.ceil(len(show_ts) / ncols)
    fig = plt.figure(figsize=(5.2 * ncols, 4.2 * nrows), layout=None)

    for i, t in enumerate(show_ts, start=1):
        ax = fig.add_subplot(nrows, ncols, i, projection="3d")
        z = _denorm_iv(snapshots[t], mean, std)
        x, y, z_grid = _surface_on_grid(tau, k, z, allow_interpolation=False)

        _plot_surface_fullres(
            ax,
            x,
            y,
            z_grid,
            cmap="plasma",
            vmin=axis_limits["z"][0],
            vmax=axis_limits["z"][1],
            alpha=0.95,
        )

        if show_points in {"context", "both"}:
            ax.scatter(tau[mask_np], k[mask_np], z[mask_np], facecolors="none", edgecolors="red", s=24, linewidths=0.9)

        if show_points in {"masked", "both"}:
            ax.scatter(tau[ctx_np], k[ctx_np], z[ctx_np], c="limegreen", s=22, alpha=0.85)

        ax.set_title(f"t={t}")
        ax.set_xlabel("TTM (Days)", labelpad=8)
        ax.set_ylabel("Log-Moneyness (k)", labelpad=4)
        ax.set_zlabel(r"Implied Volatility ($\sigma$)", labelpad=8)
        ax.view_init(elev=22, azim=-130)

        ax.set_xlim(*axis_limits["x"])
        ax.set_ylim(*axis_limits["y"])
        ax.set_zlim(*axis_limits["z"])
        ax.set_box_aspect((1.3, 1.0, 0.8))

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.06, top=0.95, wspace=0.08, hspace=0.16)
    return fig

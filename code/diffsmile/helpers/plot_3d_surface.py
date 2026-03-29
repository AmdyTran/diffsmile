from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def plot_3d_iv_surface(df: pl.DataFrame) -> None:
    maturities = df["maturity_days"].to_numpy()
    strikes = np.array(df.select(pl.exclude("maturity_days")).columns, dtype=float)
    df_pivot = df.select(pl.exclude("maturity_days"))

    # Create meshgrid
    X, Y = np.meshgrid(strikes, maturities)
    Z = df_pivot.to_numpy()

    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)

    # Labels
    ax.set_xlabel("Strike")
    ax.set_ylabel("TTM (Days)")
    ax.set_zlabel(r"Implied Volatility ($\sigma$)")
    ax.set_title("3D Implied Volatility Surface")

    # Colorbar
    fig.colorbar(surf, ax=ax, label=r"Implied Volatility ($\sigma$)")

    plt.tight_layout()
    plt.show()


def plot_iv_ribbon_surface(df: pl.DataFrame) -> None:
    maturities = df["maturity_days"].to_numpy()
    strike_cols = [col for col in df.columns if col != "maturity_days"]
    strikes = np.array(sorted([float(k) for k in strike_cols]))

    df_pivot = df.select(sorted(strike_cols, key=float))
    Z = df_pivot.to_numpy()

    Z_min, Z_max = np.nanmin(Z), np.nanmax(Z)

    if not np.isfinite(Z_min) or not np.isfinite(Z_max):
        print("Error: implied volatility surface contains only NaN or infinite values. Cannot plot.")
        return

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.cm.viridis  # ty:ignore[unresolved-attribute]
    norm = plt.Normalize(Z_min, Z_max)

    X_mesh, Y_mesh = np.meshgrid(strikes, maturities)
    Z_masked = np.ma.masked_invalid(Z)
    ax.plot_surface(X_mesh, Y_mesh, Z_masked, cmap=cmap, alpha=0.6, rstride=1, cstride=1)

    max_maturity_val = maturities.max() if len(maturities) > 0 else 0
    maturity_indices = np.linspace(0, len(maturities) - 1, 6, dtype=int)

    for i in maturity_indices:
        maturity_line = Z[i, :]
        valid_mask = ~np.isnan(maturity_line)

        if not valid_mask.any():
            continue

        line_color = cmap(norm(np.nanmean(maturity_line)))

        ax.plot(
            strikes[valid_mask],
            [max_maturity_val] * valid_mask.sum(),
            maturity_line[valid_mask],
            color=line_color,
            linestyle="-",
            linewidth=2.5,
            alpha=1.0,
        )

        last_valid_idx = np.where(valid_mask)[0][-1] if valid_mask.any() else -1
        if last_valid_idx >= 0:
            ax.text(
                strikes[last_valid_idx] * 1.02,
                max_maturity_val,
                maturity_line[last_valid_idx],
                f"{maturities[i]}d",
                color=line_color,
                fontsize=9,
                horizontalalignment="left",
            )

    min_strike_val = strikes.min() if len(strikes) > 0 else 0
    representative_strikes = np.linspace(strikes.min(), strikes.max(), 5).round().astype(int)

    for strike_val_ in representative_strikes:
        strike_idx = np.argmin(np.abs(strikes - strike_val_))
        _strike_val = strikes[strike_idx]

        term_structure_line = Z[:, strike_idx]
        valid_mask = ~np.isnan(term_structure_line)

        if not valid_mask.any():
            continue

        line_color = cmap(norm(np.nanmean(term_structure_line)))

        ax.plot(
            [min_strike_val] * valid_mask.sum(),
            maturities[valid_mask],
            term_structure_line[valid_mask],
            color=line_color,
            linestyle="--",
            linewidth=2,
            alpha=0.8,
        )

    ax.set_xlabel("Strike", fontsize=11)
    ax.set_ylabel("TTM (Days)", fontsize=11)
    ax.set_zlabel(r"Implied Volatility ($\sigma$)", fontsize=11)
    ax.set_title("Implied Volatility Surface: Skew and Term Structure Projections", fontsize=13)

    ax.set_xlim(min_strike_val, strikes.max())
    ax.set_ylim(maturities.min(), max_maturity_val)
    ax.set_zlim(Z_min * 0.9, Z_max * 1.1)

    ax.view_init(elev=25, azim=-135)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.6, aspect=10, label="Implied Volatility")

    plt.tight_layout()
    plt.show()

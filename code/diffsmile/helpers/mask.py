from __future__ import annotations

from typing import Annotated, Literal

import torch


def generate_structured_masks(  # noqa: PLR0913
    batch_size: int,
    moneyness_grid: torch.Tensor,
    maturities: torch.Tensor,
    device: torch.device | Literal["cpu", "cuda"] = "cpu",
    start_width: float = 0.4,  # Max Moneyness (±) visible at Shortest TTE
    end_width: float = 0.1,  # Max Moneyness (±) visible at Longest TTE
    row_prob: float = 0.05,  # Random missing strikes
    col_prob: float = 0.0,  # Random missing maturities
    safe_atm_width: float = 0.05,  # Always keep +/- 5% ATM clean
    *,
    flip: Annotated[bool, "column layout has changed in some code"] = False,
) -> torch.Tensor:
    """Generate masks with a 'Funnel' shape: Wide liquidity at short TTE, Narrow at long TTE.
    0.0 = Missing/Unknown (Inpaint).
    1.0 = Known Data (Keep).
    """  # noqa: D205
    H = len(maturities) if not flip else len(moneyness_grid)
    W = len(moneyness_grid) if not flip else len(maturities)

    masks = torch.zeros(batch_size, 1, H, W, device=device)

    if not flip:  # H=Maturity, W=Moneyness
        T_grid = maturities.view(1, 1, H, 1)
        K_grid = moneyness_grid.view(1, 1, 1, W)
    else:  # H=Moneyness, W=Maturity (Your preferred setup)
        T_grid = maturities.view(1, 1, 1, W)
        K_grid = moneyness_grid.view(1, 1, H, 1)

    # Normalize T from 0.0 to 1.0 based on the grid range
    T_min, T_max = maturities.min(), maturities.max()
    t_factor = (T_grid - T_min) / (T_max - T_min + 1e-6)

    # Calculate allowed width for every maturity point
    # If start_width=0.4 and end_width=0.1, this shrinks as T increases.
    current_limit = start_width + (end_width - start_width) * t_factor

    # Create the Cone/Funnel Mask
    geo_mask = (K_grid.abs() > current_limit).float()
    masks = torch.maximum(masks, geo_mask)

    # Random Structural Noise
    if row_prob > 0:
        random_rows = torch.rand(batch_size, H, device=device)
        row_mask = (random_rows < row_prob).float().view(batch_size, 1, H, 1)
        masks = torch.maximum(masks, row_mask)

    if col_prob > 0:
        random_cols = torch.rand(batch_size, W, device=device)
        col_mask = (random_cols < col_prob).float().view(batch_size, 1, 1, W)
        masks = torch.maximum(masks, col_mask)

    # Safety: Always protect the ATM region
    safe_zone = K_grid.abs() < safe_atm_width
    masks.masked_fill_(safe_zone, 0.0)

    return masks

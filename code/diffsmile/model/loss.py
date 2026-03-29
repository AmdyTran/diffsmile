from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, override

import torch
from jaxtyping import Float
from torch import nn

if TYPE_CHECKING:
    from jaxtyping import Float


class CalendarArbitrageLoss(nn.Module):
    maturities: Float[torch.Tensor, "w"]  # noqa: F821, UP037

    def __init__(self, maturities: Float[torch.Tensor, "w"]) -> None:  # noqa: F821, UP037
        super().__init__()
        self.maturities = maturities

    @override
    def forward(self, surface: Float[torch.Tensor, "b 1 h w"]) -> Float[torch.Tensor, "b"]:  # noqa: F821, UP037
        """Checks Calendar Arbitrage directly on the grid.
        Based on Lemma 2.1: Total Variance must be non-decreasing for fixed log-forward moneyness.
        """  # noqa: D205, D401
        # Following Gatheral: w = sigma^2 * T
        T_broad = self.maturities.view(1, 1, 1, -1)
        w = (surface**2) * T_broad

        # \foreach t: w(t+1) >= w(t)
        w_curr = w[:, :, :, :-1]  # Steps 0 to N-1
        w_next = w[:, :, :, 1:]  # Steps 1 to N

        # for dt term
        T_curr = self.maturities[:-1].view(1, 1, 1, -1)
        T_next = self.maturities[1:].view(1, 1, 1, -1)
        dt = T_next - T_curr

        # Loss is positive if dw/dt < 0 (Violation)
        # Normalize by dt to get proper derivative approximation
        diff = (w_curr - w_next) / dt
        return torch.relu(diff).sum(dim=(-1, -2, -3))


def black_scholes_call_normalized(
    log_moneyness: Float[torch.Tensor, ...], w: Float[torch.Tensor, ...]
) -> Float[torch.Tensor, ...]:
    sqrt_w = torch.sqrt(w.clamp(min=1e-8))

    d1 = (-log_moneyness + w / 2) / sqrt_w
    d2 = d1 - sqrt_w

    normal = torch.distributions.Normal(0.0, 1.0)
    cdf_d1 = normal.cdf(d1)
    cdf_d2 = normal.cdf(d2)

    k_exp = torch.exp(log_moneyness)
    return cdf_d1 - k_exp * cdf_d2


class CalendarArbitrageMonetaryLoss(nn.Module):
    maturities: Float[torch.Tensor, "w"]  # noqa: F821, UP037
    log_moneyness: Float[torch.Tensor, "h"]  # noqa: F821, UP037

    def __init__(self, maturities: Float[torch.Tensor, "w"], log_moneyness: Float[torch.Tensor, "h"]) -> None:  # noqa: F821, UP037
        super().__init__()
        self.maturities = maturities
        self.register_buffer("log_moneyness", log_moneyness.view(1, 1, -1, 1))

    @override
    def forward(self, surface: Float[torch.Tensor, "b 1 h w"], spot_price: float = 100.0) -> Float[torch.Tensor, "b"]:  # noqa: F821, UP037
        T_broad = self.maturities.view(1, 1, 1, -1)
        w = (surface**2) * T_broad

        normalized_prices = black_scholes_call_normalized(self.log_moneyness, w)
        prices = spot_price * normalized_prices

        prices_curr = prices[:, :, :, :-1]
        prices_next = prices[:, :, :, 1:]

        spread_value = prices_curr - prices_next

        return torch.relu(spread_value).sum(dim=(-1, -2, -3))


class ButterflyArbitrageLoss(nn.Module):
    log_moneyness: Float[torch.Tensor, "1 1 h 1"]
    maturities: Float[torch.Tensor, "1 1 1 w"]

    def __init__(self, log_moneyness: Float[torch.Tensor, "h"], maturities: Float[torch.Tensor, "w"]) -> None:  # noqa: F821, UP037
        super().__init__()
        self.register_buffer("log_moneyness", log_moneyness.view(1, 1, -1, 1))
        self.register_buffer("maturities", maturities.view(1, 1, 1, -1))

    @override
    def forward(self, surface: Float[torch.Tensor, "b 1 h w"]) -> Float[torch.Tensor, "b"]:  # noqa: F821, UP037
        w = (surface**2) * self.maturities

        sqrt_w = torch.sqrt(w.clamp(min=1e-8))
        f2 = self.log_moneyness / sqrt_w + sqrt_w / 2

        df2 = f2[:, :, 1:, :] - f2[:, :, :-1, :]

        return torch.relu(-df2).sum(dim=(-1, -2, -3))


def compute_vega_grid(
    sigma: Float[torch.Tensor, "b 1 h w"],  # volatility surface (not log-vol!)
    log_moneyness: Float[torch.Tensor, "h"],  # forward log-moneyness grid  # noqa: F821, UP037
    tau: Float[torch.Tensor, "w"],  # time-to-maturity grid (in years)  # noqa: F821, UP037
    S: float = 1.0,  # spot price (normalized to 1)
) -> Float[torch.Tensor, "b 1 h w"]:
    r"""Compute vega for each point on the volatility surface.

    vega = S * \sqrt{τ} * φ(d₁)
    d₁ = (-k + 0.5\sigma^2τ) / (\sigma\sqrt{τ})

    Args:
        sigma: Implied volatility surface (NOT log-vol, actual \sigma values)
        log_moneyness: Forward log-moneyness k = log(K/F) per strike
        tau: Time to maturity in years
        S: Spot price (default 1.0)

    Returns:
        Vega at each grid point, same shape as sigma

    """
    # Broadcast grids to (1, 1, H, W)
    k = log_moneyness.view(1, 1, -1, 1)  # (1, 1, H, 1)
    tau_grid = tau.view(1, 1, 1, -1)  # (1, 1, 1, W)

    # Compute d1
    sqrt_tau = torch.sqrt(tau_grid.clamp(min=1e-8))
    sigma_sqrt_tau = sigma * sqrt_tau

    d1 = (-k + 0.5 * sigma**2 * tau_grid) / sigma_sqrt_tau.clamp(min=1e-8)

    # Standard normal PDF: φ(x) = (1/√(2π)) * exp(-x²/2)
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    phi_d1 = inv_sqrt_2pi * torch.exp(-0.5 * d1**2)

    # vega = S * √τ * φ(d₁)
    return S * sqrt_tau * phi_d1


class SmoothnessLoss(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        maturities: Float[torch.Tensor, "w"] | None = None,  # noqa: F821, UP037
        weight_mode: Literal["uniform", "direct", "inverse_sqrt"] = "inverse_sqrt",
        min_weight: float = 0.5,
        max_weight: float = 3.0,
        eps: float = 1e-8,
        *,
        penalize_moneyness: bool = True,
        penalize_maturity: bool = True,
    ) -> None:
        super().__init__()
        self.maturities = maturities
        self.penalize_moneyness = penalize_moneyness
        self.penalize_maturity = penalize_maturity
        self.weight_mode: Literal["uniform", "direct", "inverse_sqrt"] = weight_mode
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.eps = eps

    def _build_time_weights(self, x_0_pred: torch.Tensor) -> torch.Tensor | None:
        if self.maturities is None:
            return None
        tau = self.maturities.to(device=x_0_pred.device, dtype=x_0_pred.dtype).clamp(min=self.eps)
        match self.weight_mode:
            case "uniform":
                w = torch.ones_like(tau)
            case "direct":
                # penalize long maturities more
                w = tau
            case "inverse_sqrt":
                # stronger short-TTE smoothing
                w = 1.0 / torch.sqrt(tau)
            case _:
                msg = f"Unknown weight_mode={self.weight_mode}"
                raise ValueError(msg)
        w = w / w.mean().clamp(min=self.eps)
        w = w.clamp(min=self.min_weight, max=self.max_weight)
        return w.view(1, 1, 1, -1)

    @override
    def forward(self, x_0_pred: torch.Tensor) -> torch.Tensor:
        # Compute curvatures
        d2_h = torch.diff(x_0_pred, n=2, dim=2) if self.penalize_moneyness else None  # strike curvature
        d2_w = torch.diff(x_0_pred, n=2, dim=3) if self.penalize_maturity else None  # maturity curvature

        # Build maturity weights if needed
        t_weight = self._build_time_weights(x_0_pred)

        # Apply weights where terms exist
        reg_h = torch.zeros((), device=x_0_pred.device, dtype=x_0_pred.dtype)
        if d2_h is not None:
            if t_weight is not None:
                d2_h = d2_h * t_weight
            reg_h = torch.sqrt(torch.mean(d2_h**2) + 1e-8)

        reg_w = torch.zeros((), device=x_0_pred.device, dtype=x_0_pred.dtype)
        if d2_w is not None:
            if t_weight is not None:
                d2_w = d2_w * t_weight[:, :, :, 1:-1]
            reg_w = torch.sqrt(torch.mean(d2_w**2) + 1e-8)

        return reg_h + reg_w

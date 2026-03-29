from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from jaxtyping import Float, Int


class RegionAwareScheduler(nn.Module):
    T_1: int
    T_2: int
    T_total: int

    sqrt_one_minus_alphas_cumprod_T1: torch.Tensor
    sqrt_one_minus_alphas_cumprod_T2: torch.Tensor
    alphas_T1: torch.Tensor
    alphas_T2: torch.Tensor
    sqrt_alphas_cumprod_T1: torch.Tensor
    sqrt_alphas_cumprod_T2: torch.Tensor
    alphas_cumprod_T1: torch.Tensor
    alphas_cumprod_T2: torch.Tensor

    def __init__(self, T_1: int, T_2: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> None:
        super().__init__()

        self.T_1 = T_1
        self.T_2 = T_2
        self.T_total = T_1 + T_2

        betas_T1 = torch.linspace(beta_start, beta_end, self.T_1, dtype=torch.float32)
        betas_T2 = torch.linspace(beta_start, beta_end, self.T_2, dtype=torch.float32)
        alphas_T1 = 1.0 - betas_T1
        alphas_T2 = 1.0 - betas_T2
        alphas_cumprod_T1 = torch.cumprod(alphas_T1, dim=0)
        alphas_cumprod_T2 = torch.cumprod(alphas_T2, dim=0)

        # q(x_t | x_0)  # noqa: ERA001
        sqrt_alphas_cumprod_T1 = torch.sqrt(alphas_cumprod_T1)
        sqrt_one_minus_alphas_cumprod_T1 = torch.sqrt(1.0 - alphas_cumprod_T1)

        sqrt_alphas_cumprod_T2 = torch.sqrt(alphas_cumprod_T2)
        sqrt_one_minus_alphas_cumprod_T2 = torch.sqrt(1.0 - alphas_cumprod_T2)

        self.register_buffer("betas_T1", betas_T1)
        self.register_buffer("alphas_T1", alphas_T1)
        self.register_buffer("alphas_cumprod_T1", alphas_cumprod_T1)
        self.register_buffer("sqrt_alphas_cumprod_T1", sqrt_alphas_cumprod_T1)
        self.register_buffer("sqrt_one_minus_alphas_cumprod_T1", sqrt_one_minus_alphas_cumprod_T1)

        self.register_buffer("betas_T2", betas_T2)
        self.register_buffer("alphas_T2", alphas_T2)
        self.register_buffer("alphas_cumprod_T2", alphas_cumprod_T2)
        self.register_buffer("sqrt_alphas_cumprod_T2", sqrt_alphas_cumprod_T2)
        self.register_buffer("sqrt_one_minus_alphas_cumprod_T2", sqrt_one_minus_alphas_cumprod_T2)

    def get_sampling_scales(
        self, t: Int[torch.Tensor, "b 1"], mask: Float[torch.Tensor, "b 1 h w"]
    ) -> tuple[Float[torch.Tensor, "b 1 h w"], Float[torch.Tensor, "b 1 h w"]]:
        r"""Essentially returns \bar{alpha}_t and \sqrt{1 - \bar{alpha}_t}."""
        b = t.shape[0]
        t_view = t.view(b, 1, 1, 1)

        t_masked = t.clamp(max=self.T_1 - 1)
        noise_masked = self.sqrt_one_minus_alphas_cumprod_T1[t_masked].view(b, 1, 1, 1)
        signal_masked = self.sqrt_alphas_cumprod_T1[t_masked].view(b, 1, 1, 1)

        # second part, but we have to be careful as we might not noise it
        t_unmasked = (t - self.T_1).clamp(min=0)
        noise_unmasked = self.sqrt_one_minus_alphas_cumprod_T2[t_unmasked].view(b, 1, 1, 1)
        signal_unmasked = self.sqrt_alphas_cumprod_T2[t_unmasked].view(b, 1, 1, 1)

        # Force unmasked region to be clean when t < T_1
        is_phase_1 = t_view < self.T_1
        noise_unmasked = torch.where(is_phase_1, torch.zeros_like(noise_unmasked), noise_unmasked)
        signal_unmasked = torch.where(is_phase_1, torch.ones_like(signal_unmasked), signal_unmasked)

        final_noise = mask * noise_masked + (1 - mask) * noise_unmasked
        final_signal = mask * signal_masked + (1 - mask) * signal_unmasked

        return final_signal, final_noise

    # TODO(Andy): I think this is used very little anywhere or straight forward to implement anyway.
    def forward_process(self, x_start: torch.Tensor, t: torch.Tensor, mask: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Process images with masking at specific timesteps.

        Args:
            x_start: Input images of shape (B, C, H, W)
            mask: Binary mask of shape (B, 1, H, W)
            t: Timestep tensor of shape (B, 1)
            noise: Noise tensor of shape (B, C, H, W)

        """
        signal_scales, noise_scales = self.get_sampling_scales(t, mask)

        return signal_scales * x_start + noise_scales * noise

    def predict_x0(
        self,
        x_t: Float[torch.Tensor, "b 1 h w"],
        pred_eps: Float[torch.Tensor, "b 1 h w"],
        t: Int[torch.Tensor, "b 1"],
        mask: Float[torch.Tensor, "b 1 h w"],
    ) -> Float[torch.Tensor, "b 1 h w"]:
        """Formula: x_0 = (x_t - sqrt(1 - bar_alpha_t) * eps) / sqrt(bar_alpha_t)."""  # noqa: D401
        sqrt_bar_alpha_t, sqrt_1_min_bar_alpha_t = self.get_sampling_scales(t, mask)
        x0_pred = (x_t - sqrt_1_min_bar_alpha_t * pred_eps) / (sqrt_bar_alpha_t + 1e-8)

        # no noise regions handling
        b = x_t.shape[0]
        t_view = t.view(b, 1, 1, 1)
        return torch.where((mask == 0) & (t_view < self.T_1), x_t, x0_pred)

    def backward_eps(
        self,
        x_t: Float[torch.Tensor, "b 1 h w"],
        pred_eps: Float[torch.Tensor, "b 1 h w"],
        t: Int[torch.Tensor, "b 1"],
        mask: Float[torch.Tensor, "b 1 h w"],
    ) -> Float[torch.Tensor, "b 1 h w"]:
        # mathematical formula to get x_{t-1} = 1/sqrt(alpha_t) (x_t - \frac{1-alpha_t}{\sqrt{1-bar(alpha_t)}} \epsilon_t) + eps * variance  # noqa: E501
        b = x_t.shape[0]
        t_view = t.view(b, 1, 1, 1)

        t_masked = t.clamp(max=self.T_1 - 1).view(b, 1, 1, 1)
        t_unmasked = (t - self.T_1).clamp(min=0).view(b, 1, 1, 1)

        alpha_t = mask * self.alphas_T1[t_masked] + (1 - mask) * torch.where(
            t_view < self.T_1, torch.ones_like(mask), self.alphas_T2[t_unmasked]
        )
        _, sqrt_1_min_bar_alpha_t = self.get_sampling_scales(t, mask)

        x_prev = (1 / torch.sqrt(alpha_t)) * (x_t - (1 - alpha_t) / sqrt_1_min_bar_alpha_t * pred_eps)
        # 1- bar_alpha_t can be 0 where mask == 0 and t < T_1
        # where mask == 0 and t < T_1
        x_prev = torch.where((mask == 0) & (t_view < self.T_1), x_t, x_prev)

        # posterior variance
        if t.any() > 0:
            t_masked_m1 = (t - 1).clamp(min=0, max=self.T_1 - 1).view(b, 1, 1, 1)
            t_unmasked_m1 = (t - self.T_1 - 1).clamp(min=0).view(b, 1, 1, 1)

            alpha_bar_prev = mask * self.alphas_cumprod_T1[t_masked_m1] + (1 - mask) * torch.where(
                t_view < self.T_1, torch.ones_like(mask), self.alphas_cumprod_T2[t_unmasked_m1]
            )
            # handle t=0 case for alpha_bar_prev
            alpha_bar_prev = torch.where(t_view == 0, torch.ones_like(alpha_bar_prev), alpha_bar_prev)

            alpha_bar_curr = mask * self.alphas_cumprod_T1[t_masked] + (1 - mask) * torch.where(
                t_view < self.T_1, torch.ones_like(mask), self.alphas_cumprod_T2[t_unmasked]
            )
            posterior_variance = (1 - alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_curr)
            # posterior variance should be 0 where mask == 0:
            # this should fix the nan
            posterior_variance = torch.where(mask == 0, torch.zeros_like(posterior_variance), posterior_variance)
        else:
            posterior_variance = torch.zeros_like(alpha_t)

        return x_prev + torch.sqrt(posterior_variance.clamp(min=0)) * torch.randn_like(x_t)

    def backward_velocity(
        self,
        x_t: Float[torch.Tensor, "b 1 h w"],
        pred_v: Float[torch.Tensor, "b 1 h w"],
        t: Int[torch.Tensor, "b 1"],
        mask: Float[torch.Tensor, "b 1 h w"],
    ) -> Float[torch.Tensor, "b 1 h w"]:
        # convert v to eps
        signal_t, noise_t = self.get_sampling_scales(t, mask)
        pred_eps = noise_t * x_t + signal_t * pred_v

        # this is pretty much identical to above backward_eps as we just need to have eps which we can back out from pred_v.
        # mathematical formula to get x_{t-1} = 1/sqrt(alpha_t) (x_t - \frac{1-alpha_t}{\sqrt{1-bar(alpha_t)}} \epsilon_t) + eps * variance  # noqa: E501
        b = x_t.shape[0]
        t_view = t.view(b, 1, 1, 1)

        t_masked = t.clamp(max=self.T_1 - 1).view(b, 1, 1, 1)
        t_unmasked = (t - self.T_1).clamp(min=0).view(b, 1, 1, 1)

        alpha_t = mask * self.alphas_T1[t_masked] + (1 - mask) * torch.where(
            t_view < self.T_1, torch.ones_like(mask), self.alphas_T2[t_unmasked]
        )
        _, sqrt_1_min_bar_alpha_t = self.get_sampling_scales(t, mask)

        x_prev = (1 / torch.sqrt(alpha_t)) * (x_t - (1 - alpha_t) / sqrt_1_min_bar_alpha_t * pred_eps)
        # 1- bar_alpha_t can be 0 where mask == 0 and t < T_1
        # where mask == 0 and t < T_1
        x_prev = torch.where((mask == 0) & (t_view < self.T_1), x_t, x_prev)

        # posterior variance
        if t.any() > 0:
            t_masked_m1 = (t - 1).clamp(min=0, max=self.T_1 - 1).view(b, 1, 1, 1)
            t_unmasked_m1 = (t - self.T_1 - 1).clamp(min=0).view(b, 1, 1, 1)

            alpha_bar_prev = mask * self.alphas_cumprod_T1[t_masked_m1] + (1 - mask) * torch.where(
                t_view < self.T_1, torch.ones_like(mask), self.alphas_cumprod_T2[t_unmasked_m1]
            )
            # handle t=0 case for alpha_bar_prev
            alpha_bar_prev = torch.where(t_view == 0, torch.ones_like(alpha_bar_prev), alpha_bar_prev)

            alpha_bar_curr = mask * self.alphas_cumprod_T1[t_masked] + (1 - mask) * torch.where(
                t_view < self.T_1, torch.ones_like(mask), self.alphas_cumprod_T2[t_unmasked]
            )
            posterior_variance = (1 - alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_curr)
            # posterior variance should be 0 where mask == 0:
            # this should fix the nan
            posterior_variance = torch.where(mask == 0, torch.zeros_like(posterior_variance), posterior_variance)
        else:
            posterior_variance = torch.zeros_like(alpha_t)

        return x_prev + torch.sqrt(posterior_variance.clamp(min=0)) * torch.randn_like(x_t)

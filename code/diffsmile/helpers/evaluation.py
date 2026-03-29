from __future__ import annotations

import numpy as np
import torch

TAU_UNSCALE_FACTOR: float = 1095.0
QUERY_COORDS_BATCH_NDIM: int = 3


def _denorm_iv(x: torch.Tensor, mean: float | torch.Tensor, std: float | torch.Tensor) -> np.ndarray:
    m = float(mean.item()) if torch.is_tensor(mean) else float(mean)
    s = float(std.item()) if torch.is_tensor(std) else float(std)
    return np.exp(x.reshape(-1).detach().cpu().numpy() * s + m)


def _extract_coords(
    query_coords: torch.Tensor, tau_unscale_factor: float = TAU_UNSCALE_FACTOR
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = query_coords[0] if query_coords.ndim == QUERY_COORDS_BATCH_NDIM else query_coords
    tau_scaled = q[:, 0].detach().cpu().numpy()
    k = q[:, 1].detach().cpu().numpy()
    tau_unscaled = tau_scaled * tau_unscale_factor
    return k, tau_scaled, tau_unscaled


def save_buffers_raw(
    preds_buffer: list[np.ndarray],
    gt_buffer: list[np.ndarray],
    day_indices_buffer: list[int],
    filename: str = "inference_results.npz",
) -> None:
    all_preds = np.stack(preds_buffer, axis=0)
    all_gt = np.stack(gt_buffer, axis=0)
    all_days = np.array(day_indices_buffer)

    np.savez_compressed(filename, preds=all_preds, gt=all_gt, days=all_days)
    print(f"Saved raw tensors to {filename}")


def generate_random_mask(batch_size: int, h: int, w: int, prob: float = 0.3, device: str | torch.device = "cpu") -> torch.Tensor:
    """Randomly black out regions of the grid."""
    mask = torch.rand((batch_size, 1, h, w), device=device) < prob
    return mask.float()

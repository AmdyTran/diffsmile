from __future__ import annotations

import torch


def normalize_data(data: torch.Tensor, data_min: float, data_max: float) -> torch.Tensor:
    """Normalize data to the range [-1, 1] using min-max scaling."""
    normalized_data = (data - data_min) / (data_max - data_min)
    normalized_data = torch.mul(normalized_data, 2.0) - 1.0
    return normalized_data.clip(-1.0, 1.0)


def denormalize(tensor: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """Reverses the [-1, 1] scaling to get back to original IV units."""
    tensor = tensor.clip(-1.0, 1.0)
    x_01 = (tensor + 1.0) / 2.0
    return x_01 * (max_val - min_val) + min_val

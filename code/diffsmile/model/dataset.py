from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    import datetime as dt


class TimeSeriesDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        surfaces: torch.Tensor,
        surfaces_ema_short: torch.Tensor,
        surfaces_ema_long: torch.Tensor,
        scalars: torch.Tensor,
        dates: list[dt.date],
    ) -> None:
        self.surfaces = surfaces
        self.surfaces_ema_short = surfaces_ema_short
        self.surfaces_ema_long = surfaces_ema_long
        self.scalars = scalars
        self.dates = dates

    def __len__(self) -> int:
        return len(self.surfaces) - 1

    @override
    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # today's surface, today's scalars, tomorrow's surface
        return (
            self.surfaces[index],
            self.surfaces_ema_short[index],
            self.surfaces_ema_long[index],
            self.scalars[index],
            self.surfaces[index + 1],
        )

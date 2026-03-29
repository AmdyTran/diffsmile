from __future__ import annotations

import torch

from diffsmile.config import TrainingConfigConditionalDiffusion
from diffsmile.config import config as default_config
from diffsmile.gnot_lightning import GNOTDiffusionLightningModule
from diffsmile.model.scheduler_rad import RegionAwareScheduler


def resolve_device(device: torch.device | str | None = None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint_model(
    checkpoint_path: str,
    *,
    device: torch.device | str | None = None,
    strict: bool = False,
) -> tuple[GNOTDiffusionLightningModule, torch.device]:
    resolved_device = resolve_device(device)
    model = GNOTDiffusionLightningModule.load_from_checkpoint(checkpoint_path, strict=strict)
    model = model.to(resolved_device)
    model.eval()
    return model, resolved_device


def build_scheduler(
    *,
    config: TrainingConfigConditionalDiffusion = default_config,
    device: torch.device | str | None = None,
) -> RegionAwareScheduler:
    resolved_device = resolve_device(device)
    scheduler = RegionAwareScheduler(T_1=config.T1, T_2=config.T2)
    return scheduler.to(resolved_device)

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from diffsmile.config import TrainingConfigConditionalDiffusion
from diffsmile.config import config as default_config
from diffsmile.gnot_lightning import VolSurfacePointCloudDataModule

if TYPE_CHECKING:
    import datetime as dt

    import torch
    from sklearn.preprocessing import StandardScaler

    from diffsmile.model.dataset import TimeSeriesDataset


@dataclass(slots=True)
class EvaluationData:
    datamodule: VolSurfacePointCloudDataModule
    train_dataset: TimeSeriesDataset
    val_dataset: TimeSeriesDataset
    scaler: StandardScaler
    data_mean: torch.Tensor
    data_std: torch.Tensor
    log_moneyness_grid: torch.Tensor
    ttm_grid: torch.Tensor
    train_prediction_dates: list[dt.date]
    val_prediction_dates: list[dt.date]


def _prediction_dates(dataset: TimeSeriesDataset) -> list[dt.date]:
    return list(dataset.dates[: len(dataset)])


def build_evaluation_data(
    *,
    batch_size: int = 64,
    num_workers: int = 4,
    config: TrainingConfigConditionalDiffusion = default_config,
) -> EvaluationData:
    datamodule = VolSurfacePointCloudDataModule(batch_size=batch_size, num_workers=num_workers, config=config)
    datamodule.setup()

    if datamodule.train_dataset is None:
        msg = "Training dataset was not initialized by VolSurfacePointCloudDataModule.setup()."
        raise RuntimeError(msg)
    if datamodule.val_dataset is None:
        msg = "Validation dataset was not initialized by VolSurfacePointCloudDataModule.setup()."
        raise RuntimeError(msg)
    if datamodule.data_mean is None or datamodule.data_std is None:
        msg = "Data normalization statistics were not initialized by VolSurfacePointCloudDataModule.setup()."
        raise RuntimeError(msg)
    if datamodule.log_moneyness_grid is None or datamodule.ttm_grid is None:
        msg = "Surface grids were not initialized by VolSurfacePointCloudDataModule.setup()."
        raise RuntimeError(msg)

    return EvaluationData(
        datamodule=datamodule,
        train_dataset=datamodule.train_dataset,
        val_dataset=datamodule.val_dataset,
        scaler=datamodule.scaler,
        data_mean=datamodule.data_mean,
        data_std=datamodule.data_std,
        log_moneyness_grid=datamodule.log_moneyness_grid,
        ttm_grid=datamodule.ttm_grid,
        train_prediction_dates=_prediction_dates(datamodule.train_dataset),
        val_prediction_dates=_prediction_dates(datamodule.val_dataset),
    )

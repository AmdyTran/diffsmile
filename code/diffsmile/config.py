from __future__ import annotations

import datetime as dt
from enum import IntEnum
from pathlib import Path

from pydantic import BaseModel
from pydantic.dataclasses import dataclass


class ConditioningScalarIndex(IntEnum):
    RET = 0
    ST_EWMA_RET = 1
    LT_EWMA_RET = 2
    ST_EWMA_SQ_RET = 3
    LT_EWMA_SQ_RET = 4
    VIX = 5


class TrainingConfigConditionalDiffusion(BaseModel):
    epochs: int = 1000
    lr: float = 0.0002
    T1: int = 500
    T2: int = 500
    EMBEDDING_DIM: int = 32
    SCALAR_COUNT: int = 6  #  st-ewma ret, lt-ewma ret, st-ewma squared returns, lt-ewma squared returns, vix, returns (t -> t+1)

    USE_ARBITRAGE_LOSS: bool = True
    IMAGE_HEIGHT: int = 32  # Note: here height means the log-moneyness
    IMAGE_WIDTH: int = 24  # and here widht (columns) are the maturities

    PERLIN_RES: tuple[int, int] = (8, 8)
    PERLIN_OCTAVES: int = 2
    PERLIN_PERSISTENCE: float = 0.5
    MASK_THRESHOLD_MIN: float = 0.3
    MASK_THRESHOLD_MAX: float = 0.7


@dataclass
class DatasetConfig:
    # TODO(reader): update these values
    # join forward price from "OptionMetrics - Forward Price" as the dataset does not contain it anymore!
    CORE_DATA_PATH: Path = Path("/cluster/project/math/andtran/develop/masters_thesis/code/data/")
    conditional_data_path: Path = CORE_DATA_PATH / "conditional"
    option_metrics_dataset: Path = CORE_DATA_PATH / "spx_options_data.parquet"
    train_start: dt.date = dt.date(2010, 1, 1)
    train_end: dt.date = dt.date(2020, 12, 31)

    val_start: dt.date = dt.date(2021, 1, 1)
    val_end: dt.date = dt.date(2023, 12, 31)

    output_path_surfaces: Path = CORE_DATA_PATH / "surfaces.parquet"
    output_conditioning_scalars: Path = CORE_DATA_PATH / "conditioning_vectors_w_ret.pt"
    merged_surfaces_path: Path = conditional_data_path / "spx_iv_dataset_full_365.pt"
    merged_conditioning_scalars_path: Path = CORE_DATA_PATH / "conditioning_vectors_w_ret.pt"


config = TrainingConfigConditionalDiffusion()

dataset_config = DatasetConfig()

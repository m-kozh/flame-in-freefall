"""The small model contract used by the API and all scientific services.

A future PINN can return the same arrays without exposing PyTorch to the API.
Rates and their intervals are conditional on a flame being sustained; the
separate classifier answers whether a sustained flame occurs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self, runtime_checkable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PredictionBatch:
    """One one-dimensional array per quantity, in the input row order."""

    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    extinction_probability: np.ndarray
    physics_baseline: np.ndarray
    extinction_lower: np.ndarray
    extinction_upper: np.ndarray
    uncertainty: np.ndarray
    physics_regime: np.ndarray
    in_data_range: np.ndarray
    log_std: np.ndarray


@runtime_checkable
class Predictor(Protocol):
    model_version: str
    data_source: str
    training_data: pd.DataFrame

    def fit(self, df: pd.DataFrame) -> Self: ...

    def predict(self, X: pd.DataFrame) -> PredictionBatch: ...

    def support_distance(self, X: pd.DataFrame) -> np.ndarray: ...

    def save(self, path: str | Path) -> None: ...

    @classmethod
    def load(cls, path: str | Path) -> Self: ...


@runtime_checkable
class MemberPredictor(Predictor, Protocol):
    """Optional ensemble capability for information-driven experiment design.

    Return member-by-row arrays named extinction_probability and log_spread_rate.
    An additional row-wise in_data_range array avoids a second prediction call.
    Ordinary prediction, maps and scenario analysis only need Predictor.
    """

    def predict_members(self, X: pd.DataFrame) -> dict[str, np.ndarray]: ...

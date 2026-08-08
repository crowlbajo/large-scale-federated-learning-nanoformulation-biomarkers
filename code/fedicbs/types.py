from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np
from numpy.typing import NDArray
from torch import Tensor


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class FeatureDecision(str, Enum):
    CAUSAL = "causal"
    CONFOUNDED = "confounded"
    INADEQUATE = "inadequate"


@dataclass(frozen=True)
class SiteBatch:
    features: Tensor
    targets: Tensor
    site_ids: Tensor
    sample_ids: tuple[str, ...]

    def size(self) -> int:
        return int(self.targets.shape[0])


@dataclass(frozen=True)
class SiteStatistics:
    site: str
    coefficients: FloatArray
    standard_errors: FloatArray
    covariance: FloatArray
    sample_count: int

    def validate(self) -> None:
        dimension = self.coefficients.shape[0]
        if self.standard_errors.shape != (dimension,):
            raise ValueError("standard error shape mismatch")
        if self.covariance.shape != (dimension, dimension):
            raise ValueError("covariance shape mismatch")
        if self.sample_count < 1:
            raise ValueError("sample_count must be positive")
        if np.any(self.standard_errors <= 0.0):
            raise ValueError("standard errors must be positive")


@dataclass(frozen=True)
class InvarianceResult:
    feature_index: int
    pooled_coefficient: float
    pooled_standard_error: float
    z_score: float
    q_statistic: float
    predictivity_p_value: float
    invariance_p_value: float
    decision: FeatureDecision
    contributing_sites: tuple[str, ...]


@dataclass(frozen=True)
class MetricBundle:
    auc: float
    sensitivity: float
    specificity: float
    f1: float
    threshold: float


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    standard_error: float


@dataclass(frozen=True)
class RoundRecord:
    round_index: int
    training_loss: float
    validation_auc: float
    gradient_norm: float
    invariant_count: int
    mean_q: float


@dataclass(frozen=True)
class ClientUpdate:
    site: str
    sample_count: int
    state: Mapping[str, Tensor]
    loss: float
    gradient_norm: float


@dataclass(frozen=True)
class PrivacyLedgerEntry:
    round_index: int
    epsilon: float
    delta: float
    noise_multiplier: float
    sensitivity: float


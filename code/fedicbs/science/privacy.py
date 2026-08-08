from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt

import numpy as np
from numpy.typing import NDArray

from fedicbs.types import PrivacyLedgerEntry, SiteStatistics


@dataclass(frozen=True)
class GaussianMechanism:
    epsilon: float
    delta: float
    sensitivity: float

    def validate(self) -> None:
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must lie between zero and one")
        if self.sensitivity < 0.0:
            raise ValueError("sensitivity cannot be negative")

    @property
    def standard_deviation(self) -> float:
        self.validate()
        return self.sensitivity * sqrt(2.0 * log(1.25 / self.delta)) / self.epsilon

    def perturb(
        self,
        values: NDArray[np.float64],
        generator: np.random.Generator,
    ) -> NDArray[np.float64]:
        array = np.asarray(values, dtype=np.float64)
        noise = generator.normal(0.0, self.standard_deviation, size=array.shape)
        return array + noise


@dataclass
class MomentsAccountant:
    total_delta: float
    entries: list[PrivacyLedgerEntry]

    @classmethod
    def create(cls, total_delta: float) -> MomentsAccountant:
        if not 0.0 < total_delta < 1.0:
            raise ValueError("total_delta must lie between zero and one")
        return cls(total_delta=total_delta, entries=[])

    def record(
        self,
        round_index: int,
        epsilon: float,
        delta: float,
        noise_multiplier: float,
        sensitivity: float,
    ) -> None:
        if self.entries and round_index <= self.entries[-1].round_index:
            raise ValueError("round indices must be strictly increasing")
        self.entries.append(
            PrivacyLedgerEntry(
                round_index=round_index,
                epsilon=epsilon,
                delta=delta,
                noise_multiplier=noise_multiplier,
                sensitivity=sensitivity,
            )
        )

    def basic_epsilon(self) -> float:
        return float(sum(entry.epsilon for entry in self.entries))

    def advanced_epsilon(self) -> float:
        if not self.entries:
            return 0.0
        squares = sum(entry.epsilon ** 2 for entry in self.entries)
        linear = sum(entry.epsilon * (np.exp(entry.epsilon) - 1.0) for entry in self.entries)
        return float(sqrt(2.0 * log(1.0 / self.total_delta) * squares) + linear)

    def composed_delta(self) -> float:
        return float(self.total_delta + sum(entry.delta for entry in self.entries))

    def assert_budget(self, epsilon: float, delta: float) -> None:
        if self.advanced_epsilon() > epsilon:
            raise RuntimeError("privacy epsilon budget exceeded")
        if self.composed_delta() > delta:
            raise RuntimeError("privacy delta budget exceeded")


def logistic_coefficient_sensitivity(
    sample_count: int,
    minimum_hessian_eigenvalue: float,
) -> float:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if minimum_hessian_eigenvalue <= 0.0:
        raise ValueError("minimum Hessian eigenvalue must be positive")
    return 2.0 / (sample_count * minimum_hessian_eigenvalue)


def privatize_statistics(
    statistics: SiteStatistics,
    mechanism: GaussianMechanism,
    seed: int,
) -> SiteStatistics:
    generator = np.random.default_rng(seed)
    coefficients = mechanism.perturb(statistics.coefficients, generator)
    result = SiteStatistics(
        site=statistics.site,
        coefficients=coefficients,
        standard_errors=statistics.standard_errors.copy(),
        covariance=statistics.covariance.copy(),
        sample_count=statistics.sample_count,
    )
    result.validate()
    return result


def round_epsilon(total_epsilon: float, rounds: int) -> float:
    if total_epsilon <= 0.0:
        raise ValueError("total_epsilon must be positive")
    if rounds < 1:
        raise ValueError("rounds must be positive")
    return total_epsilon / rounds


def calibrated_noise_multiplier(
    epsilon: float,
    delta: float,
    sensitivity: float,
) -> float:
    mechanism = GaussianMechanism(epsilon, delta, sensitivity)
    if sensitivity == 0.0:
        return 0.0
    return mechanism.standard_deviation / sensitivity


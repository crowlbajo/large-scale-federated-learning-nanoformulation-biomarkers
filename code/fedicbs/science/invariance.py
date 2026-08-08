from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2, norm
from sklearn.linear_model import LogisticRegression

from fedicbs.types import FeatureDecision, InvarianceResult, SiteStatistics


@dataclass(frozen=True)
class InvarianceScreen:
    alpha: float = 0.05
    minimum_site_samples: int = 80
    bonferroni_dimension: int = 128

    @property
    def adjusted_alpha(self) -> float:
        return self.alpha / self.bonferroni_dimension

    def evaluate_feature(
        self,
        feature_index: int,
        statistics: list[SiteStatistics],
    ) -> InvarianceResult:
        adequate = [
            item
            for item in statistics
            if item.sample_count >= self.minimum_site_samples
        ]
        if len(adequate) < 5:
            return InvarianceResult(
                feature_index=feature_index,
                pooled_coefficient=float("nan"),
                pooled_standard_error=float("nan"),
                z_score=float("nan"),
                q_statistic=float("nan"),
                predictivity_p_value=float("nan"),
                invariance_p_value=float("nan"),
                decision=FeatureDecision.INADEQUATE,
                contributing_sites=tuple(item.site for item in adequate),
            )
        coefficients = np.asarray(
            [item.coefficients[feature_index] for item in adequate],
            dtype=np.float64,
        )
        standard_errors = np.asarray(
            [item.standard_errors[feature_index] for item in adequate],
            dtype=np.float64,
        )
        if np.any(standard_errors <= 0.0) or not np.all(np.isfinite(standard_errors)):
            raise ValueError("standard errors must be finite and positive")
        weights = np.reciprocal(np.square(standard_errors))
        pooled = float(np.dot(weights, coefficients) / weights.sum())
        pooled_standard_error = float(sqrt(1.0 / weights.sum()))
        z_score = pooled / pooled_standard_error
        predictivity_p = float(2.0 * norm.sf(abs(z_score)))
        q_statistic = float(np.dot(weights, np.square(coefficients - pooled)))
        invariance_p = float(chi2.sf(q_statistic, df=len(adequate) - 1))
        predictive = predictivity_p < self.adjusted_alpha
        invariant = q_statistic < float(
            chi2.ppf(1.0 - self.adjusted_alpha, df=len(adequate) - 1)
        )
        decision = FeatureDecision.CAUSAL if predictive and invariant else FeatureDecision.CONFOUNDED
        return InvarianceResult(
            feature_index=feature_index,
            pooled_coefficient=pooled,
            pooled_standard_error=pooled_standard_error,
            z_score=float(z_score),
            q_statistic=q_statistic,
            predictivity_p_value=predictivity_p,
            invariance_p_value=invariance_p,
            decision=decision,
            contributing_sites=tuple(item.site for item in adequate),
        )

    def evaluate_all(
        self,
        statistics: list[SiteStatistics],
    ) -> tuple[InvarianceResult, ...]:
        if not statistics:
            raise ValueError("at least one site statistic is required")
        dimensions = {item.coefficients.shape[0] for item in statistics}
        if len(dimensions) != 1:
            raise ValueError("all sites must report the same feature dimension")
        dimension = dimensions.pop()
        return tuple(
            self.evaluate_feature(index, statistics)
            for index in range(dimension)
        )

    def selected_indices(
        self,
        statistics: list[SiteStatistics],
    ) -> NDArray[np.int64]:
        results = self.evaluate_all(statistics)
        return np.asarray(
            [
                result.feature_index
                for result in results
                if result.decision is FeatureDecision.CAUSAL
            ],
            dtype=np.int64,
        )


def fit_marginal_site_statistics(
    site: str,
    features: NDArray[np.float64],
    targets: NDArray[np.int64],
    l2_regularization: float = 1e-3,
) -> SiteStatistics:
    matrix = np.asarray(features, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    if matrix.ndim != 2:
        raise ValueError("features must have rank two")
    if labels.shape != (matrix.shape[0],):
        raise ValueError("target shape mismatch")
    if np.unique(labels).size != 2:
        raise ValueError("both target classes are required")
    dimension = matrix.shape[1]
    coefficients = np.zeros(dimension, dtype=np.float64)
    standard_errors = np.zeros(dimension, dtype=np.float64)
    covariance = np.zeros((dimension, dimension), dtype=np.float64)
    regularization_inverse = 1.0 / l2_regularization
    for feature_index in range(dimension):
        column = matrix[:, feature_index : feature_index + 1]
        estimator = LogisticRegression(
            penalty="l2",
            C=regularization_inverse,
            solver="lbfgs",
            fit_intercept=True,
            max_iter=1000,
        )
        estimator.fit(column, labels)
        coefficient = float(estimator.coef_[0, 0])
        linear = estimator.intercept_[0] + column[:, 0] * coefficient
        probability = 1.0 / (1.0 + np.exp(-np.clip(linear, -40.0, 40.0)))
        weights = probability * (1.0 - probability)
        design = np.column_stack([np.ones(matrix.shape[0]), column[:, 0]])
        information = design.T @ (weights[:, None] * design)
        information[1, 1] += l2_regularization
        inverse_information = np.linalg.pinv(information)
        standard_error = sqrt(max(float(inverse_information[1, 1]), 1e-16))
        coefficients[feature_index] = coefficient
        standard_errors[feature_index] = standard_error
        covariance[feature_index, feature_index] = inverse_information[1, 1]
    result = SiteStatistics(site, coefficients, standard_errors, covariance, len(labels))
    result.validate()
    return result


def inverse_variance_pool(
    coefficients: NDArray[np.float64],
    standard_errors: NDArray[np.float64],
) -> tuple[float, float]:
    beta = np.asarray(coefficients, dtype=np.float64)
    errors = np.asarray(standard_errors, dtype=np.float64)
    if beta.ndim != 1 or errors.shape != beta.shape:
        raise ValueError("coefficient vectors must have matching rank-one shapes")
    if np.any(errors <= 0.0):
        raise ValueError("standard errors must be positive")
    weights = 1.0 / np.square(errors)
    pooled = float(np.dot(weights, beta) / weights.sum())
    pooled_error = float(sqrt(1.0 / weights.sum()))
    return pooled, pooled_error


def cochran_q(
    coefficients: NDArray[np.float64],
    standard_errors: NDArray[np.float64],
) -> tuple[float, float]:
    pooled, _ = inverse_variance_pool(coefficients, standard_errors)
    weights = 1.0 / np.square(standard_errors)
    statistic = float(np.dot(weights, np.square(coefficients - pooled)))
    p_value = float(chi2.sf(statistic, df=len(coefficients) - 1))
    return statistic, p_value


def benjamini_hochberg(
    p_values: NDArray[np.float64],
    false_discovery_rate: float,
) -> NDArray[np.bool_]:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("p_values must have rank one")
    if not 0.0 < false_discovery_rate < 1.0:
        raise ValueError("false_discovery_rate must lie between zero and one")
    order = np.argsort(values)
    sorted_values = values[order]
    thresholds = false_discovery_rate * np.arange(1, len(values) + 1) / len(values)
    passing = sorted_values <= thresholds
    selected = np.zeros(len(values), dtype=np.bool_)
    if np.any(passing):
        largest = int(np.flatnonzero(passing)[-1])
        selected[order[: largest + 1]] = True
    return selected


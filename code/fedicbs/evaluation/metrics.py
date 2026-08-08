from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from fedicbs.types import ConfidenceInterval, MetricBundle


def binary_metrics(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    threshold: float = 0.5,
) -> MetricBundle:
    truth, scores = _validate_binary_inputs(labels, probabilities)
    predictions = (scores >= threshold).astype(np.int64)
    matrix = confusion_matrix(truth, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    sensitivity = _safe_ratio(true_positive, true_positive + false_negative)
    specificity = _safe_ratio(true_negative, true_negative + false_positive)
    return MetricBundle(
        auc=float(roc_auc_score(truth, scores)),
        sensitivity=sensitivity,
        specificity=specificity,
        f1=float(f1_score(truth, predictions, zero_division=0)),
        threshold=threshold,
    )


def youden_threshold(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
) -> float:
    truth, scores = _validate_binary_inputs(labels, probabilities)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(truth, scores)
    index = int(np.argmax(true_positive_rate - false_positive_rate))
    return float(thresholds[index])


def sensitivity_at_specificity(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    required_specificity: float,
) -> tuple[float, float]:
    truth, scores = _validate_binary_inputs(labels, probabilities)
    if not 0.0 <= required_specificity <= 1.0:
        raise ValueError("required_specificity must lie between zero and one")
    false_positive_rate, true_positive_rate, thresholds = roc_curve(truth, scores)
    eligible = np.flatnonzero(1.0 - false_positive_rate >= required_specificity)
    if eligible.size == 0:
        return 0.0, float("inf")
    index = int(eligible[np.argmax(true_positive_rate[eligible])])
    return float(true_positive_rate[index]), float(thresholds[index])


def specificity_at_sensitivity(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    required_sensitivity: float,
) -> tuple[float, float]:
    truth, scores = _validate_binary_inputs(labels, probabilities)
    if not 0.0 <= required_sensitivity <= 1.0:
        raise ValueError("required_sensitivity must lie between zero and one")
    false_positive_rate, true_positive_rate, thresholds = roc_curve(truth, scores)
    eligible = np.flatnonzero(true_positive_rate >= required_sensitivity)
    if eligible.size == 0:
        return 0.0, float("-inf")
    specificities = 1.0 - false_positive_rate[eligible]
    index = int(eligible[np.argmax(specificities)])
    return float(1.0 - false_positive_rate[index]), float(thresholds[index])


def equalized_odds_difference(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    groups: NDArray[np.str_],
    threshold: float = 0.5,
) -> float:
    truth, scores = _validate_binary_inputs(labels, probabilities)
    group_values = np.asarray(groups)
    if group_values.shape != truth.shape:
        raise ValueError("group shape mismatch")
    true_positive_rates: list[float] = []
    false_positive_rates: list[float] = []
    for group in np.unique(group_values):
        mask = group_values == group
        predictions = (scores[mask] >= threshold).astype(np.int64)
        matrix = confusion_matrix(truth[mask], predictions, labels=[0, 1])
        true_negative, false_positive, false_negative, true_positive = matrix.ravel()
        true_positive_rates.append(_safe_ratio(true_positive, true_positive + false_negative))
        false_positive_rates.append(_safe_ratio(false_positive, false_positive + true_negative))
    tpr_gap = max(true_positive_rates) - min(true_positive_rates)
    fpr_gap = max(false_positive_rates) - min(false_positive_rates)
    return float(max(tpr_gap, fpr_gap))


def bootstrap_metric(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    metric: str,
    replicates: int = 5000,
    confidence: float = 0.95,
    seed: int = 1,
) -> ConfidenceInterval:
    truth, scores = _validate_binary_inputs(labels, probabilities)
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie between zero and one")
    generator = np.random.default_rng(seed)
    positive = np.flatnonzero(truth == 1)
    negative = np.flatnonzero(truth == 0)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled_positive = generator.choice(positive, size=len(positive), replace=True)
        sampled_negative = generator.choice(negative, size=len(negative), replace=True)
        sampled = np.concatenate([sampled_positive, sampled_negative])
        generator.shuffle(sampled)
        bundle = binary_metrics(truth[sampled], scores[sampled])
        estimates.append(_metric_value(bundle, metric))
    values = np.asarray(estimates, dtype=np.float64)
    alpha = (1.0 - confidence) / 2.0
    estimate = _metric_value(binary_metrics(truth, scores), metric)
    return ConfidenceInterval(
        estimate=estimate,
        lower=float(np.quantile(values, alpha)),
        upper=float(np.quantile(values, 1.0 - alpha)),
        standard_error=float(values.std(ddof=1)),
    )


def stratified_bootstrap_bundle(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    replicates: int = 5000,
    seed: int = 1,
) -> dict[str, ConfidenceInterval]:
    return {
        metric: bootstrap_metric(
            labels,
            probabilities,
            metric,
            replicates=replicates,
            seed=seed,
        )
        for metric in ("auc", "sensitivity", "specificity", "f1")
    }


def site_auc_spread(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    sites: NDArray[np.str_],
) -> tuple[dict[str, float], float]:
    truth, scores = _validate_binary_inputs(labels, probabilities)
    site_values = np.asarray(sites)
    if site_values.shape != truth.shape:
        raise ValueError("site shape mismatch")
    values: dict[str, float] = {}
    for site in np.unique(site_values):
        mask = site_values == site
        if np.unique(truth[mask]).size != 2:
            continue
        values[str(site)] = float(roc_auc_score(truth[mask], scores[mask]))
    if len(values) < 2:
        raise ValueError("at least two evaluable sites are required")
    spread = max(values.values()) - min(values.values())
    return values, float(spread)


def subgroup_metrics(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    groups: NDArray[np.str_],
    threshold: float = 0.5,
) -> dict[str, MetricBundle]:
    truth, scores = _validate_binary_inputs(labels, probabilities)
    group_values = np.asarray(groups)
    if group_values.shape != truth.shape:
        raise ValueError("group shape mismatch")
    results: dict[str, MetricBundle] = {}
    for group in np.unique(group_values):
        mask = group_values == group
        if np.unique(truth[mask]).size != 2:
            continue
        results[str(group)] = binary_metrics(truth[mask], scores[mask], threshold)
    return results


def _validate_binary_inputs(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    truth = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if truth.ndim != 1 or scores.shape != truth.shape:
        raise ValueError("labels and probabilities must be matching rank-one arrays")
    if not np.all(np.isfinite(scores)):
        raise ValueError("probabilities must be finite")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("probabilities must lie between zero and one")
    if set(np.unique(truth)) != {0, 1}:
        raise ValueError("both binary classes are required")
    return truth, scores


def _safe_ratio(numerator: int | np.integer, denominator: int | np.integer) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _metric_value(bundle: MetricBundle, metric: str) -> float:
    if metric == "auc":
        return bundle.auc
    if metric == "sensitivity":
        return bundle.sensitivity
    if metric == "specificity":
        return bundle.specificity
    if metric == "f1":
        return bundle.f1
    raise ValueError(f"unknown metric: {metric}")


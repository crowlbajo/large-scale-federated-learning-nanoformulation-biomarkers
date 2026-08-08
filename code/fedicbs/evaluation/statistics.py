from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2, norm, ttest_ind


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float
    effect_size: float | None = None


def cohens_d(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> float:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.ndim != 1 or second.ndim != 1:
        raise ValueError("samples must have rank one")
    if first.size < 2 or second.size < 2:
        raise ValueError("each sample requires at least two observations")
    degrees = first.size + second.size - 2
    pooled_variance = (
        (first.size - 1) * first.var(ddof=1)
        + (second.size - 1) * second.var(ddof=1)
    ) / degrees
    if pooled_variance <= 0.0:
        return 0.0
    return float((first.mean() - second.mean()) / sqrt(pooled_variance))


def independent_comparison(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> TestResult:
    statistic, p_value = ttest_ind(left, right, equal_var=False)
    return TestResult(
        statistic=float(statistic),
        p_value=float(p_value),
        effect_size=cohens_d(left, right),
    )


def delong_auc_test(
    labels: NDArray[np.int64],
    first_scores: NDArray[np.float64],
    second_scores: NDArray[np.float64],
) -> TestResult:
    truth = np.asarray(labels, dtype=np.int64)
    first = np.asarray(first_scores, dtype=np.float64)
    second = np.asarray(second_scores, dtype=np.float64)
    if truth.ndim != 1 or first.shape != truth.shape or second.shape != truth.shape:
        raise ValueError("DeLong inputs must be matching rank-one arrays")
    order = np.argsort(-truth)
    sorted_truth = truth[order]
    positive_count = int(sorted_truth.sum())
    negative_count = len(sorted_truth) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("both classes are required")
    predictions = np.vstack([first[order], second[order]])
    aucs, covariance = _fast_delong(predictions, positive_count)
    difference = float(aucs[0] - aucs[1])
    contrast = np.asarray([[1.0, -1.0]])
    variance = float(contrast @ covariance @ contrast.T)
    z_score = abs(difference) / sqrt(max(variance, 1e-16))
    p_value = float(2.0 * norm.sf(z_score))
    return TestResult(statistic=z_score, p_value=p_value, effect_size=difference)


def _fast_delong(
    predictions: NDArray[np.float64],
    positive_count: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    positives = predictions[:, :positive_count]
    negatives = predictions[:, positive_count:]
    classifiers = predictions.shape[0]
    positive_midrank = np.zeros(positives.shape, dtype=np.float64)
    negative_midrank = np.zeros(negatives.shape, dtype=np.float64)
    combined_midrank = np.zeros(predictions.shape, dtype=np.float64)
    for row in range(classifiers):
        positive_midrank[row] = _midrank(positives[row])
        negative_midrank[row] = _midrank(negatives[row])
        combined_midrank[row] = _midrank(predictions[row])
    negative_count = negatives.shape[1]
    aucs = (
        combined_midrank[:, :positive_count].sum(axis=1)
        / (positive_count * negative_count)
        - (positive_count + 1.0) / (2.0 * negative_count)
    )
    positive_components = (
        combined_midrank[:, :positive_count] - positive_midrank
    ) / negative_count
    negative_components = (
        combined_midrank[:, positive_count:] - negative_midrank
    ) / positive_count
    covariance = np.cov(positive_components) / positive_count
    covariance += np.cov(negative_components) / negative_count
    covariance = np.atleast_2d(covariance)
    return aucs, covariance


def _midrank(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values)
    sorted_values = values[order]
    count = len(values)
    result = np.zeros(count, dtype=np.float64)
    start = 0
    while start < count:
        end = start
        while end < count and sorted_values[end] == sorted_values[start]:
            end += 1
        result[start:end] = 0.5 * (start + end - 1)
        start = end
    restored = np.empty(count, dtype=np.float64)
    restored[order] = result + 1.0
    return restored


def intraclass_correlation(
    ratings: NDArray[np.float64],
) -> float:
    matrix = np.asarray(ratings, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("ratings must have rank two")
    subjects, raters = matrix.shape
    if subjects < 2 or raters < 2:
        raise ValueError("ICC requires at least two subjects and raters")
    grand_mean = matrix.mean()
    subject_means = matrix.mean(axis=1)
    rater_means = matrix.mean(axis=0)
    subject_ms = raters * np.square(subject_means - grand_mean).sum() / (subjects - 1)
    residual = matrix - subject_means[:, None] - rater_means[None, :] + grand_mean
    error_ms = np.square(residual).sum() / ((subjects - 1) * (raters - 1))
    denominator = subject_ms + (raters - 1) * error_ms
    return float((subject_ms - error_ms) / denominator) if denominator else 0.0


def cochran_mantel_haenszel(
    tables: NDArray[np.int64],
) -> TestResult:
    values = np.asarray(tables, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (2, 2):
        raise ValueError("CMH tables must have shape strata by two by two")
    numerator = 0.0
    variance = 0.0
    for table in values:
        total = table.sum()
        if total <= 1:
            continue
        row_total = table.sum(axis=1)
        column_total = table.sum(axis=0)
        expected = row_total[0] * column_total[0] / total
        numerator += table[0, 0] - expected
        variance += (
            row_total[0]
            * row_total[1]
            * column_total[0]
            * column_total[1]
            / (total * total * (total - 1.0))
        )
    statistic = numerator * numerator / variance if variance > 0.0 else 0.0
    return TestResult(statistic=float(statistic), p_value=float(chi2.sf(statistic, 1)))


def bonferroni(
    p_values: NDArray[np.float64],
    alpha: float = 0.05,
) -> NDArray[np.bool_]:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("p_values must have rank one")
    return values < alpha / len(values)


def minimum_clinical_difference(
    observed: float,
    comparator: float,
    threshold: float,
) -> bool:
    return observed - comparator >= threshold


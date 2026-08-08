from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn


@dataclass(frozen=True)
class AttributionResult:
    feature_names: tuple[str, ...]
    values: NDArray[np.float64]
    baseline_prediction: float
    prediction: float

    def ranked(self) -> tuple[tuple[str, float], ...]:
        order = np.argsort(-np.abs(self.values))
        return tuple(
            (self.feature_names[index], float(self.values[index]))
            for index in order
        )


def integrated_gradients(
    model: nn.Module,
    features: Tensor,
    baseline: Tensor,
    steps: int = 64,
) -> Tensor:
    if features.shape != baseline.shape:
        raise ValueError("features and baseline must have matching shapes")
    if features.ndim != 2:
        raise ValueError("features must have rank two")
    if steps < 2:
        raise ValueError("at least two integration steps are required")
    differences = features - baseline
    accumulated = torch.zeros_like(features)
    alphas = torch.linspace(
        0.0,
        1.0,
        steps,
        device=features.device,
        dtype=features.dtype,
    )
    for alpha in alphas:
        interpolated = baseline + alpha * differences
        interpolated.requires_grad_(True)
        output = model(interpolated)
        gradient = torch.autograd.grad(
            output.sum(),
            interpolated,
            create_graph=False,
            retain_graph=False,
        )[0]
        accumulated += gradient
    average = accumulated / steps
    return differences * average


def permutation_importance(
    predict: callable,
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    metric: callable,
    repeats: int = 20,
    seed: int = 1,
) -> NDArray[np.float64]:
    matrix = np.asarray(features, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if matrix.ndim != 2 or truth.shape != (matrix.shape[0],):
        raise ValueError("feature and label shapes differ")
    generator = np.random.default_rng(seed)
    baseline = float(metric(truth, predict(matrix)))
    importances = np.zeros(matrix.shape[1], dtype=np.float64)
    for feature_index in range(matrix.shape[1]):
        losses: list[float] = []
        for _ in range(repeats):
            perturbed = matrix.copy()
            generator.shuffle(perturbed[:, feature_index])
            score = float(metric(truth, predict(perturbed)))
            losses.append(baseline - score)
        importances[feature_index] = float(np.mean(losses))
    return importances


def site_stability(
    rankings: dict[str, NDArray[np.int64]],
    feature_count: int,
    top_k: int = 20,
) -> NDArray[np.float64]:
    if not rankings:
        raise ValueError("rankings cannot be empty")
    counts = np.zeros(feature_count, dtype=np.float64)
    for ranking in rankings.values():
        values = np.asarray(ranking, dtype=np.int64)
        selected = values[:top_k]
        if np.any((selected < 0) | (selected >= feature_count)):
            raise ValueError("ranking contains an invalid feature index")
        counts[selected] += 1.0
    return counts / len(rankings)


def pairwise_rank_correlation(
    rankings: dict[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    sites = sorted(rankings)
    if len(sites) < 2:
        raise ValueError("at least two site rankings are required")
    dimensions = {np.asarray(rankings[site]).shape for site in sites}
    if len(dimensions) != 1:
        raise ValueError("ranking shapes differ")
    matrix = np.stack([rankings[site] for site in sites])
    return np.corrcoef(matrix)


def jaccard_matrix(
    selections: dict[str, set[int]],
) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    sites = tuple(sorted(selections))
    matrix = np.zeros((len(sites), len(sites)), dtype=np.float64)
    for left_index, left_site in enumerate(sites):
        for right_index, right_site in enumerate(sites):
            left = selections[left_site]
            right = selections[right_site]
            union = left | right
            intersection = left & right
            matrix[left_index, right_index] = (
                len(intersection) / len(union) if union else 1.0
            )
    return sites, matrix


def ablation_masks(
    categories: dict[str, tuple[int, ...]],
    dimension: int = 128,
) -> dict[str, NDArray[np.bool_]]:
    masks: dict[str, NDArray[np.bool_]] = {}
    all_features = set(range(dimension))
    for category, indices in categories.items():
        excluded = set(indices)
        if not excluded.issubset(all_features):
            raise ValueError(f"category contains out-of-range indices: {category}")
        mask = np.ones(dimension, dtype=np.bool_)
        mask[list(excluded)] = False
        masks[f"without_{category}"] = mask
    masks["all_features"] = np.ones(dimension, dtype=np.bool_)
    return masks


def temporal_jaccard(
    reference: set[int],
    temporal: dict[str, set[int]],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for period, selected in temporal.items():
        union = reference | selected
        values[period] = len(reference & selected) / len(union) if union else 1.0
    return values


def leave_one_feature_out(
    predict: callable,
    features: NDArray[np.float64],
    baseline_value: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix = np.asarray(features, dtype=np.float64)
    baseline = np.asarray(baseline_value, dtype=np.float64)
    if baseline.shape != (matrix.shape[1],):
        raise ValueError("baseline feature shape mismatch")
    original = np.asarray(predict(matrix), dtype=np.float64)
    changes = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=np.float64)
    for feature_index in range(matrix.shape[1]):
        modified = matrix.copy()
        modified[:, feature_index] = baseline[feature_index]
        prediction = np.asarray(predict(modified), dtype=np.float64)
        changes[:, feature_index] = original - prediction
    return changes


def interaction_strength(
    attributions: NDArray[np.float64],
    interaction_indices: tuple[int, ...],
) -> float:
    values = np.asarray(attributions, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("attributions must have rank two")
    if not interaction_indices:
        return 0.0
    selected = values[:, interaction_indices]
    total = np.abs(values).sum()
    return float(np.abs(selected).sum() / total) if total > 0.0 else 0.0


def normalize_importance(
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    magnitude = np.abs(array)
    total = magnitude.sum()
    return magnitude / total if total > 0.0 else magnitude


def aggregate_site_importance(
    importances: dict[str, NDArray[np.float64]],
    sample_counts: dict[str, int],
) -> NDArray[np.float64]:
    if importances.keys() != sample_counts.keys():
        raise ValueError("importance and sample-count sites differ")
    if not importances:
        raise ValueError("site importances cannot be empty")
    dimensions = {values.shape for values in importances.values()}
    if len(dimensions) != 1:
        raise ValueError("importance shapes differ")
    total = sum(sample_counts.values())
    if total <= 0:
        raise ValueError("total sample count must be positive")
    result = np.zeros(next(iter(dimensions)), dtype=np.float64)
    for site, values in importances.items():
        result += values * (sample_counts[site] / total)
    return result


def bootstrap_feature_stability(
    selections: list[set[int]],
    dimension: int,
) -> NDArray[np.float64]:
    if not selections:
        raise ValueError("selections cannot be empty")
    counts = np.zeros(dimension, dtype=np.float64)
    for selection in selections:
        indices = np.asarray(sorted(selection), dtype=np.int64)
        if np.any((indices < 0) | (indices >= dimension)):
            raise ValueError("selection contains invalid indices")
        counts[indices] += 1.0
    return counts / len(selections)


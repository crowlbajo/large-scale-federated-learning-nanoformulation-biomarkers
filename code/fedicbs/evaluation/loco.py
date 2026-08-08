from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fedicbs.evaluation.metrics import binary_metrics
from fedicbs.types import MetricBundle


@dataclass(frozen=True)
class LocoFold:
    held_out_site: str
    training_indices: NDArray[np.int64]
    evaluation_indices: NDArray[np.int64]


@dataclass(frozen=True)
class LocoResult:
    held_out_site: str
    metrics: MetricBundle
    training_size: int
    evaluation_size: int
    invariant_features: tuple[int, ...]


def build_loco_folds(frame: pd.DataFrame) -> tuple[LocoFold, ...]:
    if "site" not in frame or "outcome" not in frame:
        raise ValueError("frame requires site and outcome columns")
    folds: list[LocoFold] = []
    sites = sorted(str(value) for value in frame["site"].unique())
    if len(sites) < 5:
        raise ValueError("LOCO requires at least five sites")
    for site in sites:
        evaluation = frame.index[frame["site"].astype(str) == site].to_numpy(dtype=np.int64)
        training = frame.index[frame["site"].astype(str) != site].to_numpy(dtype=np.int64)
        if np.unique(frame.loc[evaluation, "outcome"]).size != 2:
            raise ValueError(f"held-out site lacks both outcome classes: {site}")
        folds.append(LocoFold(site, training, evaluation))
    return tuple(folds)


def run_loco(
    frame: pd.DataFrame,
    fit_predict: Callable[
        [pd.DataFrame, pd.DataFrame],
        tuple[NDArray[np.float64], tuple[int, ...]],
    ],
) -> tuple[LocoResult, ...]:
    results: list[LocoResult] = []
    for fold in build_loco_folds(frame):
        training = frame.loc[fold.training_indices].copy()
        evaluation = frame.loc[fold.evaluation_indices].copy()
        probabilities, invariant = fit_predict(training, evaluation)
        labels = evaluation["outcome"].to_numpy(dtype=np.int64)
        results.append(
            LocoResult(
                held_out_site=fold.held_out_site,
                metrics=binary_metrics(labels, probabilities),
                training_size=len(training),
                evaluation_size=len(evaluation),
                invariant_features=invariant,
            )
        )
    return tuple(results)


def summarize_loco(results: tuple[LocoResult, ...]) -> dict[str, float]:
    if not results:
        raise ValueError("LOCO results cannot be empty")
    aucs = np.asarray([result.metrics.auc for result in results], dtype=np.float64)
    sensitivities = np.asarray(
        [result.metrics.sensitivity for result in results],
        dtype=np.float64,
    )
    specificities = np.asarray(
        [result.metrics.specificity for result in results],
        dtype=np.float64,
    )
    feature_sets = [set(result.invariant_features) for result in results]
    pairwise_jaccard: list[float] = []
    for left_index in range(len(feature_sets)):
        for right_index in range(left_index + 1, len(feature_sets)):
            union = feature_sets[left_index] | feature_sets[right_index]
            intersection = feature_sets[left_index] & feature_sets[right_index]
            pairwise_jaccard.append(len(intersection) / len(union) if union else 1.0)
    return {
        "mean_auc": float(aucs.mean()),
        "std_auc": float(aucs.std(ddof=1)),
        "minimum_auc": float(aucs.min()),
        "maximum_auc": float(aucs.max()),
        "mean_sensitivity": float(sensitivities.mean()),
        "mean_specificity": float(specificities.mean()),
        "mean_feature_jaccard": float(np.mean(pairwise_jaccard)),
    }


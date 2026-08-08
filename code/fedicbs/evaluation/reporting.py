from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from fedicbs.types import ConfidenceInterval, InvarianceResult, MetricBundle, RoundRecord


def metric_document(
    metrics: MetricBundle,
    intervals: dict[str, ConfidenceInterval],
    site_metrics: dict[str, MetricBundle],
) -> dict[str, Any]:
    return {
        "overall": asdict(metrics),
        "confidence_intervals": {
            name: asdict(interval)
            for name, interval in intervals.items()
        },
        "sites": {
            site: asdict(bundle)
            for site, bundle in sorted(site_metrics.items())
        },
    }


def write_json_atomic(path: str | Path, document: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(document, indent=2, sort_keys=True, allow_nan=False)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_invariance_csv(
    path: str | Path,
    results: tuple[InvarianceResult, ...],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    fields = [
        "feature_index",
        "pooled_coefficient",
        "pooled_standard_error",
        "z_score",
        "q_statistic",
        "predictivity_p_value",
        "invariance_p_value",
        "decision",
        "contributing_sites",
    ]
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for result in results:
                writer.writerow(
                    {
                        "feature_index": result.feature_index,
                        "pooled_coefficient": result.pooled_coefficient,
                        "pooled_standard_error": result.pooled_standard_error,
                        "z_score": result.z_score,
                        "q_statistic": result.q_statistic,
                        "predictivity_p_value": result.predictivity_p_value,
                        "invariance_p_value": result.invariance_p_value,
                        "decision": result.decision.value,
                        "contributing_sites": "|".join(result.contributing_sites),
                    }
                )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_convergence_csv(
    path: str | Path,
    records: list[RoundRecord],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(
        [
            [
                record.round_index,
                record.training_loss,
                record.validation_auc,
                record.gradient_norm,
                record.invariant_count,
                record.mean_q,
            ]
            for record in records
        ],
        dtype=np.float64,
    )
    header = "round,train_loss,val_auc,gradient_norm,invariant_count,mean_q"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savetxt(temporary, matrix, delimiter=",", header=header, comments="")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from fedicbs.entrypoints.common import configure_logging, parser
from fedicbs.evaluation.metrics import (
    binary_metrics,
    stratified_bootstrap_bundle,
)
from fedicbs.evaluation.reporting import metric_document, write_json_atomic
from fedicbs.settings import load_settings


LOGGER = logging.getLogger(__name__)


def main() -> None:
    argument_parser = parser("Evaluate treatment response predictions")
    argument_parser.add_argument("--predictions", type=Path, required=True)
    arguments = argument_parser.parse_args()
    configure_logging(arguments.log_level)
    settings = load_settings(arguments.config, arguments.override)
    values = np.genfromtxt(
        arguments.predictions,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    labels = np.asarray(values["label"], dtype=np.int64)
    probabilities = np.asarray(values["probability"], dtype=np.float64)
    metrics = binary_metrics(labels, probabilities)
    intervals = stratified_bootstrap_bundle(
        labels,
        probabilities,
        settings.bootstrap_replicates,
        settings.seeds[0],
    )
    document = metric_document(metrics, intervals, {})
    destination = arguments.output / "metrics.json"
    write_json_atomic(destination, document)
    LOGGER.info("wrote evaluation metrics to %s", destination)


if __name__ == "__main__":
    main()


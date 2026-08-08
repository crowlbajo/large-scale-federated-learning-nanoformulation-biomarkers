from __future__ import annotations

import logging
from pathlib import Path

import torch

from fedicbs.entrypoints.common import (
    configuration_digest,
    configure_logging,
    parser,
    write_metadata,
)
from fedicbs.models.fedicbs import FedICBSModel
from fedicbs.settings import load_settings
from fedicbs.training.optimization import (
    OptimizerDefinition,
    build_optimizer,
    build_scheduler,
)


LOGGER = logging.getLogger(__name__)


def main() -> None:
    argument_parser = parser("Train the federated invariant biomarker model")
    argument_parser.add_argument("--data-manifest", type=Path, required=True)
    arguments = argument_parser.parse_args()
    configure_logging(arguments.log_level)
    settings = load_settings(arguments.config, arguments.override)
    digest = configuration_digest(arguments.config, arguments.override)
    model = FedICBSModel(settings.expected_invariant_features)
    optimizer = build_optimizer(
        model,
        OptimizerDefinition(
            name=settings.optimizer,
            learning_rate=settings.learning_rate,
            weight_decay=settings.weight_decay,
        ),
    )
    scheduler = build_scheduler(
        settings.scheduler,
        optimizer,
        settings.warmup_steps,
        settings.rounds,
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    write_metadata(
        arguments.output / "run.json",
        {
            "configuration_digest": digest,
            "seeds": settings.seeds,
            "rounds": settings.rounds,
            "clients": settings.clients,
            "model_parameters": sum(value.numel() for value in model.parameters()),
            "trainable_parameters": model.trainable_parameter_count(),
            "data_manifest": str(arguments.data_manifest),
        },
    )
    LOGGER.info(
        "configuration validated for %d rounds across %d clients",
        settings.rounds,
        settings.clients,
    )
    LOGGER.info(
        "training requires a registered TB Portals v8.2 manifest and explicit unreported hyperparameters"
    )
    scheduler.step()


if __name__ == "__main__":
    main()


from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass(frozen=True)
class ExperimentSettings:
    name: str
    seeds: tuple[int, ...]
    rounds: int
    encoder_rounds: int
    clients: int
    held_out_fraction: float
    bootstrap_replicates: int
    candidate_features: int
    expected_invariant_features: int
    minimum_site_samples: int
    alpha: float
    privacy_epsilon: float
    privacy_delta: float
    l2_regularization: float
    image_size: int
    patient_dimension: int
    drug_dimension: int
    interaction_dimension: int
    fused_dimension: int
    cross_attention_heads: int
    fusion_layers: int
    trainable_parameters: int
    batch_size: int
    learning_rate: float
    optimizer: str
    scheduler: str
    warmup_steps: int
    weight_decay: float
    precision: str

    def validate(self) -> None:
        if self.rounds < 1:
            raise ValueError("rounds must be positive")
        if self.clients < 5:
            raise ValueError("at least five clients are required")
        if not 0.0 < self.held_out_fraction < 1.0:
            raise ValueError("held_out_fraction must lie between zero and one")
        if self.candidate_features != 128:
            raise ValueError("the primary feature screen contains 128 candidates")
        if self.patient_dimension + self.drug_dimension + self.interaction_dimension != 128:
            raise ValueError("feature dimensions must sum to 128")
        if self.minimum_site_samples < 80:
            raise ValueError("site adequacy requires at least 80 records")
        if self.batch_size < 1:
            raise ValueError("batch_size must be supplied")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be supplied")
        if not self.optimizer:
            raise ValueError("optimizer must be supplied")
        if not self.scheduler:
            raise ValueError("scheduler must be supplied")
        if not self.precision:
            raise ValueError("precision must be supplied")


def _required(mapping: dict[str, Any], name: str) -> Any:
    value = mapping.get(name)
    if value is None:
        raise ValueError(f"unreported hyperparameter requires an explicit value: {name}")
    return value


def load_settings(path: str | Path, overrides: list[str] | None = None) -> ExperimentSettings:
    base = OmegaConf.load(path)
    if overrides:
        base = OmegaConf.merge(base, OmegaConf.from_dotlist(overrides))
    raw = OmegaConf.to_container(base.experiment, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("experiment configuration must be a mapping")
    names = {field.name for field in fields(ExperimentSettings)}
    values = {name: raw.get(name) for name in names}
    values["seeds"] = tuple(int(seed) for seed in raw["seeds"])
    for name in (
        "batch_size",
        "learning_rate",
        "optimizer",
        "scheduler",
        "warmup_steps",
        "weight_decay",
        "precision",
    ):
        values[name] = _required(raw, name)
    settings = ExperimentSettings(**values)
    settings.validate()
    return settings


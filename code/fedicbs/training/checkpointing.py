from __future__ import annotations

import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


@dataclass(frozen=True)
class CheckpointMetadata:
    epoch: int
    round_index: int
    global_step: int
    seed: int
    best_metric: float
    invariant_indices: tuple[int, ...]
    configuration_digest: str


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    metadata: CheckpointMetadata,
    scheduler: Any | None = None,
    scaler: Any | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metadata": asdict(metadata),
        "rng_state": capture_rng_state(),
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> CheckpointMetadata:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    restore_rng_state(payload["rng_state"])
    raw = payload["metadata"]
    return CheckpointMetadata(
        epoch=int(raw["epoch"]),
        round_index=int(raw["round_index"]),
        global_step=int(raw["global_step"]),
        seed=int(raw["seed"]),
        best_metric=float(raw["best_metric"]),
        invariant_indices=tuple(int(value) for value in raw["invariant_indices"]),
        configuration_digest=str(raw["configuration_digest"]),
    )


def write_run_manifest(
    path: str | Path,
    metadata: CheckpointMetadata,
    extra: dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = {"metadata": asdict(metadata), "extra": extra}
    encoded = json.dumps(document, sort_keys=True, indent=2)
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


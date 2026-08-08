from __future__ import annotations

import contextlib
import logging
import random
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer

from fedicbs.types import MetricBundle


LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class PrecisionPolicy:
    name: str
    enabled: bool
    dtype: torch.dtype

    @classmethod
    def from_name(cls, name: str) -> PrecisionPolicy:
        normalized = name.lower()
        if normalized == "fp32":
            return cls(name, False, torch.float32)
        if normalized == "fp16":
            return cls(name, True, torch.float16)
        if normalized == "bf16":
            return cls(name, True, torch.bfloat16)
        raise ValueError(f"unsupported precision: {name}")


@dataclass(frozen=True)
class StepOutput:
    loss: float
    gradient_norm: float
    examples: int


@dataclass(frozen=True)
class EpochOutput:
    mean_loss: float
    mean_gradient_norm: float
    examples: int
    steps: int


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_function: Callable[[Tensor, Tensor], Tensor],
        device: torch.device,
        precision: PrecisionPolicy,
        gradient_clip_norm: float | None,
        gradient_accumulation: int = 1,
    ) -> None:
        if gradient_accumulation < 1:
            raise ValueError("gradient_accumulation must be positive")
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.precision = precision
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_accumulation = gradient_accumulation
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=precision.enabled and precision.dtype is torch.float16
        )
        self.global_step = 0

    def train_epoch(
        self,
        batches: Iterable[tuple[object, Tensor]],
    ) -> EpochOutput:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        norms: list[float] = []
        examples = 0
        pending = 0
        for index, (inputs, targets) in enumerate(batches):
            targets = targets.to(self.device, non_blocking=True)
            context = self._autocast_context()
            synchronization = self._synchronization_context(index)
            with synchronization:
                with context:
                    logits = self.model(inputs)
                    loss = self.loss_function(logits, targets)
                    scaled_loss = loss / self.gradient_accumulation
                self.scaler.scale(scaled_loss).backward()
            pending += 1
            if pending == self.gradient_accumulation:
                norm = self._optimizer_step()
                norms.append(norm)
                pending = 0
            losses.append(float(loss.detach()))
            examples += int(targets.shape[0])
        if pending:
            norm = self._optimizer_step()
            norms.append(norm)
        if not losses:
            raise ValueError("training loader produced no batches")
        return EpochOutput(
            mean_loss=float(np.mean(losses)),
            mean_gradient_norm=float(np.mean(norms)) if norms else 0.0,
            examples=examples,
            steps=len(losses),
        )

    def _optimizer_step(self) -> float:
        self.scaler.unscale_(self.optimizer)
        norm = self._gradient_norm()
        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_norm,
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        return norm

    def _gradient_norm(self) -> float:
        squared = torch.zeros((), dtype=torch.float64, device=self.device)
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                squared += torch.square(parameter.grad.detach().to(torch.float64)).sum()
        return float(torch.sqrt(squared))

    def _autocast_context(self) -> contextlib.AbstractContextManager[None]:
        if self.device.type != "cuda" or not self.precision.enabled:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.precision.dtype)

    def _synchronization_context(
        self,
        batch_index: int,
    ) -> contextlib.AbstractContextManager[None]:
        should_synchronize = (batch_index + 1) % self.gradient_accumulation == 0
        if isinstance(self.model, DistributedDataParallel) and not should_synchronize:
            return self.model.no_sync()
        return contextlib.nullcontext()

    @torch.no_grad()
    def predict(
        self,
        batches: Iterable[tuple[object, Tensor]],
    ) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        labels: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        for inputs, targets in batches:
            with self._autocast_context():
                logits = self.model(inputs)
            probability = torch.sigmoid(logits)
            labels.append(targets.detach().cpu().numpy())
            probabilities.append(probability.detach().float().cpu().numpy())
        if not labels:
            raise ValueError("evaluation loader produced no batches")
        return np.concatenate(labels), np.concatenate(probabilities)


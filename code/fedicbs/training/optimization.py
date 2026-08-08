from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi
from typing import Iterable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


@dataclass(frozen=True)
class OptimizerDefinition:
    name: str
    learning_rate: float
    weight_decay: float
    momentum: float = 0.9
    beta_one: float = 0.9
    beta_two: float = 0.999
    epsilon: float = 1e-8


def parameter_groups(
    model: nn.Module,
    weight_decay: float,
) -> list[dict[str, object]]:
    decayed: list[nn.Parameter] = []
    excluded: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            excluded.append(parameter)
        else:
            decayed.append(parameter)
    return [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": excluded, "weight_decay": 0.0},
    ]


def build_optimizer(
    model: nn.Module,
    definition: OptimizerDefinition,
) -> Optimizer:
    if definition.learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")
    groups = parameter_groups(model, definition.weight_decay)
    name = definition.name.lower()
    if name == "adamw":
        return torch.optim.AdamW(
            groups,
            lr=definition.learning_rate,
            betas=(definition.beta_one, definition.beta_two),
            eps=definition.epsilon,
        )
    if name == "adam":
        return torch.optim.Adam(
            groups,
            lr=definition.learning_rate,
            betas=(definition.beta_one, definition.beta_two),
            eps=definition.epsilon,
        )
    if name == "sgd":
        return torch.optim.SGD(
            groups,
            lr=definition.learning_rate,
            momentum=definition.momentum,
            nesterov=True,
        )
    if name == "rmsprop":
        return torch.optim.RMSprop(
            groups,
            lr=definition.learning_rate,
            momentum=definition.momentum,
            eps=definition.epsilon,
        )
    raise ValueError(f"unsupported optimizer: {definition.name}")


class WarmupCosineScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        minimum_ratio: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if warmup_steps < 0 or total_steps < 1:
            raise ValueError("invalid scheduler steps")
        if warmup_steps >= total_steps:
            raise ValueError("warmup must end before training")
        if not 0.0 <= minimum_ratio <= 1.0:
            raise ValueError("minimum ratio must lie between zero and one")
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.minimum_ratio = minimum_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step = self.last_epoch
        if self.warmup_steps and step < self.warmup_steps:
            ratio = (step + 1) / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps,
                1,
            )
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + cos(pi * progress))
            ratio = self.minimum_ratio + (1.0 - self.minimum_ratio) * cosine
        return [base_lr * ratio for base_lr in self.base_lrs]


class PolynomialScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        power: float = 1.0,
        minimum_ratio: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.total_steps = total_steps
        self.power = power
        self.minimum_ratio = minimum_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        progress = min(max(self.last_epoch / max(self.total_steps, 1), 0.0), 1.0)
        ratio = (1.0 - progress) ** self.power
        ratio = self.minimum_ratio + (1.0 - self.minimum_ratio) * ratio
        return [base_lr * ratio for base_lr in self.base_lrs]


def build_scheduler(
    name: str,
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LRScheduler:
    normalized = name.lower()
    if normalized == "cosine":
        return WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    if normalized == "linear":
        return PolynomialScheduler(optimizer, total_steps, power=1.0)
    if normalized == "polynomial":
        return PolynomialScheduler(optimizer, total_steps, power=2.0)
    if normalized == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    raise ValueError(f"unsupported scheduler: {name}")


def exponential_moving_average(
    target: dict[str, torch.Tensor],
    source: dict[str, torch.Tensor],
    decay: float,
) -> None:
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must lie in [0, 1)")
    if target.keys() != source.keys():
        raise ValueError("state keys differ")
    for name in target:
        if target[name].is_floating_point():
            target[name].mul_(decay).add_(source[name], alpha=1.0 - decay)
        else:
            target[name].copy_(source[name])


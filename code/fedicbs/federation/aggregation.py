from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from fedicbs.types import ClientUpdate


State = OrderedDict[str, Tensor]


def _validate_updates(updates: list[ClientUpdate]) -> tuple[str, ...]:
    if not updates:
        raise ValueError("at least one client update is required")
    keys = tuple(updates[0].state.keys())
    for update in updates:
        if update.sample_count < 1:
            raise ValueError("client sample counts must be positive")
        if tuple(update.state.keys()) != keys:
            raise ValueError("client state keys differ")
        for key in keys:
            if update.state[key].shape != updates[0].state[key].shape:
                raise ValueError(f"client tensor shape differs for {key}")
    return keys


def sample_weighted_average(updates: list[ClientUpdate]) -> State:
    keys = _validate_updates(updates)
    total = float(sum(update.sample_count for update in updates))
    result: State = OrderedDict()
    for key in keys:
        reference = updates[0].state[key]
        if not reference.is_floating_point():
            result[key] = reference.detach().clone()
            continue
        accumulator = torch.zeros_like(reference, dtype=torch.float64)
        for update in updates:
            weight = update.sample_count / total
            accumulator.add_(update.state[key].detach().to(torch.float64), alpha=weight)
        result[key] = accumulator.to(dtype=reference.dtype)
    return result


def uniform_average(updates: list[ClientUpdate]) -> State:
    keys = _validate_updates(updates)
    result: State = OrderedDict()
    count = float(len(updates))
    for key in keys:
        reference = updates[0].state[key]
        if not reference.is_floating_point():
            result[key] = reference.detach().clone()
            continue
        accumulator = torch.zeros_like(reference, dtype=torch.float64)
        for update in updates:
            accumulator.add_(update.state[key].detach().to(torch.float64), alpha=1.0 / count)
        result[key] = accumulator.to(dtype=reference.dtype)
    return result


def coordinate_median(updates: list[ClientUpdate]) -> State:
    keys = _validate_updates(updates)
    result: State = OrderedDict()
    for key in keys:
        reference = updates[0].state[key]
        if not reference.is_floating_point():
            result[key] = reference.detach().clone()
            continue
        stack = torch.stack(
            [update.state[key].detach().to(torch.float64) for update in updates],
            dim=0,
        )
        result[key] = stack.median(dim=0).values.to(dtype=reference.dtype)
    return result


def trimmed_mean(updates: list[ClientUpdate], trim_fraction: float) -> State:
    keys = _validate_updates(updates)
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must lie in [0, 0.5)")
    trim = int(np.floor(len(updates) * trim_fraction))
    if len(updates) - 2 * trim < 1:
        raise ValueError("trimming removes every client")
    result: State = OrderedDict()
    for key in keys:
        reference = updates[0].state[key]
        if not reference.is_floating_point():
            result[key] = reference.detach().clone()
            continue
        stack = torch.stack(
            [update.state[key].detach().to(torch.float64) for update in updates],
            dim=0,
        )
        sorted_values = stack.sort(dim=0).values
        retained = sorted_values[trim : len(updates) - trim]
        result[key] = retained.mean(dim=0).to(dtype=reference.dtype)
    return result


def geometric_median(
    updates: list[ClientUpdate],
    iterations: int = 50,
    tolerance: float = 1e-6,
) -> State:
    keys = _validate_updates(updates)
    vectors, shapes, dtypes = _flatten_updates(updates, keys)
    estimate = vectors.mean(dim=0)
    for _ in range(iterations):
        distances = torch.linalg.vector_norm(vectors - estimate, dim=1).clamp_min(1e-12)
        weights = 1.0 / distances
        next_estimate = (vectors * weights[:, None]).sum(dim=0) / weights.sum()
        change = torch.linalg.vector_norm(next_estimate - estimate)
        estimate = next_estimate
        if float(change) < tolerance:
            break
    return _unflatten_state(estimate, keys, shapes, dtypes, updates[0].state)


def _flatten_updates(
    updates: list[ClientUpdate],
    keys: tuple[str, ...],
) -> tuple[Tensor, dict[str, torch.Size], dict[str, torch.dtype]]:
    floating_keys = [key for key in keys if updates[0].state[key].is_floating_point()]
    shapes = {key: updates[0].state[key].shape for key in floating_keys}
    dtypes = {key: updates[0].state[key].dtype for key in floating_keys}
    rows = [
        torch.cat(
            [
                update.state[key].detach().to(torch.float64).reshape(-1)
                for key in floating_keys
            ]
        )
        for update in updates
    ]
    return torch.stack(rows), shapes, dtypes


def _unflatten_state(
    vector: Tensor,
    keys: tuple[str, ...],
    shapes: dict[str, torch.Size],
    dtypes: dict[str, torch.dtype],
    reference: dict[str, Tensor],
) -> State:
    result: State = OrderedDict()
    offset = 0
    for key in keys:
        if key not in shapes:
            result[key] = reference[key].detach().clone()
            continue
        size = int(np.prod(shapes[key]))
        result[key] = (
            vector[offset : offset + size]
            .reshape(shapes[key])
            .to(dtype=dtypes[key])
        )
        offset += size
    if offset != vector.numel():
        raise RuntimeError("state vector was not consumed")
    return result


@dataclass
class ScaffoldControl:
    server: State
    clients: dict[str, State]

    @classmethod
    def initialize(cls, state: dict[str, Tensor], sites: list[str]) -> ScaffoldControl:
        server: State = OrderedDict(
            (key, torch.zeros_like(value))
            for key, value in state.items()
            if value.is_floating_point()
        )
        clients = {
            site: OrderedDict((key, value.clone()) for key, value in server.items())
            for site in sites
        }
        return cls(server, clients)

    def correct_gradient(self, site: str, name: str, gradient: Tensor) -> Tensor:
        return gradient - self.clients[site][name] + self.server[name]

    def update_client(
        self,
        site: str,
        global_state: dict[str, Tensor],
        local_state: dict[str, Tensor],
        local_steps: int,
        learning_rate: float,
    ) -> None:
        if local_steps < 1 or learning_rate <= 0.0:
            raise ValueError("local_steps and learning_rate must be positive")
        for key in self.server:
            difference = global_state[key] - local_state[key]
            correction = difference / (local_steps * learning_rate)
            self.clients[site][key] = self.clients[site][key] - self.server[key] + correction

    def update_server(self) -> None:
        if not self.clients:
            raise ValueError("no client controls are available")
        for key in self.server:
            stack = torch.stack([state[key] for state in self.clients.values()])
            self.server[key] = stack.mean(dim=0)


def state_distance(left: dict[str, Tensor], right: dict[str, Tensor]) -> float:
    if left.keys() != right.keys():
        raise ValueError("state keys differ")
    squared = torch.zeros((), dtype=torch.float64)
    for key in left:
        if left[key].is_floating_point():
            difference = left[key].to(torch.float64) - right[key].to(torch.float64)
            squared += torch.square(difference).sum()
    return float(torch.sqrt(squared))


from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from fedicbs.federation.aggregation import (
    State,
    coordinate_median,
    sample_weighted_average,
    uniform_average,
)
from fedicbs.types import ClientUpdate


@dataclass(frozen=True)
class AggregationContext:
    round_index: int
    total_rounds: int
    server_state: dict[str, Tensor]
    previous_server_state: dict[str, Tensor] | None


class FederatedStrategy:
    def aggregate(
        self,
        updates: list[ClientUpdate],
        context: AggregationContext,
    ) -> State:
        raise NotImplementedError


class FedAvgStrategy(FederatedStrategy):
    def aggregate(
        self,
        updates: list[ClientUpdate],
        context: AggregationContext,
    ) -> State:
        return sample_weighted_average(updates)


class UniformStrategy(FederatedStrategy):
    def aggregate(
        self,
        updates: list[ClientUpdate],
        context: AggregationContext,
    ) -> State:
        return uniform_average(updates)


class MedianStrategy(FederatedStrategy):
    def aggregate(
        self,
        updates: list[ClientUpdate],
        context: AggregationContext,
    ) -> State:
        return coordinate_median(updates)


@dataclass
class FedProxStrategy(FederatedStrategy):
    proximal_strength: float

    def aggregate(
        self,
        updates: list[ClientUpdate],
        context: AggregationContext,
    ) -> State:
        if self.proximal_strength < 0.0:
            raise ValueError("proximal strength cannot be negative")
        return sample_weighted_average(updates)

    def penalty(
        self,
        local_parameters: dict[str, Tensor],
        server_parameters: dict[str, Tensor],
    ) -> Tensor:
        values: list[Tensor] = []
        for name, local in local_parameters.items():
            server = server_parameters[name].to(local.device)
            if local.is_floating_point():
                values.append(torch.square(local - server).sum())
        if not values:
            return torch.zeros(())
        return 0.5 * self.proximal_strength * torch.stack(values).sum()


@dataclass
class FedNovaStrategy(FederatedStrategy):
    local_steps: dict[str, int]

    def aggregate(
        self,
        updates: list[ClientUpdate],
        context: AggregationContext,
    ) -> State:
        if not updates:
            raise ValueError("updates cannot be empty")
        coefficients = {}
        denominator = 0.0
        total_samples = sum(update.sample_count for update in updates)
        for update in updates:
            steps = self.local_steps.get(update.site)
            if steps is None or steps < 1:
                raise ValueError(f"missing local steps for {update.site}")
            client_weight = update.sample_count / total_samples
            coefficient = client_weight / steps
            coefficients[update.site] = coefficient
            denominator += coefficient
        result: State = OrderedDict()
        for name, server in context.server_state.items():
            if not server.is_floating_point():
                result[name] = server.detach().clone()
                continue
            normalized_delta = torch.zeros_like(server, dtype=torch.float64)
            for update in updates:
                delta = update.state[name].to(torch.float64) - server.to(torch.float64)
                normalized_delta += coefficients[update.site] * delta
            result[name] = (
                server.to(torch.float64) + normalized_delta / denominator
            ).to(server.dtype)
        return result


@dataclass
class FedDynStrategy(FederatedStrategy):
    regularization: float
    dual_states: dict[str, State]

    @classmethod
    def create(cls, regularization: float) -> FedDynStrategy:
        if regularization <= 0.0:
            raise ValueError("regularization must be positive")
        return cls(regularization, {})

    def aggregate(
        self,
        updates: list[ClientUpdate],
        context: AggregationContext,
    ) -> State:
        average = sample_weighted_average(updates)
        for update in updates:
            if update.site not in self.dual_states:
                self.dual_states[update.site] = OrderedDict(
                    (name, torch.zeros_like(value))
                    for name, value in context.server_state.items()
                    if value.is_floating_point()
                )
            dual = self.dual_states[update.site]
            for name in dual:
                dual[name] = dual[name] + self.regularization * (
                    update.state[name] - context.server_state[name]
                )
        result: State = OrderedDict()
        for name, value in average.items():
            if not value.is_floating_point():
                result[name] = value
                continue
            mean_dual = torch.stack(
                [dual[name] for dual in self.dual_states.values()]
            ).mean(dim=0)
            result[name] = value - mean_dual / self.regularization
        return result


@dataclass
class QFedAvgStrategy(FederatedStrategy):
    fairness_power: float
    epsilon: float = 1e-10

    def aggregate(
        self,
        updates: list[ClientUpdate],
        context: AggregationContext,
    ) -> State:
        if self.fairness_power < 0.0:
            raise ValueError("fairness_power cannot be negative")
        weights = torch.as_tensor(
            [
                max(update.loss, self.epsilon) ** self.fairness_power
                * update.sample_count
                for update in updates
            ],
            dtype=torch.float64,
        )
        weights /= weights.sum()
        result: State = OrderedDict()
        for name, server in context.server_state.items():
            if not server.is_floating_point():
                result[name] = server.detach().clone()
                continue
            accumulator = torch.zeros_like(server, dtype=torch.float64)
            for index, update in enumerate(updates):
                accumulator += weights[index] * update.state[name].to(torch.float64)
            result[name] = accumulator.to(server.dtype)
        return result


@dataclass
class MomentumStrategy(FederatedStrategy):
    momentum: float
    velocity: State | None = None

    def aggregate(
        self,
        updates: list[ClientUpdate],
        context: AggregationContext,
    ) -> State:
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must lie in [0, 1)")
        average = sample_weighted_average(updates)
        if self.velocity is None:
            self.velocity = OrderedDict(
                (name, torch.zeros_like(value))
                for name, value in average.items()
                if value.is_floating_point()
            )
        result: State = OrderedDict()
        for name, server in context.server_state.items():
            if not server.is_floating_point():
                result[name] = server.detach().clone()
                continue
            delta = average[name] - server
            self.velocity[name] = self.momentum * self.velocity[name] + delta
            result[name] = server + self.velocity[name]
        return result


@dataclass
class AdaptiveServerStrategy(FederatedStrategy):
    learning_rate: float
    beta_one: float
    beta_two: float
    epsilon: float
    first_moment: State | None = None
    second_moment: State | None = None

    def aggregate(
        self,
        updates: list[ClientUpdate],
        context: AggregationContext,
    ) -> State:
        average = sample_weighted_average(updates)
        if self.first_moment is None or self.second_moment is None:
            self.first_moment = OrderedDict()
            self.second_moment = OrderedDict()
            for name, value in context.server_state.items():
                if value.is_floating_point():
                    self.first_moment[name] = torch.zeros_like(value)
                    self.second_moment[name] = torch.zeros_like(value)
        result: State = OrderedDict()
        for name, server in context.server_state.items():
            if not server.is_floating_point():
                result[name] = server.detach().clone()
                continue
            gradient = server - average[name]
            self.first_moment[name] = (
                self.beta_one * self.first_moment[name]
                + (1.0 - self.beta_one) * gradient
            )
            self.second_moment[name] = (
                self.beta_two * self.second_moment[name]
                + (1.0 - self.beta_two) * torch.square(gradient)
            )
            update = self.first_moment[name] / (
                torch.sqrt(self.second_moment[name]) + self.epsilon
            )
            result[name] = server - self.learning_rate * update
        return result


@dataclass
class FedPerPartition:
    shared_prefixes: tuple[str, ...]
    personal_states: dict[str, State]

    def shared_state(self, state: dict[str, Tensor]) -> State:
        return OrderedDict(
            (name, value)
            for name, value in state.items()
            if name.startswith(self.shared_prefixes)
        )

    def personal_state(self, site: str, state: dict[str, Tensor]) -> State:
        personal = OrderedDict(
            (name, value)
            for name, value in state.items()
            if not name.startswith(self.shared_prefixes)
        )
        if site not in self.personal_states:
            self.personal_states[site] = OrderedDict(
                (name, value.detach().clone()) for name, value in personal.items()
            )
        return self.personal_states[site]

    def update_personal(self, site: str, state: dict[str, Tensor]) -> None:
        self.personal_states[site] = OrderedDict(
            (name, value.detach().clone())
            for name, value in state.items()
            if not name.startswith(self.shared_prefixes)
        )


@dataclass
class FedRepSchedule:
    representation_epochs: int
    head_epochs: int

    def phase(self, local_epoch: int) -> str:
        cycle = self.representation_epochs + self.head_epochs
        position = local_epoch % cycle
        return "representation" if position < self.representation_epochs else "head"

    def trainable_names(
        self,
        local_epoch: int,
        parameter_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        phase = self.phase(local_epoch)
        if phase == "representation":
            return tuple(name for name in parameter_names if not name.startswith("head."))
        return tuple(name for name in parameter_names if name.startswith("head."))


@dataclass
class PFedMeState:
    personalized: dict[str, State]
    regularization: float
    inner_steps: int

    def proximal_target(
        self,
        site: str,
        global_state: dict[str, Tensor],
    ) -> State:
        if site not in self.personalized:
            self.personalized[site] = OrderedDict(
                (name, value.detach().clone())
                for name, value in global_state.items()
            )
        return self.personalized[site]

    def update(
        self,
        site: str,
        local_state: dict[str, Tensor],
        global_state: dict[str, Tensor],
        mixing: float,
    ) -> None:
        target = self.proximal_target(site, global_state)
        for name, local in local_state.items():
            if local.is_floating_point():
                target[name] = (1.0 - mixing) * target[name] + mixing * local.detach()
            else:
                target[name] = local.detach().clone()


@dataclass
class MoonContrast:
    temperature: float
    strength: float

    def loss(
        self,
        current: Tensor,
        global_representation: Tensor,
        previous_representation: Tensor,
    ) -> Tensor:
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        current_normalized = torch.nn.functional.normalize(current, dim=-1)
        global_normalized = torch.nn.functional.normalize(global_representation, dim=-1)
        previous_normalized = torch.nn.functional.normalize(previous_representation, dim=-1)
        positive = (current_normalized * global_normalized).sum(dim=-1)
        negative = (current_normalized * previous_normalized).sum(dim=-1)
        logits = torch.stack([positive, negative], dim=1) / self.temperature
        targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return self.strength * torch.nn.functional.cross_entropy(logits, targets)


@dataclass
class FedBNPolicy:
    batch_norm_tokens: tuple[str, ...] = (
        "running_mean",
        "running_var",
        "num_batches_tracked",
    )

    def aggregateable(self, name: str) -> bool:
        return not any(token in name for token in self.batch_norm_tokens)

    def separate(
        self,
        state: dict[str, Tensor],
    ) -> tuple[State, State]:
        shared: State = OrderedDict()
        local: State = OrderedDict()
        for name, value in state.items():
            destination = shared if self.aggregateable(name) else local
            destination[name] = value
        return shared, local


def build_strategy(name: str, parameters: dict[str, float]) -> FederatedStrategy:
    normalized = name.lower()
    factories: dict[str, Callable[[], FederatedStrategy]] = {
        "fedavg": FedAvgStrategy,
        "uniform": UniformStrategy,
        "median": MedianStrategy,
        "fedprox": lambda: FedProxStrategy(parameters.get("mu", 0.01)),
        "qfedavg": lambda: QFedAvgStrategy(parameters.get("q", 1.0)),
        "momentum": lambda: MomentumStrategy(parameters.get("momentum", 0.9)),
        "fedadam": lambda: AdaptiveServerStrategy(
            parameters.get("lr", 0.01),
            parameters.get("beta1", 0.9),
            parameters.get("beta2", 0.999),
            parameters.get("epsilon", 1e-8),
        ),
    }
    if normalized not in factories:
        raise ValueError(f"unknown federated strategy: {name}")
    return factories[normalized]()


from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn

from fedicbs.federation.baselines import AggregationContext, FederatedStrategy
from fedicbs.science.invariance import InvarianceScreen
from fedicbs.science.privacy import GaussianMechanism, privatize_statistics
from fedicbs.types import ClientUpdate, InvarianceResult, RoundRecord, SiteStatistics


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClientDescriptor:
    site: str
    sample_count: int


@dataclass(frozen=True)
class FederationPlan:
    encoder_rounds: int
    prediction_rounds: int
    clients: tuple[ClientDescriptor, ...]
    seed: int

    def validate(self) -> None:
        if self.encoder_rounds < 0:
            raise ValueError("encoder rounds cannot be negative")
        if self.prediction_rounds < 1:
            raise ValueError("prediction rounds must be positive")
        if len(self.clients) < 5:
            raise ValueError("at least five clients are required")
        names = [client.site for client in self.clients]
        if len(names) != len(set(names)):
            raise ValueError("client names must be unique")
        if any(client.sample_count < 80 for client in self.clients):
            raise ValueError("each client requires at least 80 samples")


@dataclass
class FederationState:
    round_index: int
    global_state: OrderedDict[str, Tensor]
    previous_state: OrderedDict[str, Tensor] | None
    history: list[RoundRecord]
    invariant_results: tuple[InvarianceResult, ...] | None


class FedICBSCoordinator:
    def __init__(
        self,
        model: nn.Module,
        plan: FederationPlan,
        strategy: FederatedStrategy,
        screen: InvarianceScreen,
        privacy: GaussianMechanism | None = None,
    ) -> None:
        plan.validate()
        self.model = model
        self.plan = plan
        self.strategy = strategy
        self.screen = screen
        self.privacy = privacy
        self.state = FederationState(
            round_index=0,
            global_state=_copy_state(model.state_dict()),
            previous_state=None,
            history=[],
            invariant_results=None,
        )

    def encoder_phase(
        self,
        train_client: Callable[
            [ClientDescriptor, dict[str, Tensor], int],
            ClientUpdate,
        ],
        validate: Callable[[dict[str, Tensor]], tuple[float, float]],
    ) -> None:
        for round_index in range(1, self.plan.encoder_rounds + 1):
            updates = [
                train_client(client, self.state.global_state, round_index)
                for client in self.plan.clients
            ]
            self._aggregate_round(updates, round_index)
            validation_auc, training_loss = validate(self.state.global_state)
            gradient_norm = float(np.mean([update.gradient_norm for update in updates]))
            self.state.history.append(
                RoundRecord(
                    round_index=round_index,
                    training_loss=training_loss,
                    validation_auc=validation_auc,
                    gradient_norm=gradient_norm,
                    invariant_count=0,
                    mean_q=float("nan"),
                )
            )

    def invariance_phase(
        self,
        collect_statistics: Callable[[ClientDescriptor], SiteStatistics],
    ) -> tuple[int, ...]:
        statistics: list[SiteStatistics] = []
        for client_index, client in enumerate(self.plan.clients):
            result = collect_statistics(client)
            result.validate()
            if self.privacy is not None:
                result = privatize_statistics(
                    result,
                    self.privacy,
                    self.plan.seed + client_index,
                )
            statistics.append(result)
        results = self.screen.evaluate_all(statistics)
        self.state.invariant_results = results
        selected = tuple(
            result.feature_index
            for result in results
            if result.decision.value == "causal"
        )
        if not selected:
            raise RuntimeError("invariance screen selected no features")
        return selected

    def prediction_phase(
        self,
        train_client: Callable[
            [ClientDescriptor, dict[str, Tensor], int],
            ClientUpdate,
        ],
        validate: Callable[[dict[str, Tensor]], tuple[float, float]],
    ) -> None:
        start = self.plan.encoder_rounds + 1
        stop = self.plan.encoder_rounds + self.plan.prediction_rounds + 1
        invariant_count = self._invariant_count()
        mean_q = self._mean_q()
        for round_index in range(start, stop):
            updates = [
                train_client(client, self.state.global_state, round_index)
                for client in self.plan.clients
            ]
            self._aggregate_round(updates, round_index)
            validation_auc, training_loss = validate(self.state.global_state)
            gradient_norm = float(np.mean([update.gradient_norm for update in updates]))
            self.state.history.append(
                RoundRecord(
                    round_index=round_index,
                    training_loss=training_loss,
                    validation_auc=validation_auc,
                    gradient_norm=gradient_norm,
                    invariant_count=invariant_count,
                    mean_q=mean_q,
                )
            )

    def _aggregate_round(
        self,
        updates: list[ClientUpdate],
        round_index: int,
    ) -> None:
        expected = {client.site for client in self.plan.clients}
        received = {update.site for update in updates}
        if received != expected:
            missing = sorted(expected.difference(received))
            extra = sorted(received.difference(expected))
            raise RuntimeError(f"client update mismatch; missing={missing}, extra={extra}")
        context = AggregationContext(
            round_index=round_index,
            total_rounds=self.plan.encoder_rounds + self.plan.prediction_rounds,
            server_state=self.state.global_state,
            previous_server_state=self.state.previous_state,
        )
        aggregated = self.strategy.aggregate(updates, context)
        self.state.previous_state = _copy_state(self.state.global_state)
        self.state.global_state = _copy_state(aggregated)
        self.state.round_index = round_index
        self.model.load_state_dict(self.state.global_state, strict=True)

    def _invariant_count(self) -> int:
        if self.state.invariant_results is None:
            return 0
        return sum(
            result.decision.value == "causal"
            for result in self.state.invariant_results
        )

    def _mean_q(self) -> float:
        if self.state.invariant_results is None:
            return float("nan")
        values = [
            result.q_statistic
            for result in self.state.invariant_results
            if np.isfinite(result.q_statistic)
        ]
        return float(np.mean(values)) if values else float("nan")

    def history_array(self) -> np.ndarray:
        return np.asarray(
            [
                [
                    record.round_index,
                    record.training_loss,
                    record.validation_auc,
                    record.gradient_norm,
                    record.invariant_count,
                    record.mean_q,
                ]
                for record in self.state.history
            ],
            dtype=np.float64,
        )


def _copy_state(state: dict[str, Tensor]) -> OrderedDict[str, Tensor]:
    return OrderedDict(
        (name, value.detach().clone())
        for name, value in state.items()
    )


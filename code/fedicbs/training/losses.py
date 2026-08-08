from __future__ import annotations

import torch
from torch import Tensor, nn


class BinaryFocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        positive_weight: float | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.positive_weight = positive_weight
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        labels = targets.to(dtype=logits.dtype)
        base = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
        )
        probabilities = torch.sigmoid(logits)
        correct_probability = probabilities * labels + (1.0 - probabilities) * (1.0 - labels)
        modulation = torch.pow(1.0 - correct_probability, self.gamma)
        loss = modulation * base
        if self.positive_weight is not None:
            weights = labels * self.positive_weight + (1.0 - labels)
            loss = loss * weights
        return _reduce(loss, self.reduction)


class BalancedBinaryLoss(nn.Module):
    def __init__(self, positive_fraction: float) -> None:
        super().__init__()
        if not 0.0 < positive_fraction < 1.0:
            raise ValueError("positive_fraction must lie between zero and one")
        self.positive_fraction = positive_fraction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        labels = targets.to(dtype=logits.dtype)
        positive_weight = (1.0 - self.positive_fraction) / self.positive_fraction
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=torch.as_tensor(
                positive_weight,
                dtype=logits.dtype,
                device=logits.device,
            ),
        )


class InvariancePenalty(nn.Module):
    def __init__(self, strength: float) -> None:
        super().__init__()
        if strength < 0.0:
            raise ValueError("strength cannot be negative")
        self.strength = strength

    def forward(self, site_losses: Tensor) -> Tensor:
        if site_losses.ndim != 1:
            raise ValueError("site losses must have rank one")
        if site_losses.numel() < 2:
            return site_losses.new_zeros(())
        return self.strength * site_losses.var(unbiased=True)


class GroupDroLoss(nn.Module):
    def __init__(self, groups: int, step_size: float = 0.01) -> None:
        super().__init__()
        if groups < 2:
            raise ValueError("at least two groups are required")
        self.step_size = step_size
        self.register_buffer("weights", torch.full((groups,), 1.0 / groups))

    def forward(self, group_losses: Tensor) -> Tensor:
        if group_losses.shape != self.weights.shape:
            raise ValueError("group loss shape mismatch")
        with torch.no_grad():
            updated = self.weights * torch.exp(self.step_size * group_losses.detach())
            self.weights.copy_(updated / updated.sum())
        return torch.dot(self.weights, group_losses)


class ContrastiveAlignmentLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, first: Tensor, second: Tensor) -> Tensor:
        if first.shape != second.shape or first.ndim != 2:
            raise ValueError("contrastive inputs must have matching rank-two shapes")
        first_normalized = torch.nn.functional.normalize(first, dim=1)
        second_normalized = torch.nn.functional.normalize(second, dim=1)
        logits = first_normalized @ second_normalized.T / self.temperature
        targets = torch.arange(first.shape[0], device=first.device)
        forward = torch.nn.functional.cross_entropy(logits, targets)
        backward = torch.nn.functional.cross_entropy(logits.T, targets)
        return 0.5 * (forward + backward)


class CovariancePenalty(nn.Module):
    def __init__(self, strength: float = 1.0) -> None:
        super().__init__()
        self.strength = strength

    def forward(self, representation: Tensor) -> Tensor:
        if representation.ndim != 2:
            raise ValueError("representation must have rank two")
        centered = representation - representation.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / max(representation.shape[0] - 1, 1)
        off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
        return self.strength * torch.square(off_diagonal).sum() / representation.shape[1]


def _reduce(loss: Tensor, reduction: str) -> Tensor:
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"unsupported reduction: {reduction}")


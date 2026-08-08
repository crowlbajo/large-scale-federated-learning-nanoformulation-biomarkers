from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn


class TensorEncoder(Protocol):
    output_dimension: int

    def __call__(self, inputs: Tensor) -> Tensor:
        ...


@dataclass(frozen=True)
class EncoderDimensions:
    image_input: int = 768
    image_output: int = 128
    text_input: int = 768
    text_output: int = 128
    molecule_input: int = 256
    molecule_output: int = 128
    tabular_input: int = 43
    tabular_output: int = 64


class ProjectionEncoder(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.output_dimension = output_dimension
        self.network = nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, output_dimension),
            nn.LayerNorm(output_dimension),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 2:
            raise ValueError("projection inputs must have rank two")
        return self.network(inputs)


class FourLayerTabularEncoder(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int = 64,
        hidden_dimension: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dimensions = [
            input_dimension,
            hidden_dimension,
            hidden_dimension,
            hidden_dimension // 2,
            output_dimension,
        ]
        layers: list[nn.Module] = []
        for index, (source, target) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(source, target))
            if index < 3:
                layers.append(nn.LayerNorm(target))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)
        self.output_dimension = output_dimension

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 2:
            raise ValueError("tabular inputs must have rank two")
        return self.network(inputs)


class FrozenBackbone(nn.Module):
    def __init__(self, backbone: nn.Module, output_dimension: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.output_dimension = output_dimension
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> FrozenBackbone:
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, inputs: Tensor) -> Tensor:
        with torch.no_grad():
            output = self.backbone(inputs)
        if isinstance(output, Tensor):
            return output
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state.mean(dim=1)
        raise TypeError("backbone output cannot be converted to a tensor")


class MeanTokenPooler(nn.Module):
    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must have rank three")
        if mask is None:
            return tokens.mean(dim=1)
        if mask.shape != tokens.shape[:2]:
            raise ValueError("mask shape mismatch")
        weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return (tokens * weights).sum(dim=1) / denominator


class MolecularFingerprintPooler(nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int = 256) -> None:
        super().__init__()
        self.attention = nn.Linear(input_dimension, 1)
        self.projection = nn.Linear(input_dimension, output_dimension)
        self.output_dimension = output_dimension

    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError("molecular tokens must have rank three")
        scores = self.attention(tokens).squeeze(-1)
        if mask is not None:
            if mask.shape != scores.shape:
                raise ValueError("molecular mask shape mismatch")
            scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.einsum("bs,bsd->bd", weights, tokens)
        return self.projection(pooled)


class ModalityPresenceEmbedding(nn.Module):
    def __init__(self, modalities: int, dimension: int) -> None:
        super().__init__()
        self.present = nn.Parameter(torch.empty(modalities, dimension))
        self.absent = nn.Parameter(torch.empty(modalities, dimension))
        nn.init.normal_(self.present, mean=0.0, std=0.02)
        nn.init.normal_(self.absent, mean=0.0, std=0.02)

    def forward(self, streams: Tensor, presence: Tensor) -> Tensor:
        if streams.ndim != 3:
            raise ValueError("streams must have rank three")
        if presence.shape != streams.shape[:2]:
            raise ValueError("presence shape mismatch")
        if streams.shape[1] != self.present.shape[0]:
            raise ValueError("modality count mismatch")
        present = self.present.unsqueeze(0)
        absent = self.absent.unsqueeze(0)
        embedding = torch.where(presence.bool().unsqueeze(-1), present, absent)
        return streams + embedding


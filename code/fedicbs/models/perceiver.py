from __future__ import annotations

import torch
from torch import Tensor, nn


class FeedForward(nn.Module):
    def __init__(self, dimension: int, multiplier: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = dimension * multiplier
        self.network = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


class CrossAttention(nn.Module):
    def __init__(self, dimension: int, heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if dimension % heads != 0:
            raise ValueError("dimension must be divisible by heads")
        self.query_norm = nn.LayerNorm(dimension)
        self.context_norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension,
            heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        query: Tensor,
        context: Tensor,
        context_padding_mask: Tensor | None = None,
    ) -> Tensor:
        normalized_query = self.query_norm(query)
        normalized_context = self.context_norm(context)
        output, _ = self.attention(
            normalized_query,
            normalized_context,
            normalized_context,
            key_padding_mask=context_padding_mask,
            need_weights=False,
        )
        return output


class SelfAttention(nn.Module):
    def __init__(self, dimension: int, heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if dimension % heads != 0:
            raise ValueError("dimension must be divisible by heads")
        self.normalization = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension,
            heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        normalized = self.normalization(inputs)
        output, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        return output


class PerceiverLayer(nn.Module):
    def __init__(self, dimension: int, heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attention = CrossAttention(dimension, heads, dropout)
        self.cross_feed_forward = FeedForward(dimension, dropout=dropout)
        self.self_attention = SelfAttention(dimension, heads, dropout)
        self.self_feed_forward = FeedForward(dimension, dropout=dropout)

    def forward(
        self,
        latents: Tensor,
        context: Tensor,
        context_padding_mask: Tensor | None = None,
    ) -> Tensor:
        latents = latents + self.cross_attention(latents, context, context_padding_mask)
        latents = latents + self.cross_feed_forward(latents)
        latents = latents + self.self_attention(latents)
        latents = latents + self.self_feed_forward(latents)
        return latents


class PerceiverFusion(nn.Module):
    def __init__(
        self,
        dimension: int = 128,
        heads: int = 8,
        layers: int = 4,
        latent_count: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.latents = nn.Parameter(torch.empty(latent_count, dimension))
        nn.init.normal_(self.latents, mean=0.0, std=0.02)
        self.layers = nn.ModuleList(
            [PerceiverLayer(dimension, heads, dropout) for _ in range(layers)]
        )
        self.output_normalization = nn.LayerNorm(dimension)
        self.output_attention = nn.Linear(dimension, 1)

    def forward(
        self,
        context: Tensor,
        context_padding_mask: Tensor | None = None,
    ) -> Tensor:
        if context.ndim != 3:
            raise ValueError("fusion context must have rank three")
        if context.shape[-1] != self.dimension:
            raise ValueError("fusion context dimension mismatch")
        batch_size = context.shape[0]
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        for layer in self.layers:
            latents = layer(latents, context, context_padding_mask)
        normalized = self.output_normalization(latents)
        weights = torch.softmax(self.output_attention(normalized).squeeze(-1), dim=1)
        return torch.einsum("bl,bld->bd", weights, normalized)


class ModalityProjector(nn.Module):
    def __init__(self, input_dimension: int, shared_dimension: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, shared_dimension),
            nn.GELU(),
            nn.Linear(shared_dimension, shared_dimension),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.projection(inputs)


class MultiStreamPerceiver(nn.Module):
    def __init__(
        self,
        input_dimensions: tuple[int, ...],
        shared_dimension: int = 128,
        heads: int = 8,
        layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.projectors = nn.ModuleList(
            [ModalityProjector(value, shared_dimension) for value in input_dimensions]
        )
        self.missing_tokens = nn.Parameter(torch.empty(len(input_dimensions), shared_dimension))
        nn.init.normal_(self.missing_tokens, mean=0.0, std=0.02)
        self.fusion = PerceiverFusion(
            dimension=shared_dimension,
            heads=heads,
            layers=layers,
            dropout=dropout,
        )

    def forward(
        self,
        streams: tuple[Tensor, ...],
        presence: Tensor,
    ) -> Tensor:
        if len(streams) != len(self.projectors):
            raise ValueError("stream count mismatch")
        if presence.ndim != 2 or presence.shape[1] != len(streams):
            raise ValueError("presence shape mismatch")
        projected: list[Tensor] = []
        for index, (stream, projector) in enumerate(zip(streams, self.projectors)):
            value = projector(stream)
            missing = self.missing_tokens[index].view(1, -1).expand_as(value)
            selected = torch.where(presence[:, index : index + 1].bool(), value, missing)
            projected.append(selected)
        context = torch.stack(projected, dim=1)
        return self.fusion(context)


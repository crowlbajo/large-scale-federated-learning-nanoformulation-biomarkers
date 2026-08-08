from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from fedicbs.models.encoders import FourLayerTabularEncoder, ProjectionEncoder
from fedicbs.models.perceiver import MultiStreamPerceiver


@dataclass(frozen=True)
class MultiModalInputs:
    image_embedding: Tensor
    text_embedding: Tensor
    molecular_embedding: Tensor
    tabular_features: Tensor
    modality_presence: Tensor
    drug_properties: Tensor
    interaction_covariates: Tensor


class InteractionBuilder(nn.Module):
    def __init__(
        self,
        selected_drug_properties: int = 5,
        patient_covariates: int = 14,
    ) -> None:
        super().__init__()
        self.selected_drug_properties = selected_drug_properties
        self.patient_covariates = patient_covariates
        self.output_dimension = selected_drug_properties * patient_covariates

    def forward(self, drug: Tensor, patient: Tensor) -> Tensor:
        if drug.ndim != 2 or patient.ndim != 2:
            raise ValueError("interaction inputs must have rank two")
        if drug.shape[1] < self.selected_drug_properties:
            raise ValueError("insufficient drug properties")
        if patient.shape[1] != self.patient_covariates:
            raise ValueError("patient interaction dimension mismatch")
        selected = drug[:, : self.selected_drug_properties]
        products = patient.unsqueeze(2) * selected.unsqueeze(1)
        return products.reshape(patient.shape[0], -1)


class InvariantPredictionHead(nn.Module):
    def __init__(self, input_dimension: int = 21, hidden_dimension: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dimension, hidden_dimension // 2),
            nn.GELU(),
            nn.Linear(hidden_dimension // 2, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2:
            raise ValueError("prediction features must have rank two")
        return self.network(features).squeeze(-1)


class FeatureMask(nn.Module):
    def __init__(self, candidate_dimension: int = 128) -> None:
        super().__init__()
        self.candidate_dimension = candidate_dimension
        self.register_buffer(
            "indices",
            torch.arange(candidate_dimension, dtype=torch.long),
            persistent=True,
        )
        self.selected = False

    def select(self, indices: Tensor) -> None:
        if indices.ndim != 1:
            raise ValueError("feature indices must have rank one")
        if indices.numel() < 1:
            raise ValueError("at least one feature must be selected")
        if int(indices.min()) < 0 or int(indices.max()) >= self.candidate_dimension:
            raise ValueError("feature index out of range")
        self.indices = torch.unique(indices.to(dtype=torch.long), sorted=True)
        self.selected = True

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[-1] != self.candidate_dimension:
            raise ValueError("candidate feature dimension mismatch")
        if not self.selected:
            raise RuntimeError("invariant feature mask has not been selected")
        return features.index_select(-1, self.indices)


class CandidateFeatureAssembler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patient_projection = nn.Linear(128, 43)
        self.interactions = InteractionBuilder(5, 14)

    def forward(
        self,
        patient_representation: Tensor,
        drug_properties: Tensor,
        interaction_covariates: Tensor,
    ) -> Tensor:
        if drug_properties.shape[1] != 15:
            raise ValueError("drug property dimension must be 15")
        patient = self.patient_projection(patient_representation)
        interactions = self.interactions(drug_properties, interaction_covariates)
        result = torch.cat([patient, drug_properties, interactions], dim=1)
        if result.shape[1] != 128:
            raise RuntimeError("candidate feature assembly must produce 128 dimensions")
        return result


class FedICBSModel(nn.Module):
    def __init__(self, invariant_dimension: int = 21) -> None:
        super().__init__()
        self.image_projection = ProjectionEncoder(768, 512, 128)
        self.text_projection = ProjectionEncoder(768, 512, 128)
        self.molecule_projection = ProjectionEncoder(256, 512, 128)
        self.tabular_encoder = FourLayerTabularEncoder(43, 64, 256)
        self.fusion = MultiStreamPerceiver(
            input_dimensions=(128, 128, 128, 64),
            shared_dimension=128,
            heads=8,
            layers=4,
        )
        self.assembler = CandidateFeatureAssembler()
        self.mask = FeatureMask(128)
        self.head = InvariantPredictionHead(invariant_dimension, 128)

    def set_invariant_features(self, indices: Tensor) -> None:
        if indices.numel() != self.head.network[0].normalized_shape[0]:
            raise ValueError("selected dimension differs from prediction head")
        self.mask.select(indices)

    def encode(self, inputs: MultiModalInputs) -> Tensor:
        image = self.image_projection(inputs.image_embedding)
        text = self.text_projection(inputs.text_embedding)
        molecule = self.molecule_projection(inputs.molecular_embedding)
        tabular = self.tabular_encoder(inputs.tabular_features)
        return self.fusion(
            (image, text, molecule, tabular),
            inputs.modality_presence,
        )

    def candidate_features(self, inputs: MultiModalInputs) -> Tensor:
        patient = self.encode(inputs)
        return self.assembler(
            patient,
            inputs.drug_properties,
            inputs.interaction_covariates,
        )

    def forward(self, inputs: MultiModalInputs) -> Tensor:
        candidates = self.candidate_features(inputs)
        invariant = self.mask(candidates)
        return self.head(invariant)

    def probabilities(self, inputs: MultiModalInputs) -> Tensor:
        return torch.sigmoid(self.forward(inputs))

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


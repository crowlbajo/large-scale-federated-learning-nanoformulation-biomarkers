from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from scipy.stats import linregress


@dataclass(frozen=True)
class ReleaseProfile:
    times: NDArray[np.float64]
    released_fraction: NDArray[np.float64]

    def validate(self) -> None:
        if self.times.ndim != 1:
            raise ValueError("times must have rank one")
        if self.released_fraction.shape != self.times.shape:
            raise ValueError("release fraction shape mismatch")
        if self.times.size < 3:
            raise ValueError("at least three release observations are required")
        if np.any(self.times <= 0.0):
            raise ValueError("release times must be positive")
        if np.any(self.released_fraction <= 0.0):
            raise ValueError("released fractions must be positive")


@dataclass(frozen=True)
class FormulationProperties:
    encapsulation_efficiency: float
    particle_size_nm: float
    zeta_potential_mv: float
    drug_loading_fraction: float
    normalized_release_rate: float

    def validate(self) -> None:
        if not 0.0 <= self.encapsulation_efficiency <= 100.0:
            raise ValueError("encapsulation efficiency must be a percentage")
        if self.particle_size_nm <= 0.0:
            raise ValueError("particle size must be positive")
        if not 0.0 <= self.drug_loading_fraction <= 1.0:
            raise ValueError("drug loading fraction must lie between zero and one")
        if self.normalized_release_rate < 0.0:
            raise ValueError("release rate cannot be negative")


@dataclass(frozen=True)
class MolecularDescriptors:
    molecular_weight: float
    log_p: float
    tpsa: float
    hydrogen_bond_acceptors: float
    hydrogen_bond_donors: float
    rotatable_bonds: float
    ring_count: float
    heavy_atoms: float

    def as_array(self) -> NDArray[np.float64]:
        return np.asarray(
            [
                self.molecular_weight,
                self.log_p,
                self.tpsa,
                self.hydrogen_bond_acceptors,
                self.hydrogen_bond_donors,
                self.rotatable_bonds,
                self.ring_count,
                self.heavy_atoms,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class NanoformulationVector:
    descriptors: MolecularDescriptors
    higuchi_rate: float
    korsmeyer_peppas_exponent: float
    encapsulation_efficiency: float
    particle_size_nm: float
    zeta_potential_mv: float
    drug_loading_fraction: float
    normalized_release_rate: float
    size_charge_ratio: float

    def as_array(self) -> NDArray[np.float64]:
        return np.concatenate(
            [
                self.descriptors.as_array(),
                np.asarray(
                    [
                        self.higuchi_rate,
                        self.korsmeyer_peppas_exponent,
                        self.encapsulation_efficiency,
                        self.particle_size_nm,
                        self.zeta_potential_mv,
                        self.drug_loading_fraction,
                        self.normalized_release_rate,
                        self.size_charge_ratio,
                    ],
                    dtype=np.float64,
                ),
            ]
        )


def compute_molecular_descriptors(smiles: str) -> MolecularDescriptors:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("invalid canonical SMILES")
    return MolecularDescriptors(
        molecular_weight=float(Descriptors.MolWt(molecule)),
        log_p=float(Crippen.MolLogP(molecule)),
        tpsa=float(rdMolDescriptors.CalcTPSA(molecule)),
        hydrogen_bond_acceptors=float(Lipinski.NumHAcceptors(molecule)),
        hydrogen_bond_donors=float(Lipinski.NumHDonors(molecule)),
        rotatable_bonds=float(Lipinski.NumRotatableBonds(molecule)),
        ring_count=float(Lipinski.RingCount(molecule)),
        heavy_atoms=float(molecule.GetNumHeavyAtoms()),
    )


def canonicalize_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("invalid SMILES")
    return str(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))


def higuchi_release_rate(profile: ReleaseProfile) -> float:
    profile.validate()
    square_root_time = np.sqrt(profile.times)
    fit = linregress(square_root_time, profile.released_fraction)
    return float(fit.slope)


def korsmeyer_peppas(profile: ReleaseProfile, maximum_fraction: float = 0.6) -> tuple[float, float]:
    profile.validate()
    mask = profile.released_fraction <= maximum_fraction
    if int(mask.sum()) < 3:
        mask = np.ones(profile.times.shape, dtype=np.bool_)
    logarithmic_time = np.log(profile.times[mask])
    logarithmic_release = np.log(profile.released_fraction[mask])
    fit = linregress(logarithmic_time, logarithmic_release)
    exponent = float(fit.slope)
    rate_constant = float(np.exp(fit.intercept))
    return exponent, rate_constant


def build_nanoformulation_vector(
    smiles: str,
    profile: ReleaseProfile,
    properties: FormulationProperties,
) -> NanoformulationVector:
    properties.validate()
    descriptors = compute_molecular_descriptors(canonicalize_smiles(smiles))
    higuchi = higuchi_release_rate(profile)
    exponent, _ = korsmeyer_peppas(profile)
    denominator = abs(properties.zeta_potential_mv)
    ratio = properties.particle_size_nm / max(denominator, 1e-8)
    return NanoformulationVector(
        descriptors=descriptors,
        higuchi_rate=higuchi,
        korsmeyer_peppas_exponent=exponent,
        encapsulation_efficiency=properties.encapsulation_efficiency,
        particle_size_nm=properties.particle_size_nm,
        zeta_potential_mv=properties.zeta_potential_mv,
        drug_loading_fraction=properties.drug_loading_fraction,
        normalized_release_rate=properties.normalized_release_rate,
        size_charge_ratio=ratio,
    )


def standardize_vectors(
    vectors: list[NanoformulationVector],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if not vectors:
        raise ValueError("at least one vector is required")
    matrix = np.stack([vector.as_array() for vector in vectors], axis=0)
    means = matrix.mean(axis=0)
    standard_deviations = matrix.std(axis=0)
    standard_deviations = np.where(standard_deviations < 1e-12, 1.0, standard_deviations)
    return (matrix - means) / standard_deviations, means, standard_deviations


def interpolate_release(
    profile: ReleaseProfile,
    query_times: NDArray[np.float64],
) -> NDArray[np.float64]:
    profile.validate()
    query = np.asarray(query_times, dtype=np.float64)
    if np.any(query <= 0.0):
        raise ValueError("query times must be positive")
    order = np.argsort(profile.times)
    return np.interp(
        query,
        profile.times[order],
        profile.released_fraction[order],
        left=profile.released_fraction[order][0],
        right=profile.released_fraction[order][-1],
    )


def release_half_time(profile: ReleaseProfile) -> float:
    profile.validate()
    order = np.argsort(profile.released_fraction)
    return float(
        np.interp(
            0.5,
            profile.released_fraction[order],
            profile.times[order],
        )
    )


def dissolution_efficiency(profile: ReleaseProfile) -> float:
    profile.validate()
    order = np.argsort(profile.times)
    times = profile.times[order]
    released = profile.released_fraction[order]
    area = float(np.trapezoid(released, times))
    maximum_area = float(times[-1] * released.max())
    return area / maximum_area if maximum_area > 0.0 else 0.0


def similarity_factor(
    reference: ReleaseProfile,
    candidate: ReleaseProfile,
) -> float:
    reference.validate()
    candidate.validate()
    shared_times = np.union1d(reference.times, candidate.times)
    reference_values = interpolate_release(reference, shared_times)
    candidate_values = interpolate_release(candidate, shared_times)
    mean_square = float(np.mean(np.square(reference_values - candidate_values)))
    return 50.0 * log(100.0 / sqrt(1.0 + mean_square), 10)

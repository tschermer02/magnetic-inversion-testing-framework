"""Configuration for the E02 log-uniform susceptibility dataset."""
from __future__ import annotations

from dataclasses import dataclass

from e01_magnetic.config import MagneticSurveyConfig


@dataclass(frozen=True)
class E02DatasetConfig:
    """E02 changes only susceptibility sampling relative to E01."""

    experiment_name: str = "E02"
    dataset_version: str = "E02_loguniform_susceptibility_v1"
    random_seed: int = 20260727
    log10_susceptibility_min: float = -4.0
    log10_susceptibility_max: float = -1.0
    susceptibility_distribution: str = "log_uniform_base10"
    tmi_normalization_nt: float = 100.0
    homogeneous_body_susceptibility: bool = True
    remanent_magnetization: bool = False
    observational_noise: bool = False
    survey: MagneticSurveyConfig = MagneticSurveyConfig()

    @property
    def susceptibility_min_si(self) -> float:
        return 10.0 ** self.log10_susceptibility_min

    @property
    def susceptibility_max_si(self) -> float:
        return 10.0 ** self.log10_susceptibility_max

    def validate(self) -> None:
        self.survey.validate()
        if self.log10_susceptibility_max <= self.log10_susceptibility_min:
            raise ValueError("The susceptibility exponent maximum must exceed the minimum.")
        if self.tmi_normalization_nt <= 0.0:
            raise ValueError("TMI normalization must be positive.")


E02_DEFAULT_CONFIG = E02DatasetConfig()

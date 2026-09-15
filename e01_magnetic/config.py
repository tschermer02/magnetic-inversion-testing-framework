from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MagneticSurveyConfig:
    """Fixed-field magnetic survey used by the controlled E01 experiment."""

    field_strength_nt: float = 50_000.0
    inclination_deg: float = 75.0
    declination_deg: float = 25.0
    azimuth_deg: float = 0.0
    receiver_z_elevation_m: float = -10.0
    receiver_dx_m: float = 10.0
    receiver_dy_m: float = 10.0
    receiver_x_min_m: float = 0.0
    receiver_x_max_m: float = 800.0
    receiver_y_min_m: float = 0.0
    receiver_y_max_m: float = 800.0
    receiver_map_shape: tuple[int, int] = (81, 81)
    model_shape: tuple[int, int, int] = (24, 64, 64)

    def validate(self) -> None:
        if not np.isfinite(self.field_strength_nt) or self.field_strength_nt <= 0.0:
            raise ValueError("field_strength_nt must be positive and finite.")
        if not 0.0 <= self.inclination_deg <= 90.0:
            raise ValueError("inclination_deg must be in [0, 90].")
        if not 0.0 <= self.azimuth_deg < 360.0:
            raise ValueError("azimuth_deg must be in [0, 360).")
        if self.receiver_dx_m <= 0.0 or self.receiver_dy_m <= 0.0:
            raise ValueError("receiver grid spacing must be positive.")

    @property
    def receiver_xyz(self) -> np.ndarray:
        """Construct a 2-D surface receiver map in the MATLAB-compatible coordinate convention."""
        nx, ny = self.receiver_map_shape
        x = np.linspace(self.receiver_x_min_m, self.receiver_x_max_m, nx, dtype=np.float64)
        y = np.linspace(self.receiver_y_min_m, self.receiver_y_max_m, ny, dtype=np.float64)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        zz = np.full_like(xx, self.receiver_z_elevation_m, dtype=np.float64)
        receivers = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
        return receivers


@dataclass(frozen=True)
class E01Config:
    """Top-level configuration for the magnetic analogue of the gravity E09B-12 experiment."""

    experiment_name: str = "E01"
    seed: int = 20260727
    output_directory: Path = Path("outputs/E01")
    dataset_directory: Path = Path("datasets/E01")
    training_samples: int = 10_000
    validation_samples: int = 1_000
    test_samples: int = 100
    batch_size: int = 2
    epochs: int = 3
    base_filters: int = 8
    learning_rate: float = 1.0e-3
    survey: MagneticSurveyConfig = MagneticSurveyConfig()
    model_shape: tuple[int, int, int] = (24, 64, 64)
    input_shape: tuple[int, int, int] = (81, 81, 1)
    sensitivity_gamma: float = 0.5
    sensitivity_weight_min: float = 0.5
    sensitivity_weight_max: float = 5.0
    lambda_susceptibility: float = 1.0
    lambda_depth: float = 1.0
    lambda_tmi_forward: float = 0.001
    lambda_tversky: float = 0.1

    def validate(self) -> None:
        self.survey.validate()
        if self.base_filters < 1:
            raise ValueError("base_filters must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.training_samples < 1 or self.validation_samples < 1 or self.test_samples < 1:
            raise ValueError("All dataset split sizes must be positive.")


E01_DEFAULT_CONFIG = E01Config()

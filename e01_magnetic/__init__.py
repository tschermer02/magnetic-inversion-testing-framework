"""E01 magnetic inversion framework."""

from .config import E01Config, MagneticSurveyConfig
from .dataset import build_diagnostic_dataset, generate_diagnostic_dataset
from .losses import TMIPhysicsLoss
from .model import build_e01_model, count_trainable_parameters
from .physics import MagneticTMIModel, build_sensitivity_matrix

__all__ = [
    "E01Config",
    "MagneticSurveyConfig",
    "MagneticTMIModel",
    "TMIPhysicsLoss",
    "build_e01_model",
    "build_sensitivity_matrix",
    "build_diagnostic_dataset",
    "generate_diagnostic_dataset",
    "count_trainable_parameters",
]

"""Single-architecture E01 magnetic inversion package."""

from cnn_inversion_3d.e01_training import E01LossConfig, E01TrainingModel
from cnn_inversion_3d.model import ModelConfig, build_e01_model

__all__ = ["E01LossConfig", "E01TrainingModel", "ModelConfig", "build_e01_model"]

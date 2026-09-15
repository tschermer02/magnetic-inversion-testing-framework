"""Canonical CNN interface for TMI-to-susceptibility inversion."""
from __future__ import annotations
from dataclasses import dataclass
import tensorflow as tf
from cnn_inversion_3d.model import ModelConfig, build_e01_model

@dataclass(frozen=True)
class MagneticModelConfig:
    input_shape: tuple[int, int, int] = (81, 81, 1)
    output_shape: tuple[int, int, int, int] = (24, 64, 64, 1)
    base_filters: int = 8
    learning_rate: float = 1.0e-3

def build_tmi_inversion_model(config: MagneticModelConfig | None = None) -> tf.keras.Model:
    """Build the established single-plane CNN with magnetic semantics."""
    cfg = config or MagneticModelConfig()
    if cfg.input_shape != (81, 81, 1) or cfg.output_shape != (24, 64, 64, 1):
        raise ValueError("The canonical magnetic geometry is TMI (81,81,1) to susceptibility (24,64,64,1).")
    model = build_e01_model(ModelConfig(base_filters=cfg.base_filters, learning_rate=cfg.learning_rate))
    model._name = "tmi_to_susceptibility_cnn"
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.learning_rate), loss="mse")
    return model

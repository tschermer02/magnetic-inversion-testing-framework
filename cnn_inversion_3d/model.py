"""E01 TMI-to-susceptibility asymmetric 2-D U-Net."""
from __future__ import annotations
from dataclasses import dataclass
import tensorflow as tf
from cnn_inversion_3d.dataset import TMI_SHAPE, SUSCEPTIBILITY_SHAPE

@dataclass(frozen=True)
class ModelConfig:
    base_filters: int = 8
    learning_rate: float = 1.0e-3
    output_activation: str = "sigmoid"
    maximum_susceptibility_si: float = 0.1
    body_loss_fraction: float = 0.5
    def validate(self) -> None:
        if self.base_filters < 1: raise ValueError("base_filters must be positive.")
        if self.learning_rate <= 0: raise ValueError("learning_rate must be positive.")
        if self.output_activation not in {"sigmoid", "relu", "linear"}: raise ValueError("Invalid output activation.")
        if self.maximum_susceptibility_si <= 0: raise ValueError("maximum_susceptibility_si must be positive.")
        if not 0 < self.body_loss_fraction < 1: raise ValueError("body_loss_fraction must be in (0,1).")

def _block(inputs: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = inputs
    for index in (1, 2):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu",
            kernel_initializer="he_normal", name=f"{name}_conv_{index}")(x)
    return x

def build_e01_model(config: ModelConfig | None = None) -> tf.keras.Model:
    """Build the retained asymmetric 2-D U-Net as the E01 architecture."""
    cfg = config or ModelConfig(); cfg.validate(); filters = cfg.base_filters
    inputs = tf.keras.Input(shape=TMI_SHAPE, name="e01_tmi_nt")
    padded = tf.keras.layers.ZeroPadding2D(((7, 8), (7, 8)), name="e01_pad_81_to_96")(inputs)
    enc1 = _block(padded, filters, "e01_encoder_1")
    enc2 = _block(tf.keras.layers.MaxPool2D(2)(enc1), filters*2, "e01_encoder_2")
    enc3 = _block(tf.keras.layers.MaxPool2D(2)(enc2), filters*4, "e01_encoder_3")
    decoded = _block(tf.keras.layers.MaxPool2D(2)(enc3), filters*8, "e01_bottleneck")
    for index, (skip, count, size) in enumerate(((enc3,filters*4,24),(enc2,filters*2,48),(enc1,filters,96)), 1):
        decoded = tf.keras.layers.Conv2DTranspose(count, 2, strides=2, padding="same", activation="relu",
            kernel_initializer="he_normal", name=f"e01_upsample_to_{size}")(decoded)
        decoded = tf.keras.layers.Concatenate(name=f"e01_skip_{index}")([decoded, skip])
        decoded = _block(decoded, count, f"e01_decoder_{index}")
    lateral = tf.keras.layers.Conv2D(filters, 33, padding="valid", activation="relu",
        kernel_initializer="he_normal", name="e01_spatial_transform_96_to_64")(decoded)
    depths = tf.keras.layers.Conv2D(24, 1, activation=cfg.output_activation,
        name="e01_susceptibility_depth_channels")(lateral)
    depth_first = tf.keras.layers.Permute((3,1,2), name="e01_depth_first")(depths)
    reshaped = tf.keras.layers.Reshape(SUSCEPTIBILITY_SHAPE, name="susceptibility_fraction")(depth_first)
    outputs = tf.keras.layers.Rescaling(
        cfg.maximum_susceptibility_si, name="recovered_susceptibility_si"
    )(reshaped)
    model = tf.keras.Model(inputs, outputs, name="e01_tmi_inversion")
    if model.output_shape != (None, *SUSCEPTIBILITY_SHAPE): raise RuntimeError("Unexpected E01 output shape.")
    return model

# Clear compatibility alias for callers migrated from the selected experiment.
build_asymmetric_2d_unet_model = build_e01_model

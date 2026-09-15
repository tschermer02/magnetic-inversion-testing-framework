from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class E01ModelConfig:
    base_filters: int = 8
    output_activation: str = "sigmoid"

    def validate(self) -> None:
        if self.base_filters < 1:
            raise ValueError("base_filters must be positive.")
        if self.output_activation not in {"sigmoid", "relu", "linear"}:
            raise ValueError("Unsupported output activation.")


def count_trainable_parameters(model: tf.keras.Model) -> int:
    return int(sum(tf.keras.backend.count_params(weight) for weight in model.trainable_weights))


def _convolution_2d_block(inputs: tf.Tensor, *, filters: int, name: str) -> tf.Tensor:
    x = inputs
    for index in (1, 2):
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
            name=f"{name}_conv_{index}",
        )(x)
    return x


def build_e01_model(config: E01ModelConfig | None = None) -> tf.keras.Model:
    """Mirror the proven E09 asymmetrical 2-D U-Net architecture for TMI-to-susceptibility inversion."""
    if config is None:
        config = E01ModelConfig()
    config.validate()

    filters = config.base_filters
    inputs = tf.keras.Input(shape=(81, 81, 1), name="tmi_map")
    padded = tf.keras.layers.ZeroPadding2D(padding=((7, 8), (7, 8)), name="e01_pad_81_to_96")(inputs)

    encoder_1 = _convolution_2d_block(padded, filters=filters, name="e01_encoder_1")
    pooled_1 = tf.keras.layers.MaxPool2D(2, name="e01_pool_96_to_48")(encoder_1)
    encoder_2 = _convolution_2d_block(pooled_1, filters=filters * 2, name="e01_encoder_2")
    pooled_2 = tf.keras.layers.MaxPool2D(2, name="e01_pool_48_to_24")(encoder_2)
    encoder_3 = _convolution_2d_block(pooled_2, filters=filters * 4, name="e01_encoder_3")
    pooled_3 = tf.keras.layers.MaxPool2D(2, name="e01_pool_24_to_12")(encoder_3)
    bottleneck = _convolution_2d_block(pooled_3, filters=filters * 8, name="e01_bottleneck")

    decoded = bottleneck
    for index, (skip, output_filters, size) in enumerate(
        ((encoder_3, filters * 4, 24), (encoder_2, filters * 2, 48), (encoder_1, filters, 96)),
        start=1,
    ):
        decoded = tf.keras.layers.Conv2DTranspose(
            output_filters,
            kernel_size=2,
            strides=2,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
            name=f"e01_upsample_to_{size}",
        )(decoded)
        decoded = tf.keras.layers.Concatenate(name=f"e01_skip_connection_{index}")([decoded, skip])
        decoded = _convolution_2d_block(decoded, filters=output_filters, name=f"e01_decoder_{index}")

    lateral_64 = tf.keras.layers.Conv2D(
        filters,
        kernel_size=33,
        padding="valid",
        activation="relu",
        kernel_initializer="he_normal",
        name="e01_learned_spatial_transform_96_to_64",
    )(decoded)
    depth_channels = tf.keras.layers.Conv2D(
        24,
        kernel_size=1,
        padding="same",
        activation=config.output_activation,
        name="e01_susceptibility_depth_channels",
    )(lateral_64)
    depth_first = tf.keras.layers.Permute((3, 1, 2), name="e01_permute_depth_channels_first")(depth_channels)
    outputs = tf.keras.layers.Reshape((24, 64, 64, 1), name="recovered_susceptibility")(depth_first)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="e01_asymmetric_2d_unet")
    if model.output_shape != (None, 24, 64, 64, 1):
        raise RuntimeError(f"Unexpected E01 output shape: {model.output_shape}")
    return model

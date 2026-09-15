"""Integrated-sensitivity compensation for the unchanged E01 architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf

from cnn_inversion_3d.diagnostics import build_prediction_diagnostics


@dataclass(frozen=True)
class E01CoreLossConfig:
    lambda_susceptibility: float = 1.0
    lambda_depth: float = 1.0
    alpha_center: float = 1.0
    lambda_sensitivity: float = 1.0
    sensitivity_gamma: float = 0.5
    sensitivity_weight_min: float = 0.5
    sensitivity_weight_max: float = 5.0
    lambda_amplitude: float = 0.0
    small_body_weighting: bool = False
    volume_gamma: float = 0.5
    sample_weight_min: float = 0.5
    sample_weight_max: float = 2.0
    training_median_body_volume_cells: float = 1.0
    training_weight_mean: float = 1.0
    epsilon: float = 1.0e-8
    body_fraction: float = 0.5

    def validate(self) -> None:
        for name in (
            "lambda_susceptibility", "lambda_depth", "alpha_center",
            "lambda_sensitivity", "sensitivity_gamma", "lambda_amplitude",
            "volume_gamma",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must not be negative.")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if not 0.0 < self.sensitivity_weight_min <= 1.0:
            raise ValueError("sensitivity_weight_min must be in (0, 1].")
        if self.sensitivity_weight_max < 1.0:
            raise ValueError("sensitivity_weight_max must be at least one.")
        if self.sensitivity_weight_min > self.sensitivity_weight_max:
            raise ValueError("sensitivity weight minimum exceeds maximum.")
        if not 0.0 < self.body_fraction < 1.0:
            raise ValueError("body_fraction must be between zero and one.")
        if not 0.0 < self.sample_weight_min <= self.sample_weight_max:
            raise ValueError("sample-weight bounds must be positive and ordered.")
        if self.training_median_body_volume_cells <= 0.0:
            raise ValueError("training median body volume must be positive.")
        if self.training_weight_mean <= 0.0:
            raise ValueError("training weight mean must be positive.")


def _bounded_mean_one(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    """Scale positive values so clipped output has mean one and fixed bounds."""

    low, high = 1.0e-12, float(np.max(values)) * 1.0e12
    for _ in range(100):
        scale = np.sqrt(low * high)
        mean = float(np.mean(np.clip(values / scale, minimum, maximum)))
        if mean > 1.0:
            low = scale
        else:
            high = scale
    return np.clip(values / np.sqrt(low * high), minimum, maximum)


def build_e01_sensitivity_weights(
    *, gamma: float = 0.5, epsilon: float = 1.0e-8,
    weight_min: float = 0.5, weight_max: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical integrated sensitivity and bounded mean-one compensation."""

    depths = 5.0 + 10.0 * np.arange(24, dtype=np.float64)
    sensitivity = np.broadcast_to(
        (1.0 / np.power(depths, 3.0))[:, None, None], (24, 64, 64)
    ).copy()
    raw = np.power(float(np.max(sensitivity)) / (sensitivity.astype(np.float64) + epsilon), gamma)
    weights = _bounded_mean_one(raw, weight_min, weight_max)
    if not np.all(np.isfinite(weights)) or not np.isclose(np.mean(weights), 1.0, atol=1.0e-6):
        raise ValueError("E01Core sensitivity weights are not finite and mean-one.")
    if float(np.min(weights)) < weight_min - 1.0e-6 or float(np.max(weights)) > weight_max + 1.0e-6:
        raise ValueError("E01Core sensitivity weights violate configured clipping bounds.")
    return sensitivity.astype(np.float32), weights.astype(np.float32)


def integrated_sensitivity_compensated_mse(
    true_susceptibility: tf.Tensor, predicted_susceptibility: tf.Tensor, weights: tf.Tensor,
) -> tf.Tensor:
    """Per-sample weighted voxel MSE, then averaged across the batch."""

    weight_volume = tf.cast(weights, predicted_susceptibility.dtype)[None, ..., None]
    numerator = tf.reduce_sum(
        weight_volume * tf.square(predicted_susceptibility - true_susceptibility), axis=(1, 2, 3, 4)
    )
    denominator = tf.reduce_sum(weight_volume, axis=(1, 2, 3, 4))
    return tf.reduce_mean(tf.math.divide_no_nan(numerator, denominator))


def susceptibility_amplitude_mse_per_sample(
    true_susceptibility: tf.Tensor, predicted_susceptibility: tf.Tensor, *, epsilon: float = 1.0e-8
) -> tf.Tensor:
    """Squared body-susceptibility contrast error using the same true body mask."""

    mask = tf.cast(true_susceptibility > 0.0, predicted_susceptibility.dtype)
    count = tf.reduce_sum(mask, axis=(1, 2, 3, 4))
    true_mean = tf.math.divide_no_nan(
        tf.reduce_sum(mask * true_susceptibility, axis=(1, 2, 3, 4)), count + epsilon
    )
    predicted_mean = tf.math.divide_no_nan(
        tf.reduce_sum(mask * predicted_susceptibility, axis=(1, 2, 3, 4)), count + epsilon
    )
    return tf.square(predicted_mean - true_mean)


def body_volume_sample_weights(truth: tf.Tensor, config: E01CoreLossConfig) -> tf.Tensor:
    """Return weights based only on true volume and training-derived constants."""

    volume = tf.reduce_sum(tf.cast(truth > 0.0, tf.float32), axis=(1, 2, 3, 4))
    raw = tf.pow(config.training_median_body_volume_cells / volume, config.volume_gamma)
    clipped = tf.clip_by_value(raw, config.sample_weight_min, config.sample_weight_max)
    normalized = clipped / tf.cast(config.training_weight_mean, clipped.dtype)
    return tf.clip_by_value(
        normalized, config.sample_weight_min, config.sample_weight_max
    )


def _per_sample_e01_terms(
    truth: tf.Tensor, prediction: tf.Tensor, weights: tf.Tensor,
    config: E01CoreLossConfig,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    axes = (1, 2, 3, 4)
    error = tf.square(prediction - truth)
    body = tf.cast(truth > 0.0, prediction.dtype)
    background = 1.0 - body
    body_mse = tf.math.divide_no_nan(tf.reduce_sum(error * body, axes), tf.reduce_sum(body, axes))
    background_mse = tf.math.divide_no_nan(
        tf.reduce_sum(error * background, axes), tf.reduce_sum(background, axes)
    )
    susceptibility = config.body_fraction * body_mse + (1.0 - config.body_fraction) * background_mse
    true_profile = tf.reduce_sum(truth, axis=(2, 3, 4))
    predicted_profile = tf.reduce_sum(prediction, axis=(2, 3, 4))
    true_normalized = tf.math.divide_no_nan(
        true_profile, tf.reduce_sum(true_profile, axis=1, keepdims=True) + config.epsilon
    )
    predicted_normalized = tf.math.divide_no_nan(
        predicted_profile,
        tf.reduce_sum(predicted_profile, axis=1, keepdims=True) + config.epsilon,
    )
    profile = tf.reduce_mean(tf.square(predicted_normalized - true_normalized), axis=1)
    z = tf.cast(5.0 + 10.0 * tf.range(24, dtype=tf.float32), prediction.dtype)[None, :]
    true_center = tf.math.divide_no_nan(
        tf.reduce_sum(true_profile * z, axis=1), tf.reduce_sum(true_profile, axis=1) + config.epsilon
    )
    predicted_center = tf.math.divide_no_nan(
        tf.reduce_sum(predicted_profile * z, axis=1),
        tf.reduce_sum(predicted_profile, axis=1) + config.epsilon,
    )
    center = tf.square((predicted_center - true_center) / 230.0)
    weight_volume = tf.cast(weights, prediction.dtype)[None, ..., None]
    sensitivity = tf.math.divide_no_nan(
        tf.reduce_sum(weight_volume * error, axes), tf.reduce_sum(weight_volume, axes)
    )
    amplitude = susceptibility_amplitude_mse_per_sample(
        truth, prediction, epsilon=config.epsilon
    )
    return susceptibility, profile, center, sensitivity, amplitude


class E01CoreTrainingModel(tf.keras.Model):
    """Train unchanged E01 with corrected E01 plus sensitivity compensation."""

    def __init__(
        self, inversion_model: tf.keras.Model, sensitivity_weights: np.ndarray,
        *, loss_config: E01CoreLossConfig,
    ) -> None:
        super().__init__(name="e01_integrated_sensitivity_wrapper")
        loss_config.validate()
        self.inversion_model = inversion_model
        self.loss_config = loss_config
        self.sensitivity_weights = tf.constant(sensitivity_weights, tf.float32)
        self.trackers = {
            name: tf.keras.metrics.Mean(name=name)
            for name in (
                "loss", "susceptibility_loss", "depth_profile_loss", "z_center_loss",
                "depth_loss", "sensitivity_loss", "susceptibility_amplitude_loss",
            )
        }
        self.susceptibility_diagnostics = build_prediction_diagnostics()

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        return [*self.trackers.values(), *self.susceptibility_diagnostics]

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.inversion_model(inputs, training=training)

    def compute_loss_terms(self, tmi: tf.Tensor, truth: tf.Tensor, *, training: bool) -> tuple[tf.Tensor, ...]:
        prediction = self.inversion_model(tmi, training=training)
        cfg = self.loss_config
        per_sample = _per_sample_e01_terms(
            truth, prediction, self.sensitivity_weights, cfg
        )
        sample_weights = (
            body_volume_sample_weights(truth, cfg)
            if training and getattr(cfg, "small_body_weighting", False)
            else tf.ones(tf.shape(truth)[0], tf.float32)
        )
        reduce = lambda values: tf.reduce_mean(tf.cast(sample_weights, values.dtype) * values)
        susceptibility, profile, center, sensitivity, amplitude = map(reduce, per_sample)
        depth = profile + cfg.alpha_center * center
        total = (
            cfg.lambda_susceptibility * susceptibility + cfg.lambda_depth * depth
            + cfg.lambda_sensitivity * sensitivity
            + getattr(cfg, "lambda_amplitude", 0.0) * amplitude
        )
        return prediction, susceptibility, profile, center, depth, sensitivity, amplitude, total

    def _update(self, truth: tf.Tensor, terms: tuple[tf.Tensor, ...]) -> None:
        prediction, susceptibility, profile, center, depth, sensitivity, amplitude, total = terms
        for name, value in (
            ("loss", total), ("susceptibility_loss", susceptibility), ("depth_profile_loss", profile),
            ("z_center_loss", center), ("depth_loss", depth),
            ("sensitivity_loss", sensitivity), ("susceptibility_amplitude_loss", amplitude),
        ):
            self.trackers[name].update_state(value)
        for metric in self.susceptibility_diagnostics:
            metric.update_state(truth, prediction)

    def train_step(self, data: Any) -> dict[str, tf.Tensor]:
        tmi, truth, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            terms = self.compute_loss_terms(tmi, truth, training=True)
        gradients = tape.gradient(terms[-1], self.inversion_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.inversion_model.trainable_variables))
        self._update(truth, terms)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data: Any) -> dict[str, tf.Tensor]:
        tmi, truth, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        terms = self.compute_loss_terms(tmi, truth, training=False)
        self._update(truth, terms)
        return {metric.name: metric.result() for metric in self.metrics}

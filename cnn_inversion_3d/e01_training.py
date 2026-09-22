"""Soft-Tversky occupancy supervision for controlled experiment E01Core-12."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf

from cnn_inversion_3d.e01_physics_loss import (
    E01PhysicsLossConfig,
    E01PhysicsTrainingModel,
)


@dataclass(frozen=True)
class E01LossConfig(E01PhysicsLossConfig):
    """E01Core-10 loss configuration plus soft-Tversky occupancy supervision."""

    lambda_tversky: float = 0.0
    tversky_alpha: float = 0.7
    tversky_beta: float = 0.3
    occupancy_threshold: float = 0.1
    occupancy_sharpness: float = 10.0
    occupancy_mode: str = "legacy_threshold_sigmoid"
    occupancy_tau_si: float = 1.0e-4 / np.log(100.0)

    def validate(self) -> None:
        super().validate()
        if self.lambda_tversky < 0.0:
            raise ValueError("lambda_tversky must not be negative.")
        if self.tversky_alpha < 0.0 or self.tversky_beta < 0.0:
            raise ValueError("Tversky alpha and beta must not be negative.")
        if self.tversky_alpha + self.tversky_beta <= 0.0:
            raise ValueError("At least one Tversky error weight must be positive.")
        if not 0.0 < self.occupancy_threshold < 1.0:
            raise ValueError("occupancy_threshold must be between zero and one.")
        if self.occupancy_sharpness <= 0.0:
            raise ValueError("occupancy_sharpness must be positive.")
        if self.occupancy_mode not in {"legacy_threshold_sigmoid", "geological_exponential"}:
            raise ValueError("Unknown occupancy_mode.")
        if self.occupancy_tau_si <= 0.0 or not np.isfinite(self.occupancy_tau_si):
            raise ValueError("occupancy_tau_si must be positive and finite.")


def geological_support_mask(truth: tf.Tensor, body_mask: tf.Tensor | None = None) -> tf.Tensor:
    """Use an explicit generated mask, or exact-positive support for synthetic E02."""
    values = tf.convert_to_tensor(truth)
    if body_mask is None:
        return tf.cast(values > 0.0, values.dtype)
    mask = tf.cast(body_mask, values.dtype)
    tf.debugging.assert_equal(tf.shape(mask), tf.shape(values), message="Body-mask shape mismatch.")
    return mask


def exponential_soft_occupancy(prediction: tf.Tensor, *, tau_si: float) -> tf.Tensor:
    """Stable p=1-exp(-chi/tau), exactly zero at chi=0 and bounded in [0,1]."""
    values = tf.convert_to_tensor(prediction)
    nonnegative = tf.maximum(values, tf.cast(0.0, values.dtype))
    occupied = -tf.math.expm1(-nonnegative / tf.cast(tau_si, values.dtype))
    return tf.clip_by_value(occupied, 0.0, 1.0)


def soft_tversky_loss_per_sample(
    truth: tf.Tensor,
    prediction: tf.Tensor,
    *,
    threshold: float = 0.1,
    sharpness: float = 10.0,
    alpha: float = 0.7,
    beta: float = 0.3,
    epsilon: float = 1.0e-8,
    occupancy_mode: str = "legacy_threshold_sigmoid",
    occupancy_tau_si: float = 1.0e-4 / np.log(100.0),
    body_mask: tf.Tensor | None = None,
) -> tf.Tensor:
    """Return differentiable occupancy Tversky loss for each batch sample.

    The canonical target support is deterministic. Predicted occupancy remains
    soft and differentiable, centered on the same 0.1 dimensionless SI threshold used by
    the evaluation pipeline. ``alpha`` weights false positives and ``beta``
    weights false negatives.
    """

    values = tf.convert_to_tensor(prediction)
    if occupancy_mode == "legacy_threshold_sigmoid":
        target = tf.cast(truth >= tf.cast(threshold, truth.dtype), values.dtype)
        raw_occupied = tf.sigmoid(tf.cast(sharpness, values.dtype) *
                                  (values - tf.cast(threshold, values.dtype)))
        floor = tf.sigmoid(-tf.cast(sharpness, values.dtype) * tf.cast(threshold, values.dtype))
        occupied = tf.clip_by_value((raw_occupied - floor) / (1.0 - floor), 0.0, 1.0)
    elif occupancy_mode == "geological_exponential":
        target = geological_support_mask(truth, body_mask)
        occupied = exponential_soft_occupancy(values, tau_si=occupancy_tau_si)
    else:
        raise ValueError(f"Unknown occupancy_mode: {occupancy_mode}")
    axes = (1, 2, 3, 4)
    true_positive = tf.reduce_sum(target * occupied, axis=axes)
    false_positive = tf.reduce_sum((1.0 - target) * occupied, axis=axes)
    false_negative = tf.reduce_sum(target * (1.0 - occupied), axis=axes)
    smooth = tf.cast(epsilon, values.dtype)
    score = (true_positive + smooth) / (
        true_positive
        + tf.cast(alpha, values.dtype) * false_positive
        + tf.cast(beta, values.dtype) * false_negative
        + smooth
    )
    return 1.0 - score


class E01TrainingModel(E01PhysicsTrainingModel):
    """Train the unchanged E01 model with E01Core-10 plus soft Tversky."""

    def __init__(
        self,
        inversion_model: tf.keras.Model,
        sensitivity_weights: np.ndarray,
        forward_operator: tf.keras.layers.Layer,
        *,
        tmi_scale: float,
        loss_config: E01LossConfig,
    ) -> None:
        loss_config.validate()
        super().__init__(
            inversion_model,
            sensitivity_weights,
            forward_operator,
            tmi_scale=tmi_scale,
            loss_config=loss_config,
        )
        self._name = "e01_soft_tversky_wrapper"
        self.trackers["tversky_loss"] = tf.keras.metrics.Mean(name="tversky_loss")
        self.trackers["weighted_tversky_loss"] = tf.keras.metrics.Mean(
            name="weighted_tversky_loss"
        )
        self.trackers["global_gradient_norm"] = tf.keras.metrics.Mean(name="global_gradient_norm")

    def compute_loss_terms(
        self, tmi: tf.Tensor, truth: tf.Tensor, *, training: bool
    ) -> tuple[tf.Tensor, ...]:
        base = super().compute_loss_terms(tmi, truth, training=training)
        cfg = self.loss_config
        tversky = tf.reduce_mean(
            soft_tversky_loss_per_sample(
                truth,
                base[0],
                threshold=cfg.occupancy_threshold,
                sharpness=cfg.occupancy_sharpness,
                alpha=cfg.tversky_alpha,
                beta=cfg.tversky_beta,
                epsilon=cfg.epsilon,
                occupancy_mode=cfg.occupancy_mode,
                occupancy_tau_si=cfg.occupancy_tau_si,
            )
        )
        weighted_tversky = cfg.lambda_tversky * tversky
        total = base[-1] + weighted_tversky
        return (*base[:-1], tversky, weighted_tversky, total)

    def _update_e01(
        self,
        tmi: tf.Tensor,
        truth: tf.Tensor,
        terms: tuple[tf.Tensor, ...],
    ) -> None:
        # Restore the E01Core-9/10/11 tuple layout while retaining E01Core-12 total.
        super()._update_extended(tmi, truth, (*terms[:10], terms[-1]))
        self.trackers["tversky_loss"].update_state(terms[10])
        self.trackers["weighted_tversky_loss"].update_state(terms[11])

    def train_step(self, data: Any) -> dict[str, tf.Tensor]:
        tmi, truth, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            terms = self.compute_loss_terms(tmi, truth, training=True)
            tf.debugging.assert_all_finite(terms[-1], "Nonfinite E01/E02 total loss.")
        gradients = tape.gradient(terms[-1], self.inversion_model.trainable_variables)
        finite_gradients = [gradient for gradient in gradients if gradient is not None]
        if not finite_gradients:
            raise ValueError("No gradients were produced for the inversion model.")
        for gradient in finite_gradients:
            tf.debugging.assert_all_finite(gradient, "Nonfinite E01/E02 gradient.")
        gradient_norm = tf.linalg.global_norm(finite_gradients)
        tf.debugging.assert_all_finite(gradient_norm, "Nonfinite global gradient norm.")
        gradient_pairs = [(gradient, variable) for gradient, variable in
                          zip(gradients, self.inversion_model.trainable_variables)
                          if gradient is not None]
        self.optimizer.apply_gradients(gradient_pairs)
        self._update_e01(tmi, truth, terms)
        self.trackers["global_gradient_norm"].update_state(gradient_norm)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data: Any) -> dict[str, tf.Tensor]:
        tmi, truth, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        terms = self.compute_loss_terms(tmi, truth, training=False)
        self._update_e01(tmi, truth, terms)
        return {metric.name: metric.result() for metric in self.metrics}

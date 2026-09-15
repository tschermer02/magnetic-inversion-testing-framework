"""Body-susceptibility and physics ablations layered onto unchanged E01Core-6-prime."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf

def global_normalized_tmi_mse(true_tmi, predicted_tmi, *, tmi_scale):
    scale = tf.cast(tmi_scale, predicted_tmi.dtype)
    return tf.reduce_mean(tf.square(predicted_tmi / scale - tf.cast(true_tmi, predicted_tmi.dtype) / scale))
from cnn_inversion_3d.e01_core_loss import E01CoreLossConfig, E01CoreTrainingModel


@dataclass(frozen=True)
class E01PhysicsLossConfig(E01CoreLossConfig):
    lambda_body_susceptibility: float = 0.0
    lambda_tmi: float = 0.0

    def validate(self) -> None:
        super().validate()
        if self.lambda_body_susceptibility < 0.0 or self.lambda_tmi < 0.0:
            raise ValueError("E01Core-9/10/11 loss weights must not be negative.")


def body_susceptibility_mse_per_sample(
    truth: tf.Tensor, prediction: tf.Tensor, *, epsilon: float = 1.0e-8
) -> tf.Tensor:
    """Voxel-susceptibility MSE restricted to each sample's true body support."""

    mask = tf.cast(truth > 0.0, prediction.dtype)
    axes = (1, 2, 3, 4)
    return tf.math.divide_no_nan(
        tf.reduce_sum(mask * tf.square(prediction - truth), axis=axes),
        tf.reduce_sum(mask, axis=axes) + tf.cast(epsilon, prediction.dtype),
    )


class E01PhysicsTrainingModel(E01CoreTrainingModel):
    """Add optional body-only susceptibility and fixed forward-tmi objectives."""

    def __init__(self, inversion_model: tf.keras.Model, sensitivity_weights: np.ndarray,
                 forward_operator: tf.keras.layers.Layer, *, tmi_scale: float,
                 loss_config: E01PhysicsLossConfig) -> None:
        loss_config.validate()
        super().__init__(inversion_model, sensitivity_weights, loss_config=loss_config)
        self._name = "e01_susceptibility_physics_ablation_wrapper"
        self.forward_operator = forward_operator
        self.tmi_scale = float(tmi_scale)
        if self.tmi_scale <= 0.0: raise ValueError("tmi_scale must be positive.")
        tracker_names = ["body_susceptibility_loss"]
        if loss_config.lambda_tmi > 0.0:
            tracker_names.extend(("tmi_loss", "weighted_tmi_loss",
                                  "tmi_rmse", "tmi_correlation"))
        for name in tracker_names:
            self.trackers[name] = tf.keras.metrics.Mean(name=name)

    def compute_loss_terms(self, tmi: tf.Tensor, truth: tf.Tensor, *, training: bool) -> tuple[tf.Tensor, ...]:
        base = super().compute_loss_terms(tmi, truth, training=training)
        prediction = base[0]; cfg = self.loss_config
        body_susceptibility = tf.reduce_mean(body_susceptibility_mse_per_sample(
            truth, prediction, epsilon=cfg.epsilon))
        if cfg.lambda_tmi > 0.0:
            true_tmi = tf.cast(tmi * self.tmi_scale, tf.float32)
            predicted_tmi = self.forward_operator(prediction)
            tmi_loss = global_normalized_tmi_mse(
                true_tmi, predicted_tmi, tmi_scale=self.tmi_scale)
        else:
            tmi_loss = tf.zeros((), dtype=prediction.dtype)
        weighted_tmi = cfg.lambda_tmi * tmi_loss
        total = base[-1] + cfg.lambda_body_susceptibility * body_susceptibility + weighted_tmi
        return (*base[:-1], body_susceptibility, tmi_loss, weighted_tmi, total)

    def _update_extended(self, tmi: tf.Tensor, truth: tf.Tensor,
                         terms: tuple[tf.Tensor, ...]) -> None:
        super()._update(truth, (*terms[:7], terms[-1]))
        body_susceptibility, tmi_loss, weighted_tmi = terms[7:10]
        self.trackers["body_susceptibility_loss"].update_state(body_susceptibility)
        if self.loss_config.lambda_tmi <= 0.0:
            return
        prediction = terms[0]
        true_tmi = tf.cast(tmi * self.tmi_scale, tf.float32)
        predicted_tmi = self.forward_operator(prediction)
        residual = predicted_tmi - true_tmi
        true_flat = tf.reshape(true_tmi, (tf.shape(true_tmi)[0], -1))
        predicted_flat = tf.reshape(predicted_tmi, (tf.shape(predicted_tmi)[0], -1))
        true_centered = true_flat - tf.reduce_mean(true_flat, axis=1, keepdims=True)
        predicted_centered = predicted_flat - tf.reduce_mean(predicted_flat, axis=1, keepdims=True)
        correlation = tf.math.divide_no_nan(
            tf.reduce_sum(true_centered * predicted_centered, axis=1),
            tf.norm(true_centered, axis=1) * tf.norm(predicted_centered, axis=1),
        )
        for name, value in (("tmi_loss", tmi_loss),
                            ("weighted_tmi_loss", weighted_tmi),
                            ("tmi_rmse", tf.sqrt(tf.reduce_mean(tf.square(residual)))),
                            ("tmi_correlation", tf.reduce_mean(correlation))):
            self.trackers[name].update_state(value)

    def train_step(self, data: Any) -> dict[str, tf.Tensor]:
        tmi, truth, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            terms = self.compute_loss_terms(tmi, truth, training=True)
        gradients = tape.gradient(terms[-1], self.inversion_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.inversion_model.trainable_variables))
        self._update_extended(tmi, truth, terms)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data: Any) -> dict[str, tf.Tensor]:
        tmi, truth, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        terms = self.compute_loss_terms(tmi, truth, training=False)
        self._update_extended(tmi, truth, terms)
        return {metric.name: metric.result() for metric in self.metrics}

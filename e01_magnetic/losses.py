from __future__ import annotations

import numpy as np
import tensorflow as tf


class TMIPhysicsLoss(tf.keras.losses.Loss):
    """Physical TMI consistency term using the non-trainable forward sensitivity matrix."""

    def __init__(
        self,
        forward_operator,
        *,
        scale: float = 1.0,
        tmi_scale: float = 1.0,
        susceptibility_scale: float = 1.0,
        reduction: str = "none",
        name: str = "tmi_forward_loss",
    ) -> None:
        super().__init__(reduction=reduction, name=name)
        self.forward_operator = forward_operator
        self.scale = float(scale)
        self.tmi_scale = float(tmi_scale)
        self.susceptibility_scale = float(susceptibility_scale)
        try:
            A = np.asarray(self.forward_operator.sensitivity_matrix(), dtype=np.float32)
        except Exception:
            A = np.asarray(self.forward_operator.sensitivity_matrix_chunked(), dtype=np.float32)
        self.A = tf.constant(A, dtype=tf.float32)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        true_tmi = tf.cast(y_true, tf.float32)
        pred_kappa = tf.cast(y_pred, tf.float32)

        if true_tmi.shape.rank == 1:
            true_tmi = true_tmi[None, :]
        if pred_kappa.shape.rank == 1:
            pred_kappa = pred_kappa[None, :]

        physical_true_tmi = true_tmi * self.tmi_scale
        physical_pred_kappa = pred_kappa * self.susceptibility_scale
        predicted_tmi = tf.linalg.matmul(physical_pred_kappa, self.A, transpose_b=True)
        residual = predicted_tmi - physical_true_tmi
        loss = tf.reduce_mean(tf.square(residual), axis=-1)
        return self.scale * loss


def tmi_loss_gradient_test(forward_operator, *, seed: int = 0) -> tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    model = rng.uniform(0.01, 0.5, size=forward_operator.grid.n_cells).astype(np.float64)
    true_tmi = np.asarray(forward_operator.predict(model), dtype=np.float64)
    pred = tf.Variable(model.astype(np.float64), dtype=tf.float64)

    with tf.autodiff.ForwardAccumulator((pred,), tf.float64) as acc:
        loss = TMIPhysicsLoss(forward_operator, scale=1.0)(tf.constant(true_tmi, dtype=tf.float64), pred)
        grad = acc.jvp(loss)

    grad_value = np.asarray(grad, dtype=np.float64)
    return float(np.linalg.norm(grad_value)), grad_value

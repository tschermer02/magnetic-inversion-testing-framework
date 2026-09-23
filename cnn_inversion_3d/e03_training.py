"""E03 amplitude-aware susceptibility and differentiable-TMI training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import tensorflow as tf

from cnn_inversion_3d.e01_training import E01LossConfig, E01TrainingModel


@dataclass(frozen=True)
class E03LossConfig(E01LossConfig):
    susceptibility_floor_si: float = 1e-4
    lambda_susceptibility_relative: float = 0.0
    tmi_rms_floor_nt: float = 1.0
    tmi_reference_nt: float = 100.0
    lambda_tmi_absolute: float = 0.0
    lambda_tmi_relative: float = 0.0

    def validate(self):
        super().validate()
        for name in ("susceptibility_floor_si","tmi_rms_floor_nt","tmi_reference_nt"):
            if not np.isfinite(getattr(self,name)) or getattr(self,name) <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        for name in ("lambda_susceptibility_relative","lambda_tmi_absolute","lambda_tmi_relative"):
            if not np.isfinite(getattr(self,name)) or getattr(self,name) < 0:
                raise ValueError(f"{name} must be nonnegative and finite.")


def relative_balanced_susceptibility_mse_per_sample(truth, prediction, *, floor_si=1e-4,
                                                     body_mask=None):
    """Balanced relative MSE with a stopped, truth-only per-sample scale."""
    truth = tf.convert_to_tensor(truth)
    prediction = tf.cast(prediction, truth.dtype)
    body = tf.cast(truth > 0 if body_mask is None else body_mask, truth.dtype)
    axes = (1,2,3,4)
    count = tf.reduce_sum(body, axes)
    amplitude = tf.math.divide_no_nan(tf.reduce_sum(body*truth,axes), count)
    scale = tf.stop_gradient(tf.maximum(amplitude, tf.cast(floor_si,truth.dtype)))
    relative_square = tf.square((prediction-truth)/scale[:,None,None,None,None])
    background = 1-body
    body_mse = tf.math.divide_no_nan(tf.reduce_sum(relative_square*body,axes),count)
    background_mse = tf.math.divide_no_nan(tf.reduce_sum(relative_square*background,axes),
                                           tf.reduce_sum(background,axes))
    # Empty geological bodies have no body term and retain a finite background term.
    return 0.5*body_mse + 0.5*background_mse


def tmi_consistency_losses_per_sample(observed_normalized, predicted_nt, *, tmi_scale_nt,
                                      rms_floor_nt, reference_nt=100.0):
    """Return absolute-reference and floored-relative TMI MSE per sample."""
    observed_nt = tf.cast(observed_normalized, predicted_nt.dtype) * tf.cast(tmi_scale_nt,predicted_nt.dtype)
    rank=predicted_nt.shape.rank
    if rank is None or rank < 2:
        raise ValueError("TMI tensors require a batch dimension and at least one data dimension.")
    axes=tuple(range(1,rank))
    residual=predicted_nt-observed_nt
    rms=tf.sqrt(tf.reduce_mean(tf.square(observed_nt),axis=axes))
    scale=tf.stop_gradient(tf.maximum(rms,tf.cast(rms_floor_nt,predicted_nt.dtype)))
    absolute=tf.reduce_mean(tf.square(residual/tf.cast(reference_nt,predicted_nt.dtype)),axis=axes)
    broadcast_scale=tf.reshape(scale,[-1]+[1]*(rank-1))
    relative=tf.reduce_mean(tf.square(residual/broadcast_scale),axis=axes)
    return absolute,relative,rms


class E03TrainingModel(E01TrainingModel):
    def __init__(self,*args,loss_config:E03LossConfig,**kwargs):
        loss_config.validate(); super().__init__(*args,loss_config=loss_config,**kwargs)
        for name in ("susceptibility_relative_loss","weighted_susceptibility_relative_loss",
                     "tmi_absolute_loss","weighted_tmi_absolute_loss","tmi_relative_loss",
                     "weighted_tmi_relative_loss","observed_tmi_rms_nt"):
            self.trackers[name]=tf.keras.metrics.Mean(name=name)

    def compute_loss_terms(self,tmi,truth,*,training):
        base=super().compute_loss_terms(tmi,truth,training=training)
        cfg=self.loss_config; prediction=base[0]
        relative=tf.reduce_mean(relative_balanced_susceptibility_mse_per_sample(
            truth,prediction,floor_si=cfg.susceptibility_floor_si))
        weighted_relative=cfg.lambda_susceptibility_relative*relative
        if cfg.lambda_tmi_absolute>0 or cfg.lambda_tmi_relative>0:
            predicted_tmi=self.forward_operator(prediction)
            absolute_values,relative_values,rms=tmi_consistency_losses_per_sample(
                tmi,predicted_tmi,tmi_scale_nt=self.tmi_scale,rms_floor_nt=cfg.tmi_rms_floor_nt,
                reference_nt=cfg.tmi_reference_nt)
            absolute=tf.reduce_mean(absolute_values); tmi_relative=tf.reduce_mean(relative_values)
            observed_rms=tf.reduce_mean(rms)
        else:
            absolute=tmi_relative=observed_rms=tf.zeros((),prediction.dtype)
        weighted_absolute=cfg.lambda_tmi_absolute*absolute
        weighted_tmi_relative=cfg.lambda_tmi_relative*tmi_relative
        total=base[-1]+weighted_relative+weighted_absolute+weighted_tmi_relative
        return (*base[:-1],relative,weighted_relative,absolute,weighted_absolute,
                tmi_relative,weighted_tmi_relative,observed_rms,total)

    def _update_e03(self,tmi,truth,terms):
        super()._update_e01(tmi,truth,(*terms[:12],terms[-1]))
        for name,value in zip(("susceptibility_relative_loss","weighted_susceptibility_relative_loss",
            "tmi_absolute_loss","weighted_tmi_absolute_loss","tmi_relative_loss",
            "weighted_tmi_relative_loss","observed_tmi_rms_nt"),terms[12:19]):
            self.trackers[name].update_state(value)

    def train_step(self,data:Any):
        tmi,truth,_=tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            terms=self.compute_loss_terms(tmi,truth,training=True)
            tf.debugging.assert_all_finite(terms[-1],"Nonfinite E03 loss.")
        gradients=tape.gradient(terms[-1],self.inversion_model.trainable_variables)
        pairs=[(g,v) for g,v in zip(gradients,self.inversion_model.trainable_variables) if g is not None]
        for gradient,_ in pairs: tf.debugging.assert_all_finite(gradient,"Nonfinite E03 gradient.")
        norm=tf.linalg.global_norm([g for g,_ in pairs]); tf.debugging.assert_all_finite(norm,"Nonfinite E03 gradient norm.")
        self.optimizer.apply_gradients(pairs); self._update_e03(tmi,truth,terms)
        self.trackers["global_gradient_norm"].update_state(norm)
        return {metric.name:metric.result() for metric in self.metrics}

    def test_step(self,data):
        tmi,truth,_=tf.keras.utils.unpack_x_y_sample_weight(data)
        terms=self.compute_loss_terms(tmi,truth,training=False); self._update_e03(tmi,truth,terms)
        return {metric.name:metric.result() for metric in self.metrics}

"""E05 loss wrapper: unchanged E01 components plus supervised vertical gradients."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import tensorflow as tf
from cnn_inversion_3d.e01_training import E01LossConfig,E01TrainingModel

@dataclass(frozen=True)
class E05LossConfig(E01LossConfig):
    lambda_vertical_gradient:float=0
    susceptibility_reference_si:float=.1
    def validate(self):
        super().validate()
        if self.lambda_vertical_gradient<0: raise ValueError("lambda_vertical_gradient must be nonnegative")
        if self.susceptibility_reference_si<=0: raise ValueError("susceptibility_reference_si must be positive")

def vertical_gradient_loss_per_sample(truth,prediction,reference_si=.1):
    """MAE between adjacent-Z differences on the current uniform 10 m grid."""
    truth=tf.convert_to_tensor(truth);prediction=tf.cast(prediction,truth.dtype)
    scale=tf.cast(reference_si,truth.dtype)
    true_delta=truth[:,1:,...]/scale-truth[:,:-1,...]/scale
    predicted_delta=prediction[:,1:,...]/scale-prediction[:,:-1,...]/scale
    return tf.reduce_mean(tf.abs(predicted_delta-true_delta),axis=(1,2,3,4))

class E05TrainingModel(E01TrainingModel):
    def __init__(self,*args,loss_config:E05LossConfig,**kwargs):
        loss_config.validate();super().__init__(*args,loss_config=loss_config,**kwargs)
        self._name="e05_training_wrapper"
        self.trackers["vertical_gradient_loss"]=tf.keras.metrics.Mean(name="vertical_gradient_loss")
        self.trackers["weighted_vertical_gradient_loss"]=tf.keras.metrics.Mean(name="weighted_vertical_gradient_loss")
    def compute_loss_terms(self,tmi,truth,*,training):
        base=super().compute_loss_terms(tmi,truth,training=training);prediction=base[0]
        vertical=tf.reduce_mean(vertical_gradient_loss_per_sample(truth,prediction,
            self.loss_config.susceptibility_reference_si))
        weighted=self.loss_config.lambda_vertical_gradient*vertical
        return (*base[:-1],vertical,weighted,base[-1]+weighted)
    def _update_e05(self,tmi,truth,terms):
        super()._update_e01(tmi,truth,(*terms[:12],terms[-1]))
        self.trackers["vertical_gradient_loss"].update_state(terms[12])
        self.trackers["weighted_vertical_gradient_loss"].update_state(terms[13])
    def train_step(self,data:Any):
        tmi,truth,_=tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape: terms=self.compute_loss_terms(tmi,truth,training=True)
        tf.debugging.assert_all_finite(terms[-1],"Nonfinite E05 loss")
        gradients=tape.gradient(terms[-1],self.inversion_model.trainable_variables)
        pairs=[(g,v) for g,v in zip(gradients,self.inversion_model.trainable_variables) if g is not None]
        if not pairs: raise ValueError("E05 produced no gradients")
        for gradient,_ in pairs: tf.debugging.assert_all_finite(gradient,"Nonfinite E05 gradient")
        norm=tf.linalg.global_norm([g for g,_ in pairs]);self.optimizer.apply_gradients(pairs)
        self._update_e05(tmi,truth,terms);self.trackers["global_gradient_norm"].update_state(norm)
        return {metric.name:metric.result() for metric in self.metrics}
    def test_step(self,data):
        tmi,truth,_=tf.keras.utils.unpack_x_y_sample_weight(data)
        terms=self.compute_loss_terms(tmi,truth,training=False);self._update_e05(tmi,truth,terms)
        return {metric.name:metric.result() for metric in self.metrics}

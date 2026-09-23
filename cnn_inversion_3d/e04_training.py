"""E04 wrapper: the exact active E01 objective plus absolute TMI consistency."""
from __future__ import annotations
import tensorflow as tf
from cnn_inversion_3d.e01_core_loss import E01CoreTrainingModel
from cnn_inversion_3d.e01_training import E01TrainingModel

class E04TrainingModel(E01TrainingModel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.loss_config.lambda_tmi<=0: raise ValueError("E04 requires lambda_tmi > 0")
        self._name="e04_e01_plus_tmi_wrapper"
        self.trackers["tmi_mae_nt"]=tf.keras.metrics.Mean(name="tmi_mae_nt")
    def _update_e01(self,tmi,truth,terms):
        # Preserve the E01 tracker definitions while doing only one additional
        # diagnostic forward pass (the base implementation would not include MAE).
        E01CoreTrainingModel._update(self,truth,(*terms[:7],terms[-1]))
        self.trackers["body_susceptibility_loss"].update_state(terms[7])
        self.trackers["tmi_loss"].update_state(terms[8])
        self.trackers["weighted_tmi_loss"].update_state(terms[9])
        self.trackers["tversky_loss"].update_state(terms[10])
        self.trackers["weighted_tversky_loss"].update_state(terms[11])
        predicted_tmi=self.forward_operator(terms[0])
        observed_tmi=tf.cast(tmi*self.tmi_scale,predicted_tmi.dtype)
        residual=predicted_tmi-observed_tmi
        true_flat=tf.reshape(observed_tmi,(tf.shape(observed_tmi)[0],-1))
        predicted_flat=tf.reshape(predicted_tmi,(tf.shape(predicted_tmi)[0],-1))
        true_centered=true_flat-tf.reduce_mean(true_flat,axis=1,keepdims=True)
        predicted_centered=predicted_flat-tf.reduce_mean(predicted_flat,axis=1,keepdims=True)
        correlation=tf.math.divide_no_nan(tf.reduce_sum(true_centered*predicted_centered,axis=1),
            tf.norm(true_centered,axis=1)*tf.norm(predicted_centered,axis=1))
        self.trackers["tmi_rmse"].update_state(tf.sqrt(tf.reduce_mean(tf.square(residual))))
        self.trackers["tmi_correlation"].update_state(tf.reduce_mean(correlation))
        self.trackers["tmi_mae_nt"].update_state(tf.reduce_mean(tf.abs(residual)))

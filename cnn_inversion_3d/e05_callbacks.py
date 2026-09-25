"""Metric checkpointing, resumable state, and gradient diagnostics for E05."""
from __future__ import annotations
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import tensorflow as tf

def _utc(): return datetime.now(timezone.utc).isoformat()

def validation_selection_metrics(model,dataset,threshold_si=.001):
    """Per-sample aggregation; empty truth has body MAE 0 and standard empty-mask IoU."""
    body_mae=[];ious=[]
    for tmi,truth in dataset:
        prediction=model.inversion_model(tmi,training=False)
        for true_item,pred_item in zip(truth,prediction):
            body=true_item>0
            body_values=tf.boolean_mask(tf.abs(pred_item-true_item),body)
            body_mae.append(float(tf.reduce_mean(body_values)) if int(tf.size(body_values)) else 0.)
            true_support=true_item>=threshold_si;pred_support=pred_item>=threshold_si
            intersection=float(tf.reduce_sum(tf.cast(true_support&pred_support,tf.float32)))
            union=float(tf.reduce_sum(tf.cast(true_support|pred_support,tf.float32)))
            ious.append(intersection/union if union else 1.)
    return {"true_body_susceptibility_mae_si":float(np.mean(body_mae)),
            "support_iou":float(np.mean(ious)),"sample_count":len(body_mae),
            "empty_true_support_policy":"body MAE=0; both-empty IoU=1; one-empty IoU=0"}

class PrimaryMetricCheckpoint(tf.keras.callbacks.Callback):
    """Select minimum body MAE, tie-break maximum IoU; stop on this rule."""
    def __init__(self,validation,output,threshold_si=.001,patience=10,resume_state=None):
        super().__init__();self.validation=validation;self.output=Path(output);self.threshold=threshold_si
        self.patience=patience;state=resume_state or {};self.best_mae=float(state.get("best_mae",np.inf))
        self.best_iou=float(state.get("best_iou",-np.inf));self.best_epoch=state.get("best_epoch")
        self.wait=int(state.get("wait",0));self.history=list(state.get("history",[]))
    def on_epoch_end(self,epoch,logs=None):
        values=validation_selection_metrics(self.model,self.validation,self.threshold)
        mae=values["true_body_susceptibility_mae_si"];iou=values["support_iou"]
        improved=mae<self.best_mae-1e-12 or (abs(mae-self.best_mae)<=1e-12 and iou>self.best_iou)
        record={"epoch":epoch+1,**values,"selected":improved,"timestamp_utc":_utc()};self.history.append(record)
        if improved:
            self.best_mae=mae;self.best_iou=iou;self.best_epoch=epoch+1;self.wait=0
            self.model.save_weights(self.output/"selected.weights.h5")
        else:self.wait+=1
        state={"selection_rule":"minimum validation true-body susceptibility MAE; tie-break higher support IoU",
            "best_mae":self.best_mae,"best_iou":self.best_iou,"best_epoch":self.best_epoch,
            "wait":self.wait,"history":self.history}
        (self.output/"primary_checkpoint.json").write_text(json.dumps(state,indent=2),encoding="utf-8")
        print(f"\nE05 primary metric: body_MAE={mae:.7g}, IoU={iou:.5f}, selected={improved}")
        if self.wait>=self.patience:
            self.model.stop_training=True;print(f"E05 primary-metric early stopping after {self.wait} non-improving epochs")

class ResumeCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self,manager,status_path):super().__init__();self.manager=manager;self.status_path=Path(status_path)
    def on_epoch_end(self,epoch,logs=None):
        path=self.manager.save(checkpoint_number=epoch+1)
        state={"last_completed_epoch":epoch+1,"tensorflow_checkpoint":path,
            "optimizer_state_restored_on_resume":True,"timestamp_utc":_utc()}
        self.status_path.write_text(json.dumps(state,indent=2),encoding="utf-8")

def _gradient_vector(gradients,variables):
    parts=[];missing=0
    for gradient,variable in zip(gradients,variables):
        if gradient is None: parts.append(tf.zeros_like(tf.reshape(variable,[-1])));missing+=1
        else: parts.append(tf.reshape(tf.convert_to_tensor(gradient),[-1]))
    return tf.concat(parts,0),missing

class GradientDiagnostics(tf.keras.callbacks.Callback):
    def __init__(self,batch,output,interval=5):
        super().__init__();self.tmi,self.truth=batch;self.output=Path(output);self.interval=interval
    def on_epoch_end(self,epoch,logs=None):
        if (epoch+1)%self.interval and epoch!=0:return
        variables=self.model.inversion_model.trainable_variables;cfg=self.model.loss_config
        with tf.GradientTape(persistent=True) as tape:
            terms=self.model.compute_loss_terms(self.tmi,self.truth,training=False)
            components={"susceptibility":cfg.lambda_susceptibility*terms[1],
                "depth":cfg.lambda_depth*terms[4],"sensitivity":cfg.lambda_sensitivity*terms[5],
                "amplitude":cfg.lambda_amplitude*terms[6],"body_susceptibility":cfg.lambda_body_susceptibility*terms[7],
                "tmi":terms[9],"tversky":terms[11],"vertical_gradient":terms[13]}
        vectors={};items={}
        for name,value in components.items():
            vector,missing=_gradient_vector(tape.gradient(value,variables),variables);vectors[name]=vector
            norm=float(tf.norm(vector));items[name]={"weighted_loss":float(value),"gradient_norm":norm,
                "missing_variable_gradients":missing,"zero_gradient":norm==0,"finite":bool(tf.reduce_all(tf.math.is_finite(vector)))}
        reference=vectors["susceptibility"];reference_norm=tf.norm(reference)
        for name,vector in vectors.items():
            denominator=reference_norm*tf.norm(vector)
            items[name]["cosine_with_susceptibility"]=float(tf.math.divide_no_nan(tf.reduce_sum(reference*vector),denominator))
        prediction=terms[0];body=self.truth>0;background=~body
        def stats(values):
            array=np.asarray(tf.boolean_mask(prediction,values));
            if not array.size:return {"count":0}
            return {"count":int(array.size),"mean":float(array.mean()),"min":float(array.min()),"max":float(array.max()),
                "p01":float(np.quantile(array,.01)),"p50":float(np.quantile(array,.5)),"p99":float(np.quantile(array,.99)),
                "fraction_below_0p0001":float(np.mean(array<1e-4)),"fraction_above_0p099":float(np.mean(array>.099))}
        residual=tf.boolean_mask(prediction-self.truth,body)
        diagnostics={"epoch":epoch+1,"components":items,"prediction_inside_true_support":stats(body),
            "prediction_outside_true_support":stats(background),"true_body_signed_bias_si":float(tf.reduce_mean(residual)),
            "true_body_mae_si":float(tf.reduce_mean(tf.abs(residual))),
            "mean_background_prediction_si":float(tf.reduce_mean(tf.boolean_mask(prediction,background))),
            "total_background_prediction_si_cells":float(tf.reduce_sum(tf.boolean_mask(prediction,background))),
            "logits":"unavailable without changing architecture/checkpoint compatibility","timestamp_utc":_utc()}
        with self.output.open("a",encoding="utf-8") as stream:stream.write(json.dumps(diagnostics)+"\n")

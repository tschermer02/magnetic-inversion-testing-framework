"""Fixed-score dual checkpoint selection for E06."""
from __future__ import annotations
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import tensorflow as tf

def _utc():return datetime.now(timezone.utc).isoformat()

def validation_metrics(model,dataset,threshold_si=.001):
    """Aggregate per-sample fixed 50/50 balanced MAE and canonical IoU."""
    balanced=[];ious=[];body_valid=background_valid=0
    for tmi,truth in dataset:
        prediction=model.inversion_model(tmi,training=False)
        for true,pred in zip(truth,prediction):
            body=true>0;background=~body
            body_values=tf.boolean_mask(tf.abs(pred-true),body)
            background_values=tf.boolean_mask(tf.abs(pred-true),background)
            body_mae=float(tf.reduce_mean(body_values)) if int(tf.size(body_values)) else 0.
            background_mae=float(tf.reduce_mean(background_values)) if int(tf.size(background_values)) else 0.
            body_valid+=int(tf.size(body_values)>0);background_valid+=int(tf.size(background_values)>0)
            balanced.append(.5*body_mae/.1+.5*background_mae/.1)
            ts=true>=threshold_si;ps=pred>=threshold_si
            intersection=float(tf.reduce_sum(tf.cast(ts&ps,tf.float32)))
            union=float(tf.reduce_sum(tf.cast(ts|ps,tf.float32)));ious.append(intersection/union if union else 1.)
    return {"validation_balanced_mae":float(np.mean(balanced)),"iou":float(np.mean(ious)),
        "sample_count":len(balanced),"body_valid_count":body_valid,"background_valid_count":background_valid,
        "empty_mask_policy":"missing body/background component contributes zero; valid counts reported"}

class DualMetricCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self,validation,output,threshold_si=.001,patience=10,resume_state=None):
        super().__init__();self.validation=validation;self.output=Path(output);self.threshold=threshold_si;self.patience=patience
        state=resume_state or {};self.best_primary=float(state.get("best_primary",np.inf));self.primary_iou=float(state.get("primary_iou",-np.inf))
        self.best_geometry=float(state.get("best_geometry",-np.inf));self.geometry_score=float(state.get("geometry_score",np.inf))
        self.primary_epoch=state.get("primary_epoch");self.geometry_epoch=state.get("geometry_epoch");self.wait=int(state.get("wait",0));self.history=list(state.get("history",[]))
    def on_epoch_end(self,epoch,logs=None):
        value=validation_metrics(self.model,self.validation,self.threshold);score=value["validation_balanced_mae"];iou=value["iou"]
        primary=score<self.best_primary-1e-12 or (abs(score-self.best_primary)<=1e-12 and iou>self.primary_iou)
        geometry=iou>self.best_geometry+1e-12 or (abs(iou-self.best_geometry)<=1e-12 and score<self.geometry_score)
        if primary:
            self.best_primary=score;self.primary_iou=iou;self.primary_epoch=epoch+1;self.wait=0
            self.model.save_weights(self.output/"primary.weights.h5")
        else:self.wait+=1
        if geometry:
            self.best_geometry=iou;self.geometry_score=score;self.geometry_epoch=epoch+1
            self.model.save_weights(self.output/"geometry.weights.h5")
        self.history.append({"epoch":epoch+1,**value,"primary_selected":primary,"geometry_selected":geometry,
            "learning_rate":float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)),"timestamp_utc":_utc()})
        state={"primary_rule":"lowest fixed 50/50 validation balanced MAE; tie higher IoU","geometry_rule":"highest validation IoU; tie lower balanced MAE",
            "best_primary":self.best_primary,"primary_iou":self.primary_iou,"primary_epoch":self.primary_epoch,
            "best_geometry":self.best_geometry,"geometry_score":self.geometry_score,"geometry_epoch":self.geometry_epoch,
            "wait":self.wait,"history":self.history}
        (self.output/"checkpoint_selection.json").write_text(json.dumps(state,indent=2),encoding="utf-8")
        print(f"\nE06 selection: balanced_MAE={score:.7g}, IoU={iou:.5f}, primary={primary}, geometry={geometry}")
        if self.wait>=self.patience:self.model.stop_training=True;print(f"E06 early stopping after {self.wait} balanced-MAE non-improving epochs")

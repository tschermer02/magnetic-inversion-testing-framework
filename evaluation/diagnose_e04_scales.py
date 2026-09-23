"""Training-only loss/gradient diagnostic for controlled E04."""
from __future__ import annotations
import csv,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import tensorflow as tf
from cnn_inversion_3d.e01_core_loss import build_e01_sensitivity_weights
from cnn_inversion_3d.e01_physics_loss import global_normalized_tmi_mse
from cnn_inversion_3d.e01_training import E01TrainingModel
from cnn_inversion_3d.e04_config import E04_CONFIG
from cnn_inversion_3d.e04_forward import E04TMIForward
from cnn_inversion_3d.model import ModelConfig,build_e01_model
from cnn_inversion_3d.train import DisabledTMIForward
from cnn_inversion_3d.train_e04 import e01_loss_config

def _norm(gradients): return float(tf.linalg.global_norm([g for g in gradients if g is not None]))
def main():
    cfg=E04_CONFIG; root=cfg.dataset_directory
    with (root/"train_manifest.csv").open(newline="",encoding="utf-8") as stream: rows=list(csv.DictReader(stream))
    rows.sort(key=lambda row:float(row["susceptibility_si"]))
    selected=[rows[int(q*(len(rows)-1))] for q in (.1,.5,.9)]
    tmis=[]; truths=[]
    for row in selected:
        with np.load(root/row["relative_path"]) as saved:
            tmis.append(saved["tmi"][...,None]/cfg.tmi_scale_nt); truths.append(saved["susceptibility"][...,None])
    tmi=tf.constant(np.stack(tmis),tf.float32); truth=tf.constant(np.stack(truths),tf.float32)
    tf.keras.utils.set_random_seed(cfg.seed); inversion=build_e01_model(ModelConfig(base_filters=cfg.base_filters))
    _,weights=build_e01_sensitivity_weights(); baseline=E01TrainingModel(inversion,weights,
        DisabledTMIForward(),tmi_scale=cfg.tmi_scale_nt,loss_config=e01_loss_config(0))
    forward=E04TMIForward()
    with tf.GradientTape(persistent=True) as tape:
        e01_loss=baseline.compute_loss_terms(tmi,truth,training=True)[-1]
        prediction=inversion(tmi,training=True); predicted_tmi=forward(prediction)
        physics=global_normalized_tmi_mse(tmi*cfg.tmi_scale_nt,predicted_tmi,tmi_scale=cfg.tmi_scale_nt)
    variables=inversion.trainable_variables
    e01_norm=_norm(tape.gradient(e01_loss,variables)); physics_norm=_norm(tape.gradient(physics,variables))
    result={"source":"fresh E01-seeded initialization","split":"training only",
        "sample_ids":[row["sample_id"] for row in selected],
        "sample_susceptibility_si":[float(row["susceptibility_si"]) for row in selected],
        "e01_raw_loss":float(e01_loss),"e01_gradient_norm":e01_norm,
        "tmi_raw_loss":float(physics),"tmi_gradient_norm":physics_norm,
        "lambda_tmi":cfg.lambda_tmi,"weighted_tmi_loss":cfg.lambda_tmi*float(physics),
        "weighted_tmi_gradient_norm":cfg.lambda_tmi*physics_norm,
        "physics_gradient_nonzero":physics_norm>0,
        "coefficient_rationale":"Conservative starting influence relative to the unchanged E01 loss and gradient; fixed before validation.",
        "timestamp_utc":datetime.now(timezone.utc).isoformat()}
    output=Path("analysis_outputs/E04_loss_scale_diagnostics.json"); output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))

if __name__=="__main__": main()

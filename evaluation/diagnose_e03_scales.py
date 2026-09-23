"""Training-only raw-loss and gradient-scale diagnostic for frozen E03 weights."""
from __future__ import annotations
import argparse,csv,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import tensorflow as tf

from cnn_inversion_3d.e03_config import RUNS
from cnn_inversion_3d.e03_forward import DifferentiableTMIForward
from cnn_inversion_3d.e03_training import (relative_balanced_susceptibility_mse_per_sample,
                                           tmi_consistency_losses_per_sample)
from cnn_inversion_3d.model import ModelConfig,build_e01_model

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,default=Path("datasets/E02"))
    parser.add_argument("--output",type=Path,default=Path("analysis_outputs/E03_scale_diagnostics.json"))
    args=parser.parse_args(); cfg=RUNS["E03_combined"]
    with (args.dataset/"train_manifest.csv").open(newline="",encoding="utf-8") as stream:
        rows=list(csv.DictReader(stream))
    bins=((1e-4,1e-3),(1e-3,1e-2),(1e-2,1e-1+1e-12)); selected=[]
    for low,high in bins:
        candidates=[r for r in rows if low<=float(r["susceptibility_si"])<high]
        selected.append(candidates[len(candidates)//2])
    tmis=[]; truths=[]
    for row in selected:
        with np.load(args.dataset/row["relative_path"]) as saved:
            tmis.append(saved["tmi"]); truths.append(saved["susceptibility"])
    tmi=tf.constant(np.stack(tmis)[...,None]/cfg.tmi_scale_nt,tf.float32)
    truth=tf.constant(np.stack(truths)[...,None],tf.float32)
    tf.keras.utils.set_random_seed(cfg.seed); model=build_e01_model(ModelConfig(base_filters=cfg.base_filters))
    forward=DifferentiableTMIForward()
    with tf.GradientTape(persistent=True) as tape:
        prediction=model(tmi,training=False)
        chi_rel=tf.reduce_mean(relative_balanced_susceptibility_mse_per_sample(
            truth,prediction,floor_si=cfg.susceptibility_floor_si))
        predicted_tmi=forward(prediction)
        d_abs_values,d_rel_values,_=tmi_consistency_losses_per_sample(tmi,predicted_tmi,
            tmi_scale_nt=cfg.tmi_scale_nt,rms_floor_nt=cfg.tmi_rms_floor_nt,
            reference_nt=cfg.tmi_reference_nt)
        d_abs=tf.reduce_mean(d_abs_values); d_rel=tf.reduce_mean(d_rel_values)
    raw={"susceptibility_relative":chi_rel,"tmi_absolute":d_abs,"tmi_relative":d_rel}
    gradients={name:float(tf.linalg.global_norm([g for g in tape.gradient(value,model.trainable_variables)
        if g is not None])) for name,value in raw.items()}
    del tape
    coefficients={"susceptibility_relative":cfg.lambda_susceptibility_relative,
                  "tmi_absolute":cfg.lambda_tmi_absolute,"tmi_relative":cfg.lambda_tmi_relative}
    report={"source":"fresh seeded initialization","split":"training only",
        "sample_ids":[r["sample_id"] for r in selected],
        "sample_susceptibility_si":[float(r["susceptibility_si"]) for r in selected],
        "raw_losses":{k:float(v) for k,v in raw.items()},"raw_gradient_norms":gradients,
        "frozen_starting_coefficients":coefficients,
        "weighted_losses":{k:float(raw[k])*coefficients[k] for k in raw},
        "p2_checkpoint_diagnostic":"not run unless checkpoint weights are locally available",
        "timestamp_utc":datetime.now(timezone.utc).isoformat()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))

if __name__=="__main__": main()

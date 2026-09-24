"""Train one frozen E03 ablation on the unchanged E02 dataset."""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import tensorflow as tf

from cnn_inversion_3d.dataset import TMI_SHAPE,build_training_datasets
from cnn_inversion_3d.e01_core_loss import build_e01_sensitivity_weights
from cnn_inversion_3d.e03_config import RUNS
from cnn_inversion_3d.e03_forward import DifferentiableTMIForward
from cnn_inversion_3d.e03_training import E03LossConfig,E03TrainingModel
from cnn_inversion_3d.model import ModelConfig,build_e01_model
from cnn_inversion_3d.train import _git_state,_sha256,build_training_callbacks

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",choices=tuple(RUNS),required=True)
    parser.add_argument("--dataset",type=Path,default=Path("datasets/E02"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--forward-chunk-size",type=int,default=128)
    args=parser.parse_args(); cfg=RUNS[args.experiment]
    tf.keras.utils.set_random_seed(cfg.seed)
    train,validation,_,counts=build_training_datasets(dataset_directory=args.dataset,
        batch_size=cfg.batch_size,tmi_scale=cfg.tmi_scale_nt,susceptibility_scale=1,
        random_seed=cfg.seed)
    _,weights=build_e01_sensitivity_weights()
    loss=E03LossConfig(lambda_depth=2,lambda_amplitude=1,lambda_tmi=0,lambda_tversky=.1,
        tversky_alpha=.7,tversky_beta=.3,occupancy_threshold=.001,occupancy_sharpness=1000,
        occupancy_mode=cfg.occupancy_mode,occupancy_tau_si=cfg.occupancy_tau_si,
        susceptibility_floor_si=cfg.susceptibility_floor_si,
        lambda_susceptibility_relative=cfg.lambda_susceptibility_relative,
        tmi_rms_floor_nt=cfg.tmi_rms_floor_nt,tmi_reference_nt=cfg.tmi_reference_nt,
        lambda_tmi_absolute=cfg.lambda_tmi_absolute,lambda_tmi_relative=cfg.lambda_tmi_relative)
    forward=DifferentiableTMIForward(args.forward_chunk_size)
    model=E03TrainingModel(build_e01_model(ModelConfig(base_filters=cfg.base_filters)),weights,
        forward,tmi_scale=cfg.tmi_scale_nt,loss_config=loss)
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.learning_rate),jit_compile=False)
    model(tf.zeros((1,*TMI_SHAPE),tf.float32),training=False)
    args.output.mkdir(parents=True,exist_ok=True)
    run_config={"experiment":cfg.to_dict(),"dataset":str(args.dataset.resolve()),"counts":counts,
        "initialization_seed":cfg.seed,"data_order_seed":cfg.seed,
        "forward_operator":"DifferentiableTMIForward; physical nT; no susceptibility threshold",
        "gradient_clipping":False,"git":_git_state(),"timestamp_utc":datetime.now(timezone.utc).isoformat()}
    for filename in ("metadata.json","train_manifest.csv","validation_manifest.csv","test_manifest.csv"):
        path=args.dataset/filename
        if path.is_file(): run_config[f"dataset_{filename}_sha256"]=_sha256(path)
    (args.output/"run_config.json").write_text(json.dumps(run_config,indent=2),encoding="utf-8")
    callbacks=build_training_callbacks(args.output,patience=cfg.early_stopping_patience,
        min_delta=cfg.early_stopping_min_delta,checkpoint_filename="best.weights.h5")
    history=model.fit(train,validation_data=validation,epochs=cfg.epochs,
        callbacks=callbacks,shuffle=False)
    checkpoint=args.output/"best.weights.h5"; model.load_weights(checkpoint)
    model.inversion_model.save(args.output/"e03.keras")
    values=history.history["val_loss"]; index=int(np.argmin(values))
    metadata={**run_config,"best_epoch":index+1,"best_validation_loss":float(values[index]),
        "checkpoint_path":str(checkpoint.resolve()),"checkpoint_sha256":_sha256(checkpoint),
        "export_reloaded_from_best_checkpoint":True,
        "warning":"Relative terms couple geometry/amplitude and can create large gradients for overestimated weak samples.",
        "timestamp_utc":datetime.now(timezone.utc).isoformat()}
    (args.output/"run_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")

if __name__=="__main__": main()

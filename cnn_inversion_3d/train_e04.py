"""Train controlled E04: unchanged E01 plus differentiable TMI loss."""
from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import tensorflow as tf
from cnn_inversion_3d.dataset import TMI_SHAPE,build_training_datasets
from cnn_inversion_3d.e01_core_loss import build_e01_sensitivity_weights
from cnn_inversion_3d.e01_training import E01LossConfig
from cnn_inversion_3d.e04_config import E04_CONFIG
from cnn_inversion_3d.e04_forward import E04TMIForward
from cnn_inversion_3d.e04_training import E04TrainingModel
from cnn_inversion_3d.model import ModelConfig,build_e01_model
from cnn_inversion_3d.train import _git_state,_sha256,build_training_callbacks

def e01_loss_config(lambda_tmi):
    return E01LossConfig(lambda_susceptibility=1,lambda_depth=2,alpha_center=1,
        lambda_sensitivity=1,lambda_amplitude=1,lambda_body_susceptibility=0,
        lambda_tmi=lambda_tmi,lambda_tversky=.1,tversky_alpha=.7,tversky_beta=.3,
        occupancy_threshold=.001,occupancy_sharpness=1000,
        occupancy_mode="legacy_threshold_sigmoid")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,default=E04_CONFIG.dataset_directory)
    parser.add_argument("--output",type=Path,default=Path("outputs/E04"))
    parser.add_argument("--forward-chunk-size",type=int,default=128)
    args=parser.parse_args(); cfg=E04_CONFIG
    tf.keras.utils.set_random_seed(cfg.seed)
    train,validation,_,counts=build_training_datasets(dataset_directory=args.dataset,
        batch_size=cfg.batch_size,tmi_scale=cfg.tmi_scale_nt,
        susceptibility_scale=cfg.susceptibility_scale_si,random_seed=cfg.seed)
    _,weights=build_e01_sensitivity_weights()
    model=E04TrainingModel(build_e01_model(ModelConfig(base_filters=cfg.base_filters)),weights,
        E04TMIForward(args.forward_chunk_size),tmi_scale=cfg.tmi_scale_nt,
        loss_config=e01_loss_config(cfg.lambda_tmi))
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.learning_rate),jit_compile=False)
    model(tf.zeros((1,*TMI_SHAPE),tf.float32),training=False)
    args.output.mkdir(parents=True,exist_ok=True)
    metadata={"experiment":cfg.to_dict(),"dataset":str(args.dataset.resolve()),"counts":counts,
        "baseline":"E01 exact active objective; only addition is lambda_tmi * normalized absolute TMI MSE",
        "forward_operator":"E04TMIForward; physical nT; unthresholded susceptibility",
        "gradient_clipping":False,"git":_git_state(),"timestamp_utc":datetime.now(timezone.utc).isoformat()}
    for filename in ("metadata.json","train_manifest.csv","validation_manifest.csv","test_manifest.csv"):
        path=args.dataset/filename
        if path.is_file(): metadata[f"dataset_{filename}_sha256"]=_sha256(path)
    (args.output/"run_config.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    callbacks=build_training_callbacks(args.output,patience=cfg.early_stopping_patience,
        min_delta=cfg.early_stopping_min_delta,checkpoint_filename="best.weights.h5")
    # Shuffling is already deterministic in build_training_datasets. Explicitly
    # disable Keras' array-level shuffle flag, which is ignored for tf.data and
    # otherwise emits a misleading warning.
    history=model.fit(train,validation_data=validation,epochs=cfg.epochs,
        callbacks=callbacks,shuffle=False)
    checkpoint=args.output/"best.weights.h5"; model.load_weights(checkpoint)
    model.inversion_model.save(args.output/"e04.keras")
    best=int(np.argmin(history.history["val_loss"]))
    metadata.update({"best_epoch":best+1,"best_validation_loss":float(history.history["val_loss"][best]),
        "checkpoint_path":str(checkpoint.resolve()),"checkpoint_sha256":_sha256(checkpoint),
        "export_reloaded_from_best_checkpoint":True,"timestamp_utc":datetime.now(timezone.utc).isoformat()})
    (args.output/"run_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")

if __name__=="__main__": main()

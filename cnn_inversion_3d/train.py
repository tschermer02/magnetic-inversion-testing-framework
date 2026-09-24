"""Train the single E01 TMI inversion architecture and loss framework."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import tensorflow as tf
from cnn_inversion_3d.dataset import TMI_SHAPE, build_training_datasets
from cnn_inversion_3d.e01_core_loss import build_e01_sensitivity_weights
from cnn_inversion_3d.e01_training import E01LossConfig, E01TrainingModel
from cnn_inversion_3d.model import ModelConfig, build_e01_model
from cnn_inversion_3d.diagnostics import CollapseDetectionCallback
from cnn_inversion_3d.e02_p2_config import RUNS

class DisabledTMIForward(tf.keras.layers.Layer):
    def call(self, susceptibility):
        batch = tf.shape(susceptibility)[0]
        return tf.zeros((batch, 81, 81, 1), susceptibility.dtype)


def build_training_callbacks(
    output_directory: Path,
    *,
    patience: int = 10,
    min_delta: float = 1.0e-5,
    checkpoint_filename: str = "e01_best.weights.h5",
) -> list[tf.keras.callbacks.Callback]:
    """Stop on stagnant validation loss and retain the best E01 weights."""
    if patience < 1:
        raise ValueError("Early-stopping patience must be at least one.")
    if min_delta < 0.0:
        raise ValueError("Early-stopping min_delta must not be negative.")
    output_directory.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=output_directory / checkpoint_filename,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(output_directory / "training_history.csv"),
        CollapseDetectionCallback(patience=2, stop_training=False),
    ]


def _git_state() -> dict[str, object]:
    def run(*args):
        try:
            result = subprocess.run(["git", *args], capture_output=True, text=True,
                                    check=False, timeout=10)
        except subprocess.TimeoutExpired:
            return None
        return result.stdout.strip() if result.returncode == 0 else None
    # Do not recursively scan large untracked datasets on network filesystems.
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": run("rev-parse", "HEAD"),
            "dirty_tracked_files": bool(status) if status is not None else None,
            "untracked_files_checked": False}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=3); parser.add_argument("--batch-size",type=int,default=2)
    parser.add_argument("--tmi-scale",type=float,default=100.0); parser.add_argument("--base-filters",type=int,default=8)
    parser.add_argument("--early-stopping-patience",type=int,default=10)
    parser.add_argument("--early-stopping-min-delta",type=float,default=1.0e-5)
    parser.add_argument("--learning-rate",type=float,default=1.0e-3)
    parser.add_argument("--occupancy-mode",choices=("legacy_threshold_sigmoid","geological_exponential"),
        default="legacy_threshold_sigmoid")
    parser.add_argument("--occupancy-tau-si",type=float,default=1.0e-4 / __import__('math').log(100.0))
    parser.add_argument("--experiment",choices=tuple(RUNS),help="Apply a locked E02 Priority-2 run configuration.")
    args=parser.parse_args()
    run = RUNS.get(args.experiment) if args.experiment else None
    if run:
        args.learning_rate=run.learning_rate; args.occupancy_mode=run.occupancy_mode
        args.occupancy_tau_si=run.occupancy_tau_si; args.batch_size=run.batch_size
        args.epochs=run.epochs; args.base_filters=run.base_filters; args.tmi_scale=run.tmi_scale_nt
        args.early_stopping_patience=run.early_stopping_patience
        args.early_stopping_min_delta=run.early_stopping_min_delta
    tf.keras.utils.set_random_seed(run.initialization_seed if run else 20260727)
    train,validation,_,_=build_training_datasets(dataset_directory=args.dataset,batch_size=args.batch_size,
        tmi_scale=args.tmi_scale,susceptibility_scale=1.0,random_seed=20260727)
    _,weights=build_e01_sensitivity_weights()
    config=E01LossConfig(lambda_depth=2.0,lambda_amplitude=1.0,lambda_tmi=0.0,
        lambda_tversky=0.1,tversky_alpha=0.7,tversky_beta=0.3,
        occupancy_threshold=0.001,occupancy_sharpness=1000.0,
        occupancy_mode=args.occupancy_mode,occupancy_tau_si=args.occupancy_tau_si)
    model=E01TrainingModel(build_e01_model(ModelConfig(base_filters=args.base_filters)),weights,
        DisabledTMIForward(),tmi_scale=args.tmi_scale,loss_config=config)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),jit_compile=False)
    # Custom train_step invokes the inner CNN directly. Call the wrapper once
    # so ModelCheckpoint can save it after the first validation epoch.
    model(tf.zeros((1, *TMI_SHAPE), dtype=tf.float32), training=False)
    checkpoint_filename="best.weights.h5" if run else "e01_best.weights.h5"
    callbacks=build_training_callbacks(args.output,patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,checkpoint_filename=checkpoint_filename)
    args.output.mkdir(parents=True,exist_ok=True)
    initial_metadata={"experiment":args.experiment or "legacy_custom",
        "configuration":run.to_dict() if run else vars(args),
        "initialization_seed":run.initialization_seed if run else 20260727,
        "data_order_seed":run.data_order_seed if run else 20260727,
        "dataset":str(args.dataset.resolve()),"git":_git_state(),
        "timestamp_utc":datetime.now(timezone.utc).isoformat()}
    for filename in ("metadata.json","train_manifest.csv","validation_manifest.csv","test_manifest.csv"):
        path=args.dataset/filename
        if path.is_file(): initial_metadata[f"dataset_{filename}_sha256"]=_sha256(path)
    (args.output/"run_config.json").write_text(
        json.dumps(initial_metadata,indent=2,default=str),encoding="utf-8")
    history=model.fit(train,validation_data=validation,epochs=args.epochs,
        callbacks=callbacks,shuffle=False)
    checkpoint=args.output/checkpoint_filename
    model.load_weights(checkpoint)
    model.inversion_model.save(args.output/"e01.keras")
    losses=history.history.get("val_loss",[]); best_index=int(__import__('numpy').argmin(losses)) if losses else None
    metadata={"experiment":args.experiment or "legacy_custom", "configuration":run.to_dict() if run else vars(args),
        "occupancy":{"mode":args.occupancy_mode,"tau_si":args.occupancy_tau_si,
            "mapping":"-expm1(-max(prediction,0)/tau)" if args.occupancy_mode=="geological_exponential" else "legacy floor-corrected threshold sigmoid",
            "true_support":"true_susceptibility > 0" if args.occupancy_mode=="geological_exponential" else "true_susceptibility >= 0.001 SI"},
        "gradient_clipping":False,"best_epoch":best_index+1 if best_index is not None else None,
        "best_validation_loss":float(losses[best_index]) if best_index is not None else None,
        "checkpoint_path":str(checkpoint.resolve()),"checkpoint_sha256":_sha256(checkpoint),
        "export_reloaded_from_best_checkpoint":True,"dataset":str(args.dataset.resolve()),
        "collapse_detection":next(callback.result() for callback in callbacks
            if isinstance(callback,CollapseDetectionCallback)),
        "git":_git_state(),"timestamp_utc":datetime.now(timezone.utc).isoformat()}
    (args.output/"run_metadata.json").write_text(json.dumps(metadata,indent=2,default=str),encoding="utf-8")

if __name__=="__main__": main()

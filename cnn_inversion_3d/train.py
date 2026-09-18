"""Train the single E01 TMI inversion architecture and loss framework."""
from __future__ import annotations
import argparse
from pathlib import Path
import tensorflow as tf
from cnn_inversion_3d.dataset import build_training_datasets
from cnn_inversion_3d.e01_core_loss import build_e01_sensitivity_weights
from cnn_inversion_3d.e01_training import E01LossConfig, E01TrainingModel
from cnn_inversion_3d.model import ModelConfig, build_e01_model

class DisabledTMIForward(tf.keras.layers.Layer):
    def call(self, susceptibility):
        batch = tf.shape(susceptibility)[0]
        return tf.zeros((batch, 81, 81, 1), susceptibility.dtype)


def build_training_callbacks(
    output_directory: Path,
    *,
    patience: int = 10,
    min_delta: float = 1.0e-5,
) -> list[tf.keras.callbacks.Callback]:
    """Stop on stagnant validation loss and retain the best E01 weights."""
    if patience < 1:
        raise ValueError("Early-stopping patience must be at least one.")
    if min_delta < 0.0:
        raise ValueError("Early-stopping min_delta must not be negative.")
    output_directory.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=output_directory / "e01_best.weights.h5",
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
    ]

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=3); parser.add_argument("--batch-size",type=int,default=2)
    parser.add_argument("--tmi-scale",type=float,default=100.0); parser.add_argument("--base-filters",type=int,default=8)
    parser.add_argument("--early-stopping-patience",type=int,default=10)
    parser.add_argument("--early-stopping-min-delta",type=float,default=1.0e-5)
    args=parser.parse_args()
    train,validation,_,_=build_training_datasets(dataset_directory=args.dataset,batch_size=args.batch_size,
        tmi_scale=args.tmi_scale,susceptibility_scale=1.0,random_seed=20260727)
    _,weights=build_e01_sensitivity_weights()
    config=E01LossConfig(lambda_depth=2.0,lambda_amplitude=1.0,lambda_tmi=0.0,
        lambda_tversky=0.1,tversky_alpha=0.7,tversky_beta=0.3,
        occupancy_threshold=0.001,occupancy_sharpness=1000.0)
    model=E01TrainingModel(build_e01_model(ModelConfig(base_filters=args.base_filters)),weights,
        DisabledTMIForward(),tmi_scale=args.tmi_scale,loss_config=config)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),jit_compile=False)
    callbacks=build_training_callbacks(args.output,patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta)
    model.fit(train,validation_data=validation,epochs=args.epochs,callbacks=callbacks)
    model.inversion_model.save(args.output/"e01.keras")

if __name__=="__main__": main()

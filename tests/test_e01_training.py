import numpy as np
import tensorflow as tf

from cnn_inversion_3d.e01_core_loss import build_e01_sensitivity_weights
from cnn_inversion_3d.e01_training import E01LossConfig, E01TrainingModel
from cnn_inversion_3d.model import ModelConfig, build_e01_model
from cnn_inversion_3d.evaluate import save_susceptibility_comparison
from cnn_inversion_3d.dataset import TMI_SHAPE
from cnn_inversion_3d.train import DisabledTMIForward, build_training_callbacks


def test_e01_train_step_accepts_tmi_batch():
    _, weights = build_e01_sensitivity_weights()
    inversion = build_e01_model(ModelConfig(base_filters=1))
    forward = tf.keras.layers.Lambda(
        lambda susceptibility: tf.zeros(
            (tf.shape(susceptibility)[0], 81, 81, 1),
            dtype=susceptibility.dtype,
        )
    )
    model = E01TrainingModel(
        inversion,
        weights,
        forward,
        tmi_scale=1.0,
        loss_config=E01LossConfig(
            lambda_tmi=0.0,
            lambda_tversky=0.1,
            occupancy_threshold=0.001,
            occupancy_sharpness=1000.0,
        ),
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(1.0e-3))
    tmi = tf.zeros((1, 81, 81, 1), dtype=tf.float32)
    susceptibility = np.zeros((1, 24, 64, 64, 1), dtype=np.float32)
    susceptibility[:, 2:4, 30:34, 30:34, :] = 0.02
    logs = model.train_step((tmi, tf.constant(susceptibility)))
    assert "loss" in logs
    assert np.isfinite(float(logs["loss"]))


def test_e01_output_is_limited_to_generated_susceptibility_range():
    model = build_e01_model(ModelConfig(base_filters=1))
    prediction = model(tf.zeros((1, 81, 81, 1)), training=False).numpy()
    assert prediction.min() >= 0.0
    assert prediction.max() <= 0.1


def test_susceptibility_comparison_figure_is_written(tmp_path):
    truth = np.zeros((24, 64, 64), dtype=np.float32)
    truth[5:9, 28:34, 30:38] = 0.02
    output = tmp_path / "comparison.png"
    save_susceptibility_comparison(truth, truth * 0.9, output)
    assert output.exists() and output.stat().st_size > 0


def test_training_callbacks_restore_best_validation_weights(tmp_path):
    callbacks = build_training_callbacks(tmp_path, patience=7, min_delta=2.0e-5)
    early_stopping = next(
        callback for callback in callbacks
        if isinstance(callback, tf.keras.callbacks.EarlyStopping)
    )
    checkpoint = next(
        callback for callback in callbacks
        if isinstance(callback, tf.keras.callbacks.ModelCheckpoint)
    )
    assert early_stopping.monitor == "val_loss"
    assert early_stopping.patience == 7
    assert early_stopping.min_delta == 2.0e-5
    assert early_stopping.restore_best_weights is True
    assert checkpoint.save_best_only is True


def test_built_training_wrapper_can_save_checkpoint(tmp_path):
    _, weights = build_e01_sensitivity_weights()
    model = E01TrainingModel(
        build_e01_model(ModelConfig(base_filters=1)),
        weights,
        DisabledTMIForward(),
        tmi_scale=100.0,
        loss_config=E01LossConfig(),
    )
    model(tf.zeros((1, *TMI_SHAPE), dtype=tf.float32), training=False)
    checkpoint = tmp_path / "checkpoint.weights.h5"
    model.save_weights(checkpoint)
    assert checkpoint.is_file()

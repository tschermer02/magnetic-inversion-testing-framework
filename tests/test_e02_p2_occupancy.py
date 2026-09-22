"""Tests for Priority-2 E02 geological occupancy supervision."""
import math

import numpy as np
import tensorflow as tf

from cnn_inversion_3d.e01_training import (
    exponential_soft_occupancy, geological_support_mask,
    soft_tversky_loss_per_sample,
)
from cnn_inversion_3d.model import ModelConfig, build_e01_model

TAU = 1.0e-4 / math.log(100.0)


def _volume(values):
    return tf.reshape(tf.constant(values, tf.float32), (1, 1, 1, len(values), 1))


def test_geological_support_includes_all_e02_body_amplitudes():
    mask = geological_support_mask(_volume([0.0, 1e-4, 1e-3, 1e-1])).numpy().ravel()
    np.testing.assert_array_equal(mask, [0, 1, 1, 1])


def test_exponential_occupancy_values_transition_and_gradient():
    values = tf.Variable([0.0, TAU, 1e-4, 1e-3, 1e-1], dtype=tf.float32)
    with tf.GradientTape() as tape:
        occupied = exponential_soft_occupancy(values, tau_si=TAU)
    gradient = tape.gradient(occupied, values).numpy()
    result = occupied.numpy()
    assert result[0] == 0.0
    assert np.isclose(result[1], 1.0 - math.exp(-1.0), rtol=1e-6)
    assert np.isclose(result[2], 0.99, rtol=1e-6)
    assert result[3] > 0.999999
    assert result[4] == 1.0
    assert np.all(np.isfinite(result)) and np.all(np.isfinite(gradient))
    assert gradient[0] > gradient[1] > gradient[2] > 0


def test_weak_body_overlap_missing_false_positive_and_empty_behavior():
    truth = _volume([0.0, 1e-4, 0.0])
    accurate = soft_tversky_loss_per_sample(truth, truth,
        occupancy_mode="geological_exponential", occupancy_tau_si=TAU).numpy()[0]
    missing = soft_tversky_loss_per_sample(truth, tf.zeros_like(truth),
        occupancy_mode="geological_exponential", occupancy_tau_si=TAU).numpy()[0]
    false_positive = soft_tversky_loss_per_sample(truth, _volume([1e-4, 1e-4, 0.0]),
        occupancy_mode="geological_exponential", occupancy_tau_si=TAU).numpy()[0]
    empty = soft_tversky_loss_per_sample(tf.zeros_like(truth), tf.zeros_like(truth),
        occupancy_mode="geological_exponential", occupancy_tau_si=TAU).numpy()[0]
    assert accurate < 0.01
    assert missing > 0.99
    assert false_positive > accurate
    assert empty == 0.0


def test_gradient_is_finite_through_actual_model_output_path():
    model = build_e01_model(ModelConfig(base_filters=1))
    inputs = tf.zeros((1, 81, 81, 1), tf.float32)
    truth = tf.zeros((1, 24, 64, 64, 1), tf.float32)
    truth = tf.tensor_scatter_nd_update(truth, [[0, 4, 30, 30, 0]], [1e-4])
    with tf.GradientTape() as tape:
        prediction = model(inputs, training=True)
        loss = tf.reduce_mean(soft_tversky_loss_per_sample(truth, prediction,
            occupancy_mode="geological_exponential", occupancy_tau_si=TAU))
    gradients = [value for value in tape.gradient(loss, model.trainable_variables) if value is not None]
    assert gradients
    assert all(np.all(np.isfinite(value.numpy())) for value in gradients)


def test_legacy_mapping_is_unchanged_and_excludes_subthreshold_truth():
    truth = _volume([0.0, 1e-4, 1e-3, 0.1])
    prediction = _volume([0.0, 2e-4, 2e-3, 0.09])
    actual = soft_tversky_loss_per_sample(truth, prediction, threshold=0.001,
        sharpness=1000.0, alpha=0.7, beta=0.3,
        occupancy_mode="legacy_threshold_sigmoid").numpy()[0]
    target = (truth.numpy().ravel() >= 0.001).astype(np.float32)
    values = prediction.numpy().ravel()
    raw = 1.0 / (1.0 + np.exp(-1000.0 * (values - 0.001)))
    floor = 1.0 / (1.0 + np.exp(1.0))
    occupied = np.clip((raw - floor) / (1.0 - floor), 0.0, 1.0)
    tp = np.sum(target * occupied); fp = np.sum((1-target)*occupied); fn = np.sum(target*(1-occupied))
    expected = 1.0 - (tp + 1e-8) / (tp + 0.7*fp + 0.3*fn + 1e-8)
    assert np.isclose(actual, expected, rtol=1e-6)
    assert target[1] == 0 and target[2] == 1

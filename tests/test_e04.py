import numpy as np
import tensorflow as tf
from pathlib import Path
from cnn_inversion_3d.e01_core_loss import build_e01_sensitivity_weights
from cnn_inversion_3d.e01_physics_loss import global_normalized_tmi_mse
from cnn_inversion_3d.e01_training import E01TrainingModel
from cnn_inversion_3d.e04_config import E04_CONFIG
from cnn_inversion_3d.e04_forward import E04TMIForward
from cnn_inversion_3d.e04_training import E04TrainingModel
from cnn_inversion_3d.model import ModelConfig,build_e01_model
from cnn_inversion_3d.train import DisabledTMIForward
from cnn_inversion_3d.train_e04 import e01_loss_config

def _sample():
    path=Path("datasets/E01_soft_tversky_full/samples/sample_000000.npz")
    if not path.is_file():
        import pytest; pytest.skip("E01 full dataset unavailable")
    with np.load(path) as saved: return saved["tmi"],saved["susceptibility"]

def test_e04_locks_e01_dataset_architecture_and_normalization():
    assert E04_CONFIG.dataset_directory==Path("datasets/E01_soft_tversky_full")
    assert E04_CONFIG.tmi_scale_nt==100 and E04_CONFIG.susceptibility_scale_si==1
    model=build_e01_model(ModelConfig(base_filters=E04_CONFIG.base_filters))
    assert model.name=="e01_tmi_inversion" and model.output_shape==(None,24,64,64,1)
    assert model.get_layer("recovered_susceptibility_si").scale==.1

def test_forward_shape_finite_and_reproduces_e01_tmi():
    observed,truth=_sample(); predicted=E04TMIForward()(truth[None,...,None]).numpy()
    assert predicted.shape==(1,81,81,1) and np.all(np.isfinite(predicted))
    np.testing.assert_allclose(predicted[0,...,0],observed,rtol=2e-4,atol=4e-5)

def test_tmi_loss_zero_for_exact_and_increases_when_perturbed():
    observed,truth=_sample(); operator=E04TMIForward()
    exact=operator(truth[None,...,None]); changed=operator((truth*1.1)[None,...,None])
    exact_loss=float(global_normalized_tmi_mse(observed[None,...,None],exact,tmi_scale=100))
    changed_loss=float(global_normalized_tmi_mse(observed[None,...,None],changed,tmi_scale=100))
    assert exact_loss<1e-12 and changed_loss>exact_loss

def test_e04_total_is_e01_plus_weighted_tmi_and_zero_lambda_reproduces_e01():
    observed,truth=_sample(); tmi=tf.constant(observed[None,...,None]/100,tf.float32)
    target=tf.constant(truth[None,...,None],tf.float32); tf.keras.utils.set_random_seed(22)
    inversion=build_e01_model(ModelConfig(base_filters=1)); _,weights=build_e01_sensitivity_weights()
    e01=E01TrainingModel(inversion,weights,DisabledTMIForward(),tmi_scale=100,
        loss_config=e01_loss_config(0)); base=e01.compute_loss_terms(tmi,target,training=False)[-1]
    cfg=e01_loss_config(1e-3); e04=E04TrainingModel(inversion,weights,E04TMIForward(),
        tmi_scale=100,loss_config=cfg); terms=e04.compute_loss_terms(tmi,target,training=False)
    np.testing.assert_allclose(float(terms[-1]),float(base+cfg.lambda_tmi*terms[8]),rtol=2e-6)
    assert float(base)==float(e01.compute_loss_terms(tmi,target,training=False)[-1])

def test_physics_loss_has_finite_nonzero_cnn_gradients():
    observed,_=_sample(); tmi=tf.constant(observed[None,...,None]/100,tf.float32)
    tf.keras.utils.set_random_seed(33); model=build_e01_model(ModelConfig(base_filters=1)); operator=E04TMIForward()
    with tf.GradientTape() as tape:
        predicted=operator(model(tmi,training=True))
        loss=global_normalized_tmi_mse(tmi*100,predicted,tmi_scale=100)
    gradients=[g for g in tape.gradient(loss,model.trainable_variables) if g is not None]
    norm=float(tf.linalg.global_norm(gradients))
    assert np.isfinite(float(loss)) and np.isfinite(norm) and norm>0

def test_e04_training_step_is_finite_and_logs_physics_metrics():
    observed,truth=_sample(); tmi=tf.constant(observed[None,...,None]/100,tf.float32)
    target=tf.constant(truth[None,...,None],tf.float32); tf.keras.utils.set_random_seed(44)
    _,weights=build_e01_sensitivity_weights()
    model=E04TrainingModel(build_e01_model(ModelConfig(base_filters=1)),weights,E04TMIForward(),
        tmi_scale=100,loss_config=e01_loss_config(E04_CONFIG.lambda_tmi))
    model.compile(optimizer=tf.keras.optimizers.Adam(E04_CONFIG.learning_rate),jit_compile=False)
    metrics=model.train_step((tmi,target))
    for name in ("loss","tmi_loss","weighted_tmi_loss","tmi_rmse","tmi_mae_nt",
                 "tmi_correlation","global_gradient_norm"):
        assert name in metrics and np.isfinite(float(metrics[name]))
    assert float(metrics["global_gradient_norm"])>0

import numpy as np
import tensorflow as tf
from pathlib import Path

from cnn_inversion_3d.e03_forward import DifferentiableTMIForward
from cnn_inversion_3d.e03_training import (relative_balanced_susceptibility_mse_per_sample,
                                           tmi_consistency_losses_per_sample,
                                           E03LossConfig,E03TrainingModel)
from cnn_inversion_3d.e01_core_loss import build_e01_sensitivity_weights
from cnn_inversion_3d.e01_training import E01LossConfig,E01TrainingModel
from cnn_inversion_3d.model import ModelConfig,build_e01_model
from cnn_inversion_3d.train import DisabledTMIForward
from forward_modeling.forward_model import TMIForwardModel,make_tensor_grid

def _v(body,background=0.0):
    return tf.constant([[[[[body],[background]]]]],tf.float32)

def test_fractional_errors_are_comparable_above_floor_and_exact_is_zero():
    low=relative_balanced_susceptibility_mse_per_sample(_v(1e-3),_v(1.1e-3)).numpy()[0]
    high=relative_balanced_susceptibility_mse_per_sample(_v(1e-1),_v(1.1e-1)).numpy()[0]
    exact=relative_balanced_susceptibility_mse_per_sample(_v(1e-3),_v(1e-3)).numpy()[0]
    assert np.isclose(low,high,rtol=2e-5) and exact==0

def test_background_leakage_and_empty_body_are_finite():
    clean=relative_balanced_susceptibility_mse_per_sample(_v(1e-3),_v(1e-3)).numpy()[0]
    leak=relative_balanced_susceptibility_mse_per_sample(_v(1e-3),_v(1e-3,1e-3)).numpy()[0]
    empty=relative_balanced_susceptibility_mse_per_sample(tf.zeros((1,1,1,2,1)),_v(0,1e-4)).numpy()[0]
    assert leak>clean and np.isfinite(empty)

def test_zero_weak_tmi_losses_are_finite():
    observed=tf.zeros((1,2,2,1)); predicted=tf.ones_like(observed)*1e-5
    absolute,relative,rms=tmi_consistency_losses_per_sample(observed,predicted,
        tmi_scale_nt=100,rms_floor_nt=.03)
    assert all(np.all(np.isfinite(x.numpy())) for x in (absolute,relative,rms))

def _small_operators():
    grid=make_tensor_grid([0,20,0,20,0,20],[10,10],10)
    receivers=np.array([[0.,0.,-10.],[20.,20.,-10.]])
    numpy_op=TMIForwardModel(grid,receivers,50000,75,25,0,receiver_chunk_size=1)
    tf_op=DifferentiableTMIForward(1,grid=grid,receiver_xyz=receivers,
        field_strength_nt=50000,inclination_deg=75,declination_deg=25,azimuth_deg=0)
    return grid,numpy_op,tf_op

def test_differentiable_forward_matches_numpy_and_scales_linearly():
    grid,numpy_op,tf_op=_small_operators()
    zyx=np.arange(8,dtype=np.float32).reshape(2,2,2)*1e-3
    expected=numpy_op.predict(zyx.transpose(2,1,0))
    actual=tf_op(zyx[None,...,None]).numpy().ravel()
    np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=1e-6)
    np.testing.assert_allclose(tf_op((2*zyx)[None,...,None]).numpy().ravel(),2*actual,rtol=1e-6)

def test_forward_gradient_matches_directional_finite_difference():
    _,_,operator=_small_operators()
    values=tf.Variable(np.full((1,2,2,2,1),.002,np.float32)); direction=tf.constant(
        np.linspace(-1,1,8,dtype=np.float32).reshape(1,2,2,2,1))
    with tf.GradientTape() as tape: objective=tf.reduce_sum(tf.square(operator(values)))
    analytic=float(tf.reduce_sum(tape.gradient(objective,values)*direction))
    step=1e-5
    plus=float(tf.reduce_sum(tf.square(operator(values+step*direction))))
    minus=float(tf.reduce_sum(tf.square(operator(values-step*direction))))
    finite=(plus-minus)/(2*step)
    assert np.isclose(analytic,finite,rtol=2e-3,atol=1e-3)

def test_full_forward_reproduces_stored_e02_tmi():
    path=Path("datasets/E02/samples/sample_000000.npz")
    if not path.is_file():
        import pytest; pytest.skip("E02 dataset is not available")
    with np.load(path) as saved:
        truth=saved["susceptibility"]; expected=saved["tmi"]
    actual=DifferentiableTMIForward()(truth[None,...,None]).numpy()[0,...,0]
    np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=2e-4)

def _p2_kwargs():
    return dict(lambda_depth=2,lambda_amplitude=1,lambda_tmi=0,lambda_tversky=.1,
        tversky_alpha=.7,tversky_beta=.3,occupancy_threshold=.001,
        occupancy_sharpness=1000,occupancy_mode="geological_exponential",
        occupancy_tau_si=1e-4/np.log(100))

def test_disabling_all_e03_terms_exactly_reproduces_p2_objective():
    tf.keras.utils.set_random_seed(123)
    inversion=build_e01_model(ModelConfig(base_filters=1)); _,weights=build_e01_sensitivity_weights()
    p2=E01TrainingModel(inversion,weights,DisabledTMIForward(),tmi_scale=100,
        loss_config=E01LossConfig(**_p2_kwargs()))
    e03=E03TrainingModel(inversion,weights,DisabledTMIForward(),tmi_scale=100,
        loss_config=E03LossConfig(**_p2_kwargs()))
    tmi=tf.random.normal((1,81,81,1)); truth=tf.zeros((1,24,64,64,1))
    truth=tf.tensor_scatter_nd_update(truth,[[0,4,20,20,0]],[.003])
    assert float(p2.compute_loss_terms(tmi,truth,training=False)[-1]) == float(
        e03.compute_loss_terms(tmi,truth,training=False)[-1])

def test_wrapper_checkpoint_reload_reproduces_predictions(tmp_path):
    tf.keras.utils.set_random_seed(456); _,weights=build_e01_sensitivity_weights()
    config=E03LossConfig(**_p2_kwargs())
    first=E03TrainingModel(build_e01_model(ModelConfig(base_filters=1)),weights,
        DisabledTMIForward(),tmi_scale=100,loss_config=config)
    sample=tf.random.normal((1,81,81,1)); expected=first(sample,training=False).numpy()
    checkpoint=tmp_path/"best.weights.h5"; first.save_weights(checkpoint)
    second=E03TrainingModel(build_e01_model(ModelConfig(base_filters=1)),weights,
        DisabledTMIForward(),tmi_scale=100,loss_config=config)
    second(tf.zeros_like(sample),training=False); second.load_weights(checkpoint)
    np.testing.assert_array_equal(second(sample,training=False).numpy(),expected)

def test_combined_e03_single_training_step_has_finite_loss_and_gradients():
    path=Path("datasets/E02/samples/sample_000000.npz")
    if not path.is_file():
        import pytest; pytest.skip("E02 dataset is not available")
    with np.load(path) as saved:
        tmi=tf.constant(saved["tmi"][None,...,None]/100,tf.float32)
        truth=tf.constant(saved["susceptibility"][None,...,None],tf.float32)
    tf.keras.utils.set_random_seed(789); _,weights=build_e01_sensitivity_weights()
    config=E03LossConfig(**_p2_kwargs(),susceptibility_floor_si=1e-4,
        lambda_susceptibility_relative=1e-6,tmi_rms_floor_nt=.03296747710542477,
        tmi_reference_nt=100,lambda_tmi_absolute=1e-4,lambda_tmi_relative=1e-9)
    model=E03TrainingModel(build_e01_model(ModelConfig(base_filters=1)),weights,
        DifferentiableTMIForward(),tmi_scale=100,loss_config=config)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),jit_compile=False)
    result=model.train_step((tmi,truth))
    assert all(np.isfinite(float(value)) for value in result.values())
    assert float(result["global_gradient_norm"])>0

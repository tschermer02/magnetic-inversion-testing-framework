import numpy as np
import tensorflow as tf

from e01_magnetic.config import E01Config, MagneticSurveyConfig
from e01_magnetic.dataset import build_diagnostic_dataset
from e01_magnetic.losses import TMIPhysicsLoss, tmi_loss_gradient_test
from e01_magnetic.model import build_e01_model, count_trainable_parameters
from e01_magnetic.physics import MagneticTensorGrid, MagneticTMIModel
from forward_modeling.forward_model import TMIForwardModel, TensorGrid, make_tensor_grid, inducing_field_direction


def test_matlab_example_in_tmi_forward():
    grid = make_tensor_grid(bounds=[0.0, 100.0, 0.0, 100.0, 0.0, 30.0], xy_size=[10.0, 10.0], dz=5.0)
    receivers = np.zeros((3, 3), dtype=np.float64)
    receivers[:, 0] = np.linspace(10.0, 90.0, 3)
    receivers[:, 1] = np.linspace(10.0, 90.0, 3)
    receivers[:, 2] = -10.0
    model = TMIForwardModel(grid, receivers, field_strength_nt=50000.0, inclination_deg=75.0, declination_deg=25.0, azimuth_deg=0.0)
    response = model.predict(np.ones(grid.n_cells, dtype=np.float64))
    assert response.shape == (3,)
    assert np.all(np.isfinite(response))
    assert not np.allclose(response, 0.0)


def test_linearity_of_forward_model():
    grid = make_tensor_grid(bounds=[0.0, 40.0, 0.0, 40.0, 0.0, 20.0], xy_size=[10.0, 10.0], dz=5.0)
    receivers = np.zeros((5, 3), dtype=np.float64)
    receivers[:, 0] = np.linspace(5.0, 35.0, 5)
    receivers[:, 1] = 20.0
    receivers[:, 2] = -5.0
    model = TMIForwardModel(grid, receivers, field_strength_nt=50000.0, inclination_deg=75.0, declination_deg=25.0, azimuth_deg=0.0)
    k1 = np.ones(grid.n_cells, dtype=np.float64)
    k2 = np.linspace(0.1, 1.0, grid.n_cells, dtype=np.float64)
    a, b = 2.5, -1.3
    lhs = model.predict(a * k1 + b * k2)
    rhs = a * model.predict(k1) + b * model.predict(k2)
    assert np.allclose(lhs, rhs, rtol=1e-10, atol=1e-10)


def test_predict_matches_sensitivity_operator():
    grid = make_tensor_grid(bounds=[0.0, 40.0, 0.0, 40.0, 0.0, 20.0], xy_size=[10.0, 10.0], dz=5.0)
    receivers = np.zeros((5, 3), dtype=np.float64)
    receivers[:, 0] = np.linspace(5.0, 35.0, 5)
    receivers[:, 1] = 20.0
    receivers[:, 2] = -5.0
    forward = MagneticTMIModel(grid, receivers, field_strength_nt=50000.0, inclination_deg=75.0, declination_deg=25.0, azimuth_deg=0.0)
    model = np.linspace(0.05, 0.5, grid.n_cells, dtype=np.float64)
    pred = forward.predict(model)
    sens = forward.sensitivity_matrix_chunked() 
    assert sens.shape[1] == grid.n_cells
    assert np.allclose(pred, sens @ model, rtol=1e-10, atol=1e-10)


def test_batched_prediction_matches_individual():
    grid = make_tensor_grid(bounds=[0.0, 40.0, 0.0, 40.0, 0.0, 20.0], xy_size=[10.0, 10.0], dz=5.0)
    receivers = np.zeros((4, 3), dtype=np.float64)
    receivers[:, 0] = [10.0, 20.0, 30.0, 20.0]
    receivers[:, 1] = [10.0, 20.0, 10.0, 30.0]
    receivers[:, 2] = -5.0
    forward = MagneticTMIModel(grid, receivers, field_strength_nt=50000.0, inclination_deg=75.0, declination_deg=25.0, azimuth_deg=0.0)
    models = np.vstack([
        np.linspace(0.1, 0.8, grid.n_cells, dtype=np.float64),
        np.linspace(0.2, 0.9, grid.n_cells, dtype=np.float64),
    ])
    expected = np.stack([forward.predict(model) for model in models], axis=0)
    actual = forward.predict_many(models)
    assert np.allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_fortran_ordering_round_trip():
    dims = (4, 3, 2)
    model = np.arange(np.prod(dims), dtype=np.float64).reshape(dims, order="C")
    flat_f = model.ravel(order="F")
    roundtrip = flat_f.reshape(dims, order="F")
    assert roundtrip.shape == dims
    assert np.allclose(roundtrip, model)
    assert not np.allclose(flat_f.reshape(dims, order="C"), model)


def test_receiver_map_reshape():
    tmi_map = np.arange(81 * 81, dtype=np.float64).reshape(81, 81)
    flat = tmi_map.reshape(-1)
    reshaped = flat.reshape((81, 81), order="C")
    assert reshaped.shape == (81, 81)
    assert np.allclose(reshaped, tmi_map)


def test_no_receiver_coincides_with_cell_center():
    grid = make_tensor_grid(bounds=[0.0, 20.0, 0.0, 20.0, 0.0, 10.0], xy_size=[10.0, 10.0], dz=5.0)
    bad_receiver = np.array([[5.0, 5.0, 2.5]], dtype=np.float64)
    try:
        TMIForwardModel(grid, bad_receiver, field_strength_nt=50000.0, inclination_deg=75.0, declination_deg=25.0, azimuth_deg=0.0)
        raise AssertionError('Expected ValueError for receiver on cell center.')
    except ValueError:
        pass


def test_tmi_outputs_are_finite():
    grid = make_tensor_grid(bounds=[0.0, 100.0, 0.0, 100.0, 0.0, 30.0], xy_size=[10.0, 10.0], dz=5.0)
    receivers = np.random.default_rng(0).random((10, 3), dtype=np.float64)
    receivers[:, 0] = 10.0 + 90.0 * receivers[:, 0]
    receivers[:, 1] = 10.0 + 90.0 * receivers[:, 1]
    receivers[:, 2] = -10.0
    model = TMIForwardModel(grid, receivers, field_strength_nt=50000.0, inclination_deg=75.0, declination_deg=25.0, azimuth_deg=0.0)
    response = model.predict(np.ones(grid.n_cells))
    assert np.isfinite(response).all()


def test_tmi_loss_gradient_propagates():
    grid = make_tensor_grid(bounds=[0.0, 60.0, 0.0, 60.0, 0.0, 30.0], xy_size=[10.0, 10.0], dz=5.0)
    config = E01Config()
    survey = config.survey
    forward = MagneticTMIModel(grid, survey.receiver_xyz, field_strength_nt=survey.field_strength_nt, inclination_deg=survey.inclination_deg, declination_deg=survey.declination_deg, azimuth_deg=survey.azimuth_deg)
    true_tmi = forward.predict(np.linspace(0.05, 0.5, grid.n_cells, dtype=np.float64))
    pred = tf.Variable(np.linspace(0.02, 0.4, grid.n_cells, dtype=np.float64), dtype=tf.float64)
    tangents = tf.ones_like(pred)
    with tf.autodiff.ForwardAccumulator((pred,), (tangents,)) as acc:
        loss = TMIPhysicsLoss(forward_operator=forward, scale=1.0)(true_tmi, pred)
        grad = acc.jvp(loss)
    assert np.isfinite(grad.numpy()).all()
    assert np.linalg.norm(grad.numpy()) > 0.0


def test_deterministic_dataset_generation():
    cfg = E01Config(seed=123)
    ds1 = build_diagnostic_dataset(cfg, sample_count=3)
    ds2 = build_diagnostic_dataset(cfg, sample_count=3)
    assert ds1[0][0].shape == ds2[0][0].shape
    assert np.allclose(ds1[0][0], ds2[0][0])


def test_model_parameter_count_matches_reference():
    model = build_e01_model()
    parameters = count_trainable_parameters(model)
    assert parameters == 190592


def test_survey_direction_matches_matlab_convention():
    direction = inducing_field_direction(75.0, 25.0, 0.0)
    assert direction.shape == (3,)
    assert np.all(np.isfinite(direction))
    assert direction[2] > 0.0

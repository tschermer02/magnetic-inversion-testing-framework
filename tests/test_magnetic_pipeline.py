import numpy as np
from cnn_inversion_3d.dataset import load_npz_sample, TMI_SHAPE, SUSCEPTIBILITY_SHAPE
from dataset_generation.generate_single_plane_dataset import MODEL_SHAPE_ZYX, sample_susceptibility
from evaluation.tmi_metrics import calculate_tmi_fit_metrics, calculate_susceptibility_metrics

def test_sample_is_susceptibility_in_si():
    model, metadata = sample_susceptibility(np.random.default_rng(4))
    assert model.shape == MODEL_SHAPE_ZYX
    assert 0.001 <= metadata["susceptibility_si"] <= 0.1
    assert np.isclose(model.max(), metadata["susceptibility_si"])

def test_loader_uses_canonical_magnetic_keys(tmp_path):
    path = tmp_path / "sample.npz"
    np.savez(path, tmi=np.ones(TMI_SHAPE[:-1]), susceptibility=np.ones(SUSCEPTIBILITY_SHAPE[:-1]))
    tmi, susceptibility = load_npz_sample(path, tmi_shape=TMI_SHAPE)
    assert tmi.shape == TMI_SHAPE
    assert susceptibility.shape == SUSCEPTIBILITY_SHAPE

def test_tmi_and_susceptibility_metrics_are_exact_for_identity():
    tmi = np.array([[-2.0, 3.0]])
    model = np.array([[[0.0, 0.02]]])
    assert calculate_tmi_fit_metrics(tmi, tmi)["tmi_rmse_nt"] == 0.0
    assert calculate_susceptibility_metrics(model, model, 0.001)["support_iou"] == 1.0

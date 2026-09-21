import numpy as np
from cnn_inversion_3d.dataset import load_npz_sample, TMI_SHAPE, SUSCEPTIBILITY_SHAPE
from dataset_generation.generate_single_plane_dataset import (
    MODEL_SHAPE_ZYX, build_generation_summary, sample_susceptibility,
)
from dataset_generation.e02_config import E02_DEFAULT_CONFIG
from e01_magnetic.config import MagneticSurveyConfig
from forward_modeling.forward_model import TMIForwardModel, make_tensor_grid
from evaluation.tmi_metrics import calculate_tmi_fit_metrics, calculate_susceptibility_metrics
from evaluation.plot_e01_3d import select_representative_samples

def test_sample_is_susceptibility_in_si():
    model, metadata = sample_susceptibility(np.random.default_rng(4))
    assert model.shape == MODEL_SHAPE_ZYX
    assert 0.0001 <= metadata["susceptibility_si"] <= 0.1
    assert np.isclose(model.max(), metadata["susceptibility_si"])

def test_e02_log_uniform_sampling_is_reproducible_and_spans_decades():
    first = np.random.default_rng(20260727)
    second = np.random.default_rng(20260727)
    first_values = [sample_susceptibility(first)[1]["susceptibility_si"] for _ in range(3000)]
    second_values = [sample_susceptibility(second)[1]["susceptibility_si"] for _ in range(3000)]
    np.testing.assert_array_equal(first_values, second_values)
    exponents = np.log10(first_values)
    assert np.all((-4.0 <= exponents) & (exponents <= -1.0))
    counts = np.histogram(exponents, bins=[-4.0, -3.0, -2.0, -1.0])[0]
    assert np.all(counts > 850)
    assert np.all(counts < 1150)

def test_e02_preserves_tmi_normalization_and_finite_forward_output():
    assert E02_DEFAULT_CONFIG.tmi_normalization_nt == 100.0
    model, _ = sample_susceptibility(np.random.default_rng(7))
    survey = MagneticSurveyConfig()
    grid = make_tensor_grid([0, 640, 0, 640, 0, 240], [10, 10], 10)
    selected_receivers = survey.receiver_xyz[[0, 3280, -1]]
    forward = TMIForwardModel(
        grid, selected_receivers, survey.field_strength_nt,
        survey.inclination_deg, survey.declination_deg, survey.azimuth_deg,
    )
    tmi = forward.predict(np.transpose(model, (2, 1, 0)))
    assert tmi.shape == (3,)
    assert np.all(np.isfinite(tmi))
    normalized = tmi / E02_DEFAULT_CONFIG.tmi_normalization_nt
    np.testing.assert_allclose(normalized * 100.0, tmi)

def test_e02_summary_bins_are_non_overlapping_and_exhaustive():
    susceptibility = [1e-4, 1e-3, 1e-2, 1e-1]
    peaks = [0.5, 1.0, 10.0, 1000.0]
    rows = [
        {"susceptibility_si": value, "tmi_min_nt": -peak,
         "tmi_max_nt": peak, "peak_absolute_tmi_nt": peak,
         "split": ("train", "validation", "test", "test")[index]}
        for index, (value, peak) in enumerate(zip(susceptibility, peaks))
    ]
    summary = build_generation_summary(rows)
    assert sum(summary["susceptibility_decade_counts"].values()) == 4
    assert sum(summary["peak_absolute_tmi_counts"].values()) == 4
    assert summary["peak_absolute_tmi_counts"]["below_1_nt"] == 1
    assert summary["peak_absolute_tmi_counts"]["1_to_below_10_nt"] == 1
    assert summary["peak_absolute_tmi_counts"]["10_to_below_100_nt"] == 1
    assert summary["peak_absolute_tmi_counts"]["1000_nt_and_above"] == 1

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

def test_representative_results_select_best_average_and_worst(tmp_path):
    metrics = tmp_path / "combined_test_metrics.csv"
    metrics.write_text(
        "sample_id,support_iou\n"
        "sample_a,0.1\n"
        "sample_b,0.3\n"
        "sample_c,0.6\n"
        "sample_d,0.9\n",
        encoding="utf-8",
    )
    selected = select_representative_samples(metrics)
    assert [item[1] for item in selected] == ["sample_d", "sample_c", "sample_a"]

"""Regression tests for the separate E02 evaluation baseline."""
import csv
import json

import numpy as np

from evaluation.audit_e02 import run_audit
from evaluation.tmi_metrics import calculate_e02_support_metrics, calculate_support_metrics


def test_weak_real_body_is_geologically_present():
    truth = np.array([0.0, 0.0001, 0.0])
    prediction = np.array([0.0, 0.0001, 0.0])
    result = calculate_e02_support_metrics(truth, prediction)
    assert result["geological_support"]["true_occupied_cells"] == 1
    assert result["geological_support"]["iou"] == 1
    assert result["thresholded_susceptibility_support"]["true_occupied_cells"] == 0


def test_support_perfect_empty_disjoint_and_overextended():
    truth = np.array([True, True, False, False])
    assert calculate_support_metrics(truth, truth)["iou"] == 1
    assert calculate_support_metrics(np.zeros(4, bool), np.zeros(4, bool)) == {
        "iou":1.0,"dice":1.0,"precision":1.0,"recall":1.0,
        "true_occupied_cells":0,"predicted_occupied_cells":0,
        "predicted_to_true_volume_ratio":1.0}
    empty = calculate_support_metrics(truth, np.zeros(4, bool))
    assert (empty["iou"], empty["precision"], empty["recall"], empty["predicted_to_true_volume_ratio"]) == (0,0,0,0)
    disjoint = calculate_support_metrics(truth, ~truth)
    assert disjoint["iou"] == disjoint["dice"] == 0
    extended = calculate_support_metrics(truth, np.array([True, True, True, False]))
    assert np.isclose(extended["precision"], 2/3)
    assert extended["recall"] == 1
    assert extended["predicted_to_true_volume_ratio"] == 1.5
    assert calculate_support_metrics(np.zeros(4, bool), truth)["predicted_to_true_volume_ratio"] is None


def test_threshold_equality_is_included_for_truth_and_prediction():
    truth = np.array([0.001, 0.0])
    prediction = np.array([0.001, 0.00005])
    result = calculate_e02_support_metrics(truth, prediction)
    assert result["thresholded_susceptibility_support"]["iou"] == 1
    assert result["geological_support"]["predicted_occupied_cells"] == 2


def test_audit_csv_json_and_metadata_agree(tmp_path):
    dataset = tmp_path / "dataset"
    predictions = tmp_path / "predictions"
    output = tmp_path / "audit"
    (dataset / "samples").mkdir(parents=True)
    (predictions / "tmi_consistency" / "sample_000001").mkdir(parents=True)
    truth = np.zeros((2, 2, 2), np.float32)
    truth[0, 0, 0] = 0.0001
    predicted = truth.copy()
    tmi = np.ones((2, 2), np.float32)
    np.savez(dataset / "samples" / "sample_000001.npz", tmi=tmi, susceptibility=truth)
    np.savez(predictions / "sample_000001_prediction.npz", tmi=tmi,
             true_susceptibility=truth, recovered_susceptibility=predicted)
    np.save(predictions / "tmi_consistency" / "sample_000001" / "recovered_tmi.npy", tmi)
    (dataset / "metadata.json").write_text(json.dumps({"dataset_version":"test",
        "tmi_input_normalization":{"operation":"divide","scale_nt":100.0}}))
    with (dataset / "test_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id","relative_path","susceptibility_si",
            "z_start","z_end","y_start","y_end","x_start","x_end"])
        writer.writeheader()
        writer.writerow({"sample_id":"sample_000001","relative_path":"samples/sample_000001.npz",
            "susceptibility_si":0.0001,"z_start":0,"z_end":1,"y_start":0,"y_end":1,
            "x_start":0,"x_end":1})
    run_audit(dataset, predictions, output)
    with (output / "per_sample_metrics.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    report = json.loads((output / "per_sample_reports.json").read_text())[0]
    metadata = json.loads((output / "evaluation_metadata.json").read_text())
    assert float(row["geological_iou"]) == report["geological_support"]["iou"] == 1
    assert metadata["status"] == "unverified_saved_prediction_arrays"
    assert metadata["checkpoint_path"] is None
    assert metadata["geological_prediction_threshold_si"] == 0.00005
    assert metadata["thresholded_diagnostic_si"] == 0.001

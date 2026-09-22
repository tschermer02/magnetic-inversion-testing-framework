"""Re-audit E02 predictions without changing historical evaluation artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from evaluation.tmi_metrics import (
    calculate_e02_support_metrics, calculate_susceptibility_metrics,
    calculate_tmi_fit_metrics,
)

DEFAULT_DATASET = Path("datasets/E02")
DEFAULT_PREDICTIONS = Path("prediction_outputs/E02")
DEFAULT_OUTPUT = Path("prediction_outputs/E02_baseline_audit")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, object]:
    def run(*args):
        try:
            result = subprocess.run(["git", *args], capture_output=True, text=True,
                                    check=False, timeout=10)
        except subprocess.TimeoutExpired:
            return None
        return result.stdout.strip() if result.returncode == 0 else None
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": run("rev-parse", "HEAD"),
            "dirty_tracked_files": bool(status) if status is not None else None,
            "untracked_files_checked": False}


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict]) -> dict:
    if not rows:
        return {"sample_count": 0}
    columns = [key for key, value in rows[0].items() if isinstance(value, (int, float))
               and not isinstance(value, bool)]
    result = {"sample_count": len(rows)}
    for column in columns:
        values = np.array([row[column] for row in rows if row[column] is not None], dtype=float)
        finite = values[np.isfinite(values)]
        result[column] = {"mean": float(np.mean(finite)) if len(finite) else None,
                          "median": float(np.median(finite)) if len(finite) else None,
                          "finite_count": int(len(finite))}
    return result


def _flatten(prefix: str, values: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _body_mask(row: dict, shape: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[int(row["z_start"]):int(row["z_end"]),
         int(row["y_start"]):int(row["y_end"]),
         int(row["x_start"]):int(row["x_end"])] = True
    return mask


def _decade(value: float) -> str:
    if 0.0001 <= value < 0.001:
        return "0.0001_to_below_0.001_si"
    if 0.001 <= value < 0.01:
        return "0.001_to_below_0.01_si"
    if 0.01 <= value <= 0.1:
        return "0.01_to_0.1_si"
    raise ValueError(f"Susceptibility {value} SI is outside E02's stated range.")


def _load_checkpoint_predictions(checkpoint: Path, dataset: Path, manifest: list[dict],
                                 tmi_scale: float, run_metadata: dict | None = None) -> tuple[dict[str, np.ndarray], float]:
    """Explicitly load wrapper weights and evaluate validation before test inference."""
    import tensorflow as tf
    from cnn_inversion_3d.dataset import TMI_SHAPE, build_training_datasets
    from cnn_inversion_3d.e01_core_loss import build_e01_sensitivity_weights
    from cnn_inversion_3d.e01_training import E01LossConfig, E01TrainingModel
    from cnn_inversion_3d.model import ModelConfig, build_e01_model
    from cnn_inversion_3d.train import DisabledTMIForward

    _, weights = build_e01_sensitivity_weights()
    occupancy = (run_metadata or {}).get("occupancy", {})
    config = E01LossConfig(lambda_depth=2.0, lambda_amplitude=1.0, lambda_tmi=0.0,
        lambda_tversky=0.1, tversky_alpha=0.7, tversky_beta=0.3,
        occupancy_threshold=0.001, occupancy_sharpness=1000.0,
        occupancy_mode=occupancy.get("mode", "legacy_threshold_sigmoid"),
        occupancy_tau_si=float(occupancy.get("tau_si", 1.0e-4 / np.log(100.0))))
    model = E01TrainingModel(build_e01_model(ModelConfig(base_filters=8)), weights,
        DisabledTMIForward(), tmi_scale=tmi_scale, loss_config=config)
    model(tf.zeros((1, *TMI_SHAPE), tf.float32), training=False)
    model.load_weights(checkpoint)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), jit_compile=False)
    _, validation, _, _ = build_training_datasets(dataset_directory=dataset,
        batch_size=2, tmi_scale=tmi_scale, susceptibility_scale=1.0, random_seed=20260727)
    print("Verifying checkpoint validation loss...", flush=True)
    validation_loss = float(model.evaluate(validation, verbose=0, return_dict=True)["loss"])
    print(f"Validation loss: {validation_loss:.8g}", flush=True)
    predictions = {}
    for index, row in enumerate(manifest, start=1):
        with np.load(dataset / row["relative_path"]) as saved:
            tmi = saved["tmi"]
        predictions[row["sample_id"]] = np.asarray(
            model.inversion_model(tmi[None, ..., None] / tmi_scale, training=False)[0, ..., 0]
        )
        if index % 10 == 0 or index == len(manifest):
            print(f"CNN predictions: {index}/{len(manifest)}", flush=True)
    return predictions, validation_loss


def validate_checkpoint(checkpoint: Path, dataset: Path, output: Path,
                        validation_tolerance: float = 1e-3) -> dict:
    """Evaluate only validation loss, leaving the test split untouched."""
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    run_metadata_path = checkpoint.parent / "run_metadata.json"
    if not run_metadata_path.is_file():
        raise FileNotFoundError(f"Matching run metadata is required: {run_metadata_path}")
    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    dataset_metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    tmi_scale = float(dataset_metadata["tmi_input_normalization"]["scale_nt"])
    _, validation_loss = _load_checkpoint_predictions(
        checkpoint, dataset, [], tmi_scale, run_metadata
    )
    recorded = float(run_metadata["best_validation_loss"])
    difference = abs(validation_loss - recorded)
    result = {"experiment":run_metadata.get("experiment"),
              "checkpoint_path":str(checkpoint.resolve()),
              "checkpoint_sha256":_sha256(checkpoint),
              "recorded_best_epoch":run_metadata.get("best_epoch"),
              "recorded_best_validation_loss":recorded,
              "recomputed_validation_loss":validation_loss,
              "absolute_difference":difference,
              "tolerance":validation_tolerance,
              "within_tolerance":difference <= validation_tolerance,
              "dataset_version":dataset_metadata["dataset_version"],
              "test_set_evaluated":False,
              "timestamp_utc":datetime.now(timezone.utc).isoformat()}
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation_checkpoint.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    if not result["within_tolerance"]:
        raise ValueError(f"Recomputed validation loss differs by {difference:g}, exceeding {validation_tolerance:g}.")
    print(json.dumps(result,indent=2))
    return result


def run_audit(dataset: Path, predictions: Path, output: Path, *,
              prediction_threshold_si: float = 0.00005,
              diagnostic_threshold_si: float = 0.001,
              checkpoint: Path | None = None,
              validation_tolerance: float = 1e-3) -> dict:
    if output.resolve() == predictions.resolve():
        raise ValueError("Audit output must not overwrite historical predictions.")
    if not 0 < prediction_threshold_si < diagnostic_threshold_si:
        raise ValueError("Require 0 < geological prediction threshold < diagnostic threshold.")
    metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    tmi_scale = float(metadata["tmi_input_normalization"]["scale_nt"])
    with (dataset / "test_manifest.csv").open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    if not manifest:
        raise ValueError("E02 test manifest is empty.")
    history_path = Path("outputs/E02/training_history.csv")
    best_epoch = best_val = None
    if history_path.exists():
        with history_path.open(newline="", encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
        best = min(history, key=lambda row: float(row["val_loss"]))
        best_epoch, best_val = int(best["epoch"]) + 1, float(best["val_loss"])
    checkpoint_run_metadata = None
    if checkpoint is not None and (checkpoint.parent / "run_metadata.json").is_file():
        checkpoint_run_metadata = json.loads((checkpoint.parent / "run_metadata.json").read_text(encoding="utf-8"))
        best_epoch = checkpoint_run_metadata.get("best_epoch")
        best_val = checkpoint_run_metadata.get("best_validation_loss")
    regenerated = None
    validation_loss = None
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        regenerated, validation_loss = _load_checkpoint_predictions(
            checkpoint, dataset, manifest, tmi_scale, checkpoint_run_metadata
        )
        if best_val is not None and abs(validation_loss - best_val) > validation_tolerance:
            raise ValueError(f"Checkpoint validation loss {validation_loss:.6g} differs from "
                             f"recorded best {best_val:.6g} by more than {validation_tolerance:g}.")
    output.mkdir(parents=True, exist_ok=True)

    batched_recovered_tmi = None
    if regenerated is not None:
        from e01_magnetic.config import MagneticSurveyConfig
        from forward_modeling.forward_model import TMIForwardModel, make_tensor_grid
        survey = MagneticSurveyConfig()
        forward = TMIForwardModel(make_tensor_grid([0,640,0,640,0,240],[10,10],10),
            survey.receiver_xyz, survey.field_strength_nt, survey.inclination_deg,
            survey.declination_deg, survey.azimuth_deg)
        model_matrix = np.stack([
            np.asarray(regenerated[item["sample_id"]], dtype=np.float64)
              .transpose(2,1,0).ravel(order="F")
            for item in manifest
        ])
        print(f"Forward modeling {len(manifest)} predictions together (CPU/NumPy)...", flush=True)
        responses = forward.predict_many(model_matrix).reshape(len(manifest),81,81)
        batched_recovered_tmi = {
            item["sample_id"]: responses[index] for index,item in enumerate(manifest)
        }
        print("Forward modeling complete.", flush=True)

    historical = {}
    old_csv = predictions / "combined_test_metrics.csv"
    if old_csv.exists():
        with old_csv.open(newline="", encoding="utf-8") as stream:
            historical = {row["sample_id"]: row for row in csv.DictReader(stream)}
    rows = []
    reports = []
    historical_mismatch = 0
    historical_true_mismatch = 0
    historical_predicted_mismatch = 0
    below_diagnostic = 0
    cached_tmi_count = 0
    for item in manifest:
        sample_id = item["sample_id"]
        with np.load(dataset / item["relative_path"]) as saved:
            tmi = np.asarray(saved["tmi"], dtype=np.float64)
            truth = np.asarray(saved["susceptibility"], dtype=np.float64)
        if regenerated is None:
            with np.load(predictions / f"{sample_id}_prediction.npz") as saved:
                if not np.array_equal(saved["tmi"], tmi) or not np.array_equal(saved["true_susceptibility"], truth):
                    raise ValueError(f"Saved prediction truth/TMI differs from dataset for {sample_id}.")
                predicted = np.asarray(saved["recovered_susceptibility"], dtype=np.float64)
        else:
            predicted = np.asarray(regenerated[sample_id], dtype=np.float64)
            regenerated_directory = output / "regenerated_predictions"
            regenerated_directory.mkdir(exist_ok=True)
            np.savez_compressed(regenerated_directory / f"{sample_id}_prediction.npz",
                tmi=tmi.astype(np.float32), true_susceptibility=truth.astype(np.float32),
                recovered_susceptibility=predicted.astype(np.float32))
        body = _body_mask(item, truth.shape)
        if not np.array_equal(body, truth > 0):
            raise ValueError(f"Manifest body mask differs from exact-zero dataset truth for {sample_id}.")
        if float(item["susceptibility_si"]) < diagnostic_threshold_si:
            below_diagnostic += 1
        support = calculate_e02_support_metrics(truth, predicted, geological_mask=body,
            prediction_threshold_si=prediction_threshold_si,
            diagnostic_threshold_si=diagnostic_threshold_si)
        susceptibility = calculate_susceptibility_metrics(truth, predicted, diagnostic_threshold_si)
        old = historical.get(sample_id)
        if old:
            truth_differs = int(old["true_occupied_cells"]) != susceptibility["true_occupied_cells"]
            predicted_differs = int(old["recovered_occupied_cells"]) != susceptibility["recovered_occupied_cells"]
            historical_mismatch += int(truth_differs or predicted_differs)
            historical_true_mismatch += int(truth_differs)
            historical_predicted_mismatch += int(predicted_differs)
        zero_support = calculate_e02_support_metrics(truth, np.zeros_like(truth), geological_mask=body,
            prediction_threshold_si=prediction_threshold_si,
            diagnostic_threshold_si=diagnostic_threshold_si)
        zero_susceptibility = calculate_susceptibility_metrics(truth, np.zeros_like(truth), diagnostic_threshold_si)
        if regenerated is None:
            cache = predictions / "tmi_consistency" / sample_id / "recovered_tmi.npy"
            if not cache.is_file():
                raise FileNotFoundError(f"Missing saved forward-TMI map: {cache}")
            recovered_tmi = np.asarray(np.load(cache), dtype=np.float64)
            cached_tmi_count += 1
        else:
            recovered_tmi = batched_recovered_tmi[sample_id]
        tmi_metrics = calculate_tmi_fit_metrics(tmi, recovered_tmi)
        zero_tmi = calculate_tmi_fit_metrics(tmi, np.zeros_like(tmi))
        row = {"sample_id":sample_id, "susceptibility_si":float(item["susceptibility_si"]),
               "susceptibility_decade":_decade(float(item["susceptibility_si"])),
               **_flatten("geological", support["geological_support"]),
               **_flatten("thresholded_0p001_si", support["thresholded_susceptibility_support"]),
               **{key:value for key,value in susceptibility.items() if key in (
                   "susceptibility_mae_si","susceptibility_rmse_si","susceptibility_relative_l2")},
               **tmi_metrics,
               **_flatten("zero_geological", zero_support["geological_support"]),
               **_flatten("zero_thresholded_0p001_si", zero_support["thresholded_susceptibility_support"]),
               **_flatten("zero", {**{key:value for key,value in zero_susceptibility.items()
                   if key.startswith("susceptibility_")}, **zero_tmi})}
        rows.append(row)
        reports.append({"sample_id":sample_id, "geological_support":support["geological_support"],
                        "thresholded_susceptibility_support":support["thresholded_susceptibility_support"],
                        "susceptibility_errors":{key:row[key] for key in (
                            "susceptibility_mae_si","susceptibility_rmse_si","susceptibility_relative_l2")},
                        "tmi_fit":tmi_metrics})
        if len(rows) % 10 == 0 or len(rows) == len(manifest):
            print(f"Metrics complete: {len(rows)}/{len(manifest)}", flush=True)
    _write_csv(output / "per_sample_metrics.csv", rows)
    (output / "per_sample_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    grouped = {name:_summary([row for row in rows if row["susceptibility_decade"] == name])
               for name in ("0.0001_to_below_0.001_si","0.001_to_below_0.01_si","0.01_to_0.1_si")}
    summary = {"overall":_summary(rows), "by_susceptibility_decade":grouped}
    (output / "aggregate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = list(grouped)
    means = [grouped[label].get("geological_iou", {}).get("mean") or 0 for label in labels]
    threshold_means = [grouped[label].get("thresholded_0p001_si_iou", {}).get("mean") or 0 for label in labels]
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(x - 0.18, means, 0.36, label="Geological support (prediction >= 0.00005 SI)")
    axis.bar(x + 0.18, threshold_means, 0.36, label="Symmetric >= 0.001 SI diagnostic")
    axis.set_xticks(x, ["0.0001-0.001", "0.001-0.01", "0.01-0.1"])
    axis.set_xlabel("True body susceptibility decade (SI)")
    axis.set_ylabel("Mean IoU")
    axis.set_title("E02 support metrics by susceptibility decade")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "support_iou_by_decade.png", dpi=160)
    plt.close(figure)
    provenance = {"status":"verified_explicit_checkpoint" if checkpoint else "unverified_saved_prediction_arrays",
                  "checkpoint_path":str(checkpoint.resolve()) if checkpoint else None,
                  "checkpoint_sha256":_sha256(checkpoint) if checkpoint else None,
                  "checkpoint_epoch":best_epoch if checkpoint else None,
                  "training_commit":(checkpoint_run_metadata or {}).get("git",{}).get("commit"),
                  "recorded_best_validation_epoch":best_epoch,
                  "recorded_best_validation_loss":best_val,
                  "recomputed_validation_loss":validation_loss,
                  "validation_loss_tolerance":validation_tolerance,
                  "model":"E01 asymmetric 2-D U-Net, base_filters=8, 190592 parameters",
                  "loss":"E01 composite soft-Tversky; depth=2, amplitude=1, TMI=0, Tversky=0.1",
                  "dataset_path":str(dataset.resolve()),
                  "dataset_version":metadata["dataset_version"],
                  "dataset_metadata_sha256":_sha256(dataset / "metadata.json"),
                  "test_manifest_sha256":_sha256(dataset / "test_manifest.csv"),
                  "tmi_normalization":metadata["tmi_input_normalization"],
                  "geological_prediction_threshold_si":prediction_threshold_si,
                  "thresholded_diagnostic_si":diagnostic_threshold_si,
                  "threshold_policy":"geological truth=manifest body; predictions >= threshold; diagnostic truth and predictions >= threshold",
                  "empty_mask_policy":"both empty scores=1; one empty scores=0; true-empty predicted-nonempty volume ratio=null",
                  "geological_threshold_note":"0.00005 SI is provisional evaluation convention, not validated detection limit",
                  "tmi_source":"saved recovered_tmi.npy (not independently forward-verified)" if cached_tmi_count else "fresh forward calculation",
                  "git":_git_state(), "timestamp_utc":datetime.now(timezone.utc).isoformat()}
    (output / "evaluation_metadata.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    report = (f"# E02 baseline audit\n\n"
        f"- {len(rows)} test samples; dataset truth and TMI exactly match saved prediction arrays.\n"
        f"- Corrected mean geological IoU={summary['overall']['geological_iou']['mean']:.4f}; "
        f"mean symmetric 0.001 SI IoU={summary['overall']['thresholded_0p001_si_iou']['mean']:.4f}.\n"
        f"- Saved predictions have mean susceptibility MAE={summary['overall']['susceptibility_mae_si']['mean']:.6g} SI "
        f"and cached-forward TMI MAE={summary['overall']['tmi_mae_nt']['mean']:.4g} nT; "
        f"zero predictions score {summary['overall']['zero_susceptibility_mae_si']['mean']:.6g} SI "
        f"and {summary['overall']['zero_tmi_mae_nt']['mean']:.4g} nT, respectively.\n"
        f"- {below_diagnostic} test bodies have susceptibility below {diagnostic_threshold_si:g} SI; "
        "the symmetric thresholded diagnostic correctly excludes them, while geological support retains them.\n"
        f"- {historical_mismatch} historical CSV rows have occupied-cell counts inconsistent with the "
        "current symmetric >0.001 SI evaluator applied to saved arrays. No historical threshold is inferred.\n"
        f"  Truth counts differ in {historical_true_mismatch} rows; predicted counts differ in "
        f"{historical_predicted_mismatch} rows.\n"
        "- Training inconsistency remains: Tversky target uses 0.001 SI whereas susceptibility/depth losses "
        "use all positive-susceptibility bodies. Training is unchanged.\n"
        f"- Checkpoint provenance: {provenance['status']}. Recorded best validation epoch {best_epoch} "
        "does not establish the source of saved predictions.\n"
        "- Historical artifacts were not overwritten. Cached forward-TMI maps are reused when no checkpoint "
        "is available; their correspondence to saved prediction arrays remains unverified.\n"
        "- Geological prediction threshold 0.00005 SI is provisional, not a detection limit and not test-optimized.\n")
    (output / "audit_report.md").write_text(report, encoding="utf-8")
    print(report)
    return {"summary":summary, "metadata":provenance, "historical_mismatch_count":historical_mismatch}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prediction-threshold-si", type=float, default=0.00005)
    parser.add_argument("--diagnostic-threshold-si", type=float, default=0.001)
    parser.add_argument("--checkpoint", type=Path,
        help="Explicit E01 wrapper .weights.h5; triggers validation and regenerated test predictions.")
    parser.add_argument("--validation-tolerance", type=float, default=1e-3)
    parser.add_argument("--validation-only", action="store_true",
        help="Verify checkpoint validation loss without reading or evaluating the test split.")
    args = parser.parse_args()
    if args.validation_only:
        if args.checkpoint is None:
            parser.error("--validation-only requires --checkpoint")
        validate_checkpoint(args.checkpoint,args.dataset,args.output,args.validation_tolerance)
        return
    run_audit(args.dataset,args.predictions,args.output,
        prediction_threshold_si=args.prediction_threshold_si,
        diagnostic_threshold_si=args.diagnostic_threshold_si,
        checkpoint=args.checkpoint,validation_tolerance=args.validation_tolerance)


if __name__ == "__main__":
    main()

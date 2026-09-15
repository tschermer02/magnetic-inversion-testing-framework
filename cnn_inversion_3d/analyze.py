"""Analyze one E01 susceptibility prediction and its reconstructed TMI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from e01_magnetic.config import MagneticSurveyConfig
from evaluation.tmi_metrics import (
    calculate_susceptibility_metrics,
    calculate_tmi_fit_metrics,
)
from forward_modeling.forward_model import TMIForwardModel, make_tensor_grid


def analyze_prediction(
    path: Path,
    *,
    threshold_si: float = 0.001,
    figure_path: Path | None = None,
) -> dict[str, object]:
    """Calculate susceptibility and forward-TMI metrics for one prediction."""
    with np.load(path) as archive:
        required = {"tmi", "true_susceptibility", "recovered_susceptibility"}
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"{path.name} is missing arrays: {sorted(missing)}")
        observed_tmi = np.asarray(archive["tmi"], dtype=np.float64)
        truth = np.asarray(archive["true_susceptibility"], dtype=np.float64)
        recovered = np.asarray(archive["recovered_susceptibility"], dtype=np.float64)

    if observed_tmi.shape != (81, 81):
        raise ValueError(f"Expected TMI shape (81, 81), received {observed_tmi.shape}.")
    if truth.shape != (24, 64, 64) or recovered.shape != truth.shape:
        raise ValueError("Susceptibility volumes must both have shape (24, 64, 64).")

    survey = MagneticSurveyConfig()
    grid = make_tensor_grid([0, 640, 0, 640, 0, 240], [10, 10], 10)
    forward = TMIForwardModel(
        grid,
        survey.receiver_xyz,
        survey.field_strength_nt,
        survey.inclination_deg,
        survey.declination_deg,
        survey.azimuth_deg,
    )
    recovered_tmi = forward.predict(np.transpose(recovered, (2, 1, 0))).reshape(81, 81)
    residual_tmi = recovered_tmi - observed_tmi
    if figure_path is not None:
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(figure_path.parent / "true_tmi.npy", observed_tmi.astype(np.float32))
        np.save(figure_path.parent / "recovered_tmi.npy", recovered_tmi.astype(np.float32))
        np.save(figure_path.parent / "tmi_residual.npy", residual_tmi.astype(np.float32))
        observed_limit = max(float(np.max(np.abs(observed_tmi))), 1e-12)
        recovered_limit = max(float(np.max(np.abs(recovered_tmi))), 1e-12)
        residual_limit = max(float(np.max(np.abs(residual_tmi))), 1e-12)
        figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
        panels = (
            (observed_tmi, f"True / Input TMI\nrange {observed_tmi.min():.3g} to {observed_tmi.max():.3g} nT", observed_limit),
            (recovered_tmi, f"Recovered TMI\nrange {recovered_tmi.min():.3g} to {recovered_tmi.max():.3g} nT", recovered_limit),
            (residual_tmi, "TMI Residual", residual_limit),
        )
        for axis, (values, title, limit) in zip(axes, panels):
            image = axis.imshow(values, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit,
                                extent=(survey.receiver_x_min_m, survey.receiver_x_max_m,
                                        survey.receiver_y_min_m, survey.receiver_y_max_m))
            axis.set_title(title); axis.set_xlabel("Easting (m)"); axis.set_ylabel("Northing (m)")
            figure.colorbar(image, ax=axis, shrink=0.82, label="TMI (nT)")
        figure.suptitle("E01 magnetic forward-consistency analysis")
        figure.savefig(figure_path, dpi=180)
        plt.close(figure)
    return {
        "prediction_file": str(path.resolve()),
        "units": {"tmi": "nT", "susceptibility": "dimensionless SI"},
        "susceptibility": calculate_susceptibility_metrics(truth, recovered, threshold_si),
        "tmi_fit": calculate_tmi_fit_metrics(observed_tmi, recovered_tmi),
        "observed_tmi_range_nt": [float(observed_tmi.min()), float(observed_tmi.max())],
        "recovered_tmi_range_nt": [float(recovered_tmi.min()), float(recovered_tmi.max())],
        "magnetic_comparison_figure": str(figure_path.resolve()) if figure_path is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold-si", type=float, default=0.001)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    figure_path = args.figure or args.output.with_suffix(".png")
    report = analyze_prediction(args.prediction, threshold_si=args.threshold_si, figure_path=figure_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Analysis written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()

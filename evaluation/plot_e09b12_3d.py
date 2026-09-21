"""Create interactive Plotly 3D density comparisons for E09B-12 predictions."""

from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DEFAULT_PREDICTIONS = Path("prediction_outputs/E09B_12_soft_tversky")
DEFAULT_OUTPUT = Path("analysis_outputs/E09B_12_soft_tversky/plotly_3d")
DENSITY_SHAPE = (24, 64, 64)
CELL_SIZE_M = 10.0


def _model_domain_box(*, showlegend: bool) -> list[go.BaseTraceType]:
    """Return a faint purple 640 x 640 x 240 m model-domain enclosure."""

    x = [0, 640, 640, 0, 0, 640, 640, 0]
    y = [0, 0, 640, 640, 0, 0, 640, 640]
    z = [0, 0, 0, 0, 240, 240, 240, 240]
    faces_i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
    faces_j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
    faces_k = [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]
    surface = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=faces_i,
        j=faces_j,
        k=faces_k,
        color="#c4a7e7",
        opacity=0.09,
        flatshading=True,
        hoverinfo="skip",
        name="Model domain (640 × 640 × 240 m)",
        legendgroup="model_domain",
        showlegend=showlegend,
    )
    edge_pairs = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    for start, end in edge_pairs:
        edge_x.extend((x[start], x[end], None))
        edge_y.extend((y[start], y[end], None))
        edge_z.extend((z[start], z[end], None))
    edges = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line={"color": "#9d7cd8", "width": 3},
        opacity=0.5,
        hoverinfo="skip",
        name="Model-domain edges",
        legendgroup="model_domain",
        showlegend=False,
    )
    return [surface, edges]


def normalize_sample_id(value: str) -> str:
    """Return the canonical ``sample_XXXXXX`` identifier."""

    text = value.strip()
    if text.startswith("sample_"):
        text = text[7:]
    if not text.isdigit():
        raise ValueError(f"Invalid sample identifier: {value!r}")
    return f"sample_{int(text):06d}"


def available_samples(predictions: Path) -> list[str]:
    """Return sorted sample identifiers having saved prediction volumes."""

    suffix = "_prediction.npz"
    return sorted(
        path.name[: -len(suffix)]
        for path in predictions.glob(f"sample_*{suffix}")
    )


def select_samples(
    available: list[str], requested: list[str] | None, random_count: int, seed: int
) -> list[str]:
    """Resolve explicit samples or a reproducible random subset."""

    if requested:
        selected = [normalize_sample_id(value) for value in requested]
        missing = sorted(set(selected) - set(available))
        if missing:
            raise FileNotFoundError(
                "Prediction files do not exist for: " + ", ".join(missing)
            )
        return list(dict.fromkeys(selected))
    if random_count < 1:
        raise ValueError("--random-samples must be at least one.")
    if random_count > len(available):
        raise ValueError(
            f"Requested {random_count} random samples, but only {len(available)} exist."
        )
    return sorted(random.Random(seed).sample(available, random_count))


def load_density_pair(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate canonical true/predicted ``density[z,y,x]`` arrays."""

    with np.load(path) as saved:
        required = {"true_density", "predicted_density"}
        missing = required - set(saved.files)
        if missing:
            raise KeyError(f"{path} is missing arrays: {sorted(missing)}")
        truth = np.asarray(saved["true_density"], dtype=np.float32).squeeze()
        prediction = np.asarray(saved["predicted_density"], dtype=np.float32).squeeze()
    if truth.shape != DENSITY_SHAPE or prediction.shape != DENSITY_SHAPE:
        raise ValueError(
            f"Expected density arrays {DENSITY_SHAPE}; got {truth.shape} and "
            f"{prediction.shape} in {path}."
        )
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        raise ValueError(f"Nonfinite density value in {path}.")
    return truth, prediction


def _isosurface(
    density: np.ndarray,
    *,
    threshold: float,
    common_maximum: float,
    name: str,
    showscale: bool,
) -> go.Isosurface:
    z, y, x = np.meshgrid(
        CELL_SIZE_M * (np.arange(24) + 0.5),
        CELL_SIZE_M * (np.arange(64) + 0.5),
        CELL_SIZE_M * (np.arange(64) + 0.5),
        indexing="ij",
    )
    return go.Isosurface(
        x=x.ravel(),
        y=y.ravel(),
        z=z.ravel(),
        value=density.ravel(),
        isomin=threshold,
        isomax=common_maximum,
        cmin=threshold,
        cmax=common_maximum,
        surface_count=5,
        colorscale="Viridis",
        opacity=0.55,
        caps={"x_show": False, "y_show": False, "z_show": False},
        colorbar={
            "title": "Density<br>(g/cm³)",
            "x": 1.02,
            "len": 0.75,
        },
        showscale=showscale,
        name=name,
        hovertemplate=(
            "x=%{x:.0f} m<br>y=%{y:.0f} m<br>depth=%{z:.0f} m"
            "<br>density=%{value:.3f} g/cm³<extra>" + name + "</extra>"
        ),
    )


def build_figure(
    sample_id: str,
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    threshold: float,
) -> go.Figure:
    """Build a common-scale true-versus-reconstructed 3D comparison."""

    common_maximum = max(float(np.max(truth)), float(np.max(prediction)), threshold)
    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("True density", "E09B-12 reconstructed density"),
        horizontal_spacing=0.04,
    )
    for trace in _model_domain_box(showlegend=True):
        figure.add_trace(trace, row=1, col=1)
    for trace in _model_domain_box(showlegend=False):
        figure.add_trace(trace, row=1, col=2)
    figure.add_trace(
        _isosurface(
            truth,
            threshold=threshold,
            common_maximum=common_maximum,
            name="True",
            showscale=False,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        _isosurface(
            prediction,
            threshold=threshold,
            common_maximum=common_maximum,
            name="Prediction",
            showscale=True,
        ),
        row=1,
        col=2,
    )
    occupied_true = int(np.count_nonzero(truth >= threshold))
    occupied_prediction = int(np.count_nonzero(prediction >= threshold))
    axis = {
        "xaxis": {"title": "X east (m)", "range": [0, 640]},
        "yaxis": {"title": "Y north (m)", "range": [0, 640]},
        "zaxis": {"title": "Depth (m)", "range": [240, 0]},
        "aspectmode": "manual",
        "aspectratio": {"x": 1.0, "y": 1.0, "z": 0.5},
        "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.0}},
    }
    figure.update_layout(
        title={
            "text": (
                f"{sample_id}: 3D density reconstruction<br>"
                f"<sup>Common density scale; occupancy ≥ {threshold:g} g/cm³ | "
                f"true cells={occupied_true:,}, predicted cells={occupied_prediction:,}</sup>"
            ),
            "x": 0.5,
        },
        scene=axis,
        scene2=axis,
        width=1400,
        height=720,
        margin={"l": 20, "r": 100, "t": 100, "b": 20},
    )
    return figure


def write_index(output: Path, sample_ids: list[str]) -> None:
    """Write a small local index linking the generated standalone figures."""

    links = "\n".join(
        f'<li><a href="{html.escape(sample_id)}_density_3d.html">'
        f"{html.escape(sample_id)}</a></li>"
        for sample_id in sample_ids
    )
    (output / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>E09B-12 3D density</title>"
        "<h1>E09B-12 interactive 3D density reconstructions</h1><ul>"
        f"{links}</ul>",
        encoding="utf-8",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sample",
        action="append",
        default=None,
        help="Sample ID or number; repeat for multiple explicit samples.",
    )
    parser.add_argument(
        "--random-samples",
        type=int,
        default=5,
        help="Reproducible random count when --sample is omitted. Default: 5.",
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument(
        "--include-plotlyjs",
        choices=("cdn", "directory"),
        default="cdn",
        help="Use CDN (small HTML) or save one local plotly.min.js copy.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between zero and one.")
    predictions = args.predictions.resolve()
    if not predictions.is_dir():
        raise FileNotFoundError(f"Prediction directory does not exist: {predictions}")
    available = available_samples(predictions)
    if not available:
        raise FileNotFoundError(f"No sample prediction files found in {predictions}")
    selected = select_samples(
        available, args.sample, args.random_samples, args.seed
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for sample_id in selected:
        source = predictions / f"{sample_id}_prediction.npz"
        truth, prediction = load_density_pair(source)
        figure = build_figure(
            sample_id, truth, prediction, threshold=args.threshold
        )
        destination = output / f"{sample_id}_density_3d.html"
        figure.write_html(
            destination,
            include_plotlyjs=args.include_plotlyjs,
            full_html=True,
            auto_open=False,
        )
        records.append(
            {
                "sample_id": sample_id,
                "source": str(source),
                "figure": destination.name,
                "threshold_g_cm3": args.threshold,
                "true_occupied_cells": int(np.count_nonzero(truth >= args.threshold)),
                "predicted_occupied_cells": int(
                    np.count_nonzero(prediction >= args.threshold)
                ),
                "true_maximum_density_g_cm3": float(np.max(truth)),
                "predicted_maximum_density_g_cm3": float(np.max(prediction)),
            }
        )
        print(f"Created: {destination}")
    write_index(output, selected)
    (output / "plot_metadata.json").write_text(
        json.dumps(
            {
                "experiment": "E09B-12",
                "density_order": "density[z,y,x]",
                "cell_size_m": CELL_SIZE_M,
                "coordinates": "+x east, +y north, +z downward",
                "common_density_scale_within_each_sample": True,
                "samples": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Index: {output / 'index.html'}")


if __name__ == "__main__":
    main()

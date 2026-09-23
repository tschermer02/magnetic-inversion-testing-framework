"""Create E01 Plotly 3D comparisons for the best, average, and worst results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULT_PREDICTIONS = Path("prediction_outputs/E01_soft_tversky_full")
DEFAULT_OUTPUT = Path("analysis_outputs/E01_soft_tversky_full/plotly_3d")
MODEL_SHAPE = (24, 64, 64)
CELL_SIZE_M = 10.0


def select_representative_samples(metrics_path: Path, metric: str = "support_iou"):
    """Select maximum, closest-to-mean, and minimum metric samples."""
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or metric not in rows[0]:
        raise ValueError(f"No {metric!r} values found in {metrics_path}.")
    values = [(row["sample_id"], float(row[metric])) for row in rows]
    best = max(values, key=lambda item: item[1])
    worst = min(values, key=lambda item: item[1])
    mean = float(np.mean([value for _, value in values]))
    remaining = [item for item in values if item[0] not in {best[0], worst[0]}]
    average = min(remaining or values, key=lambda item: abs(item[1] - mean))
    return [("best", *best), ("average", *average), ("worst", *worst)]


def load_susceptibility_pair(path: Path):
    with np.load(path) as saved:
        truth = np.asarray(saved["true_susceptibility"], np.float32).squeeze()
        prediction = np.asarray(saved["recovered_susceptibility"], np.float32).squeeze()
    if truth.shape != MODEL_SHAPE or prediction.shape != MODEL_SHAPE:
        raise ValueError(f"Expected {MODEL_SHAPE}; got {truth.shape} and {prediction.shape}.")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        raise ValueError(f"Nonfinite susceptibility in {path}.")
    return truth, prediction


def _model_domain_box(showlegend: bool):
    x = [0, 640, 640, 0, 0, 640, 640, 0]
    y = [0, 0, 640, 640, 0, 0, 640, 640]
    z = [0, 0, 0, 0, 240, 240, 240, 240]
    i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
    j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
    k = [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]
    surface = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color="#c4a7e7",
                        opacity=0.09, flatshading=True, hoverinfo="skip",
                        name="Model domain (640 x 640 x 240 m)", showlegend=showlegend)
    pairs = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
    ex, ey, ez = [], [], []
    for start, end in pairs:
        ex += [x[start], x[end], None]; ey += [y[start], y[end], None]; ez += [z[start], z[end], None]
    edges = go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                         line={"color": "#9d7cd8", "width": 3}, opacity=0.5,
                         hoverinfo="skip", showlegend=False)
    return surface, edges


def _isosurface(values, threshold, common_maximum, name, showscale):
    z, y, x = np.meshgrid(CELL_SIZE_M * (np.arange(24) + 0.5),
                          CELL_SIZE_M * (np.arange(64) + 0.5),
                          CELL_SIZE_M * (np.arange(64) + 0.5), indexing="ij")
    return go.Isosurface(
        x=x.ravel(), y=y.ravel(), z=z.ravel(), value=values.ravel(),
        isomin=threshold, isomax=common_maximum, cmin=threshold, cmax=common_maximum,
        surface_count=5, colorscale="Viridis", opacity=0.55,
        caps={"x_show": False, "y_show": False, "z_show": False},
        colorbar={"title": "Susceptibility<br>(SI)", "x": 1.02, "len": 0.75},
        showscale=showscale, name=name,
        hovertemplate="x=%{x:.0f} m<br>y=%{y:.0f} m<br>depth=%{z:.0f} m"
                      "<br>susceptibility=%{value:.5f} SI<extra>" + name + "</extra>")


def build_figure(sample_id, truth, prediction, *, threshold=0.001, label=None, metric_value=None,
                 experiment="E01"):
    """Build an E09B-12-style, common-scale true/predicted E01 figure."""
    common_maximum = max(float(np.max(truth)), float(np.max(prediction)), threshold * 1.001)
    figure = make_subplots(rows=1, cols=2, specs=[[{"type":"scene"},{"type":"scene"}]],
                           subplot_titles=("True susceptibility", f"{experiment} reconstructed susceptibility"),
                           horizontal_spacing=0.04)
    for trace in _model_domain_box(True): figure.add_trace(trace, row=1, col=1)
    for trace in _model_domain_box(False): figure.add_trace(trace, row=1, col=2)
    figure.add_trace(_isosurface(truth, threshold, common_maximum, "True", False), row=1, col=1)
    figure.add_trace(_isosurface(prediction, threshold, common_maximum, "Prediction", True), row=1, col=2)
    axis = {"xaxis":{"title":"X east (m)","range":[0,640]},
            "yaxis":{"title":"Y north (m)","range":[0,640]},
            "zaxis":{"title":"Depth (m)","range":[240,0]}, "aspectmode":"manual",
            "aspectratio":{"x":1,"y":1,"z":0.5}, "camera":{"eye":{"x":1.45,"y":1.45,"z":1}}}
    descriptor = f"{label.title()} result | support IoU={metric_value:.4f} | " if label else ""
    figure.update_layout(title={"text":f"{sample_id}: {experiment} 3D susceptibility reconstruction<br>"
        f"<sup>{descriptor}common scale; occupancy &gt;= {threshold:g} SI | "
        f"true cells={np.count_nonzero(truth >= threshold):,}, predicted cells={np.count_nonzero(prediction >= threshold):,}</sup>","x":0.5},
        scene=axis, scene2=axis, width=1400, height=720, margin={"l":20,"r":100,"t":100,"b":20})
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metric", default="support_iou")
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--include-plotlyjs", choices=("cdn", "directory"), default="cdn")
    args = parser.parse_args()
    selected = select_representative_samples(args.predictions / "combined_test_metrics.csv", args.metric)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for label, sample_id, value in selected:
        source = args.predictions / f"{sample_id}_prediction.npz"
        truth, prediction = load_susceptibility_pair(source)
        destination = args.output / f"{label}_{sample_id}_susceptibility_3d.html"
        build_figure(sample_id, truth, prediction, threshold=args.threshold,
                     label=label, metric_value=value).write_html(
                         destination, include_plotlyjs=args.include_plotlyjs, full_html=True)
        records.append({"selection":label,"sample_id":sample_id,args.metric:value,"figure":destination.name})
        print(f"Created: {destination}")
    (args.output / "plot_metadata.json").write_text(json.dumps({"experiment":"E01",
        "selection_rule":"best=max, average=closest to arithmetic mean, worst=min",
        "metric":args.metric,"threshold_si":args.threshold,"samples":records}, indent=2), encoding="utf-8")


if __name__ == "__main__": main()

"""Export an E01-compatible Plotly 3D comparison as a rotating GIF."""
from __future__ import annotations
import argparse, csv, io, math
from pathlib import Path
import numpy as np
from PIL import Image
from evaluation.plot_e01_3d import (DEFAULT_OUTPUT, DEFAULT_PREDICTIONS, build_figure,
                                    load_susceptibility_pair, select_representative_samples)


def render_gif(figure, output_path, frames=80, duration_seconds=8.0):
    images = []
    for index, theta in enumerate(np.linspace(0, 2 * math.pi, frames, endpoint=False)):
        camera = {"eye":{"x":2.05*math.cos(theta),"y":2.05*math.sin(theta),"z":1.0}}
        figure.update_layout(scene_camera=camera, scene2_camera=camera)
        png = figure.to_image(format="png", width=1600, height=900, scale=1)
        images.append(Image.open(io.BytesIO(png)).convert("RGB"))
        print(f"Rendered frame {index + 1}/{frames}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output_path, save_all=True, append_images=images[1:],
                   duration=round(duration_seconds * 1000 / frames), loop=0,
                   disposal=2, optimize=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--result", choices=("best","average","median","worst"), default="average")
    parser.add_argument("--sample-id",
        help="Explicit sample ID; overrides automatic best/average/worst selection.")
    parser.add_argument("--metric", default="support_iou")
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--experiment", default="E01",
        help="Experiment label shown in the figure title, for example E04.")
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--duration-seconds", type=float, default=8.0)
    args = parser.parse_args()
    if args.sample_id:
        with (args.predictions / "combined_test_metrics.csv").open(newline="",encoding="utf-8") as stream:
            matching=[row for row in csv.DictReader(stream) if row["sample_id"]==args.sample_id]
        if not matching: raise ValueError(f"No metrics row found for {args.sample_id}")
        sample_id,value=args.sample_id,float(matching[0][args.metric])
    else:
        if args.result=="median": raise ValueError("--result median requires --sample-id")
        selected = {label:(sample,value) for label,sample,value in
                    select_representative_samples(args.predictions / "combined_test_metrics.csv", args.metric)}
        sample_id, value = selected[args.result]
    truth, prediction = load_susceptibility_pair(args.predictions / f"{sample_id}_prediction.npz")
    figure = build_figure(sample_id, truth, prediction, threshold=args.threshold,
                          label=args.result, metric_value=value,experiment=args.experiment)
    destination = args.output / f"{args.result}_{sample_id}_rotating_3d.gif"
    render_gif(figure, destination, args.frames, args.duration_seconds)
    print(f"Created: {destination}")


if __name__ == "__main__": main()

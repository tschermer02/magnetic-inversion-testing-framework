"""Run E01 dataset generation, training, prediction, and analysis."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _run(arguments: list[str]) -> None:
    print("\n>", " ".join(arguments), flush=True)
    subprocess.run(arguments, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="E01_soft_tversky", help="Experiment folder name.")
    parser.add_argument("--train-count", type=int, default=10)
    parser.add_argument("--validation-count", type=int, default=2)
    parser.add_argument("--test-count", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base-filters", type=int, default=8)
    parser.add_argument("--tmi-scale", type=float, default=100.0)
    parser.add_argument("--threshold-si", type=float, default=0.001)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    dataset = root / "datasets" / args.name
    training_output = root / "outputs" / args.name
    prediction_output = root / "prediction_outputs" / args.name
    if args.overwrite:
        for target, parent in (
            (training_output, root / "outputs"),
            (prediction_output, root / "prediction_outputs"),
        ):
            resolved_target = target.resolve()
            resolved_parent = parent.resolve()
            if resolved_target.parent != resolved_parent:
                raise RuntimeError(f"Refusing to remove unsafe output path: {resolved_target}")
            if resolved_target.exists():
                shutil.rmtree(resolved_target)
    prediction_output.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    generate = [python, "-m", "dataset_generation.generate_single_plane_dataset",
                "--output", str(dataset), "--train-count", str(args.train_count),
                "--validation-count", str(args.validation_count), "--test-count", str(args.test_count)]
    if args.overwrite:
        generate.append("--overwrite")
    _run(generate)
    _run([python, "-m", "cnn_inversion_3d.train", "--dataset", str(dataset),
          "--output", str(training_output), "--epochs", str(args.epochs),
          "--batch-size", str(args.batch_size), "--base-filters", str(args.base_filters),
          "--tmi-scale", str(args.tmi_scale)])

    # This evaluation pass performs prediction and forward-consistency
    # analysis for every test sample, without loading the model repeatedly.
    _run([python, "-m", "cnn_inversion_3d.evaluate", "--model", str(training_output / "e01.keras"),
          "--dataset", str(dataset), "--output", str(prediction_output),
          "--tmi-scale", str(args.tmi_scale), "--threshold-si", str(args.threshold_si)])
    print(f"\nE01 pipeline complete.\nModel: {training_output}\nPredictions and analysis: {prediction_output}")


if __name__ == "__main__":
    main()

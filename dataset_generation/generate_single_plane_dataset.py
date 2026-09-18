"""Generate the E02 log-uniform TMI/susceptibility dataset."""
from __future__ import annotations
import argparse, csv, json, shutil
from pathlib import Path
import numpy as np
from dataset_generation.e02_config import E02_DEFAULT_CONFIG, E02DatasetConfig
from forward_modeling.forward_model import TMIForwardModel, make_tensor_grid

SEED = 20260727
MODEL_SHAPE_ZYX = (24, 64, 64)

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--validation-count", type=int, default=100)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser

def sample_susceptibility(
    rng: np.random.Generator,
    config: E02DatasetConfig = E02_DEFAULT_CONFIG,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Sample one positive, induced-magnetization rectangular body."""
    width_x, width_y = (int(rng.integers(4, 17)) for _ in range(2))
    thickness = int(rng.integers(2, 9))
    top = int(rng.integers(2, min(17, 24 - thickness + 1)))
    x_start = int(rng.integers(8, 64 - 8 - width_x + 1))
    y_start = int(rng.integers(8, 64 - 8 - width_y + 1))
    susceptibility_log10 = float(rng.uniform(
        config.log10_susceptibility_min,
        config.log10_susceptibility_max,
    ))
    susceptibility_si = float(10.0 ** susceptibility_log10)
    model = np.zeros(MODEL_SHAPE_ZYX, dtype=np.float32)
    model[top:top+thickness, y_start:y_start+width_y, x_start:x_start+width_x] = susceptibility_si
    return model, {"x_start": x_start, "x_end": x_start+width_x, "y_start": y_start,
                   "y_end": y_start+width_y, "z_start": top, "z_end": top+thickness,
                   "susceptibility_si": susceptibility_si,
                   "susceptibility_log10": susceptibility_log10}

def build_generation_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    """Summarize realized E02 susceptibility and TMI distributions."""
    susceptibility = np.asarray([row["susceptibility_si"] for row in rows], dtype=np.float64)
    tmi_minimum = np.asarray([row["tmi_min_nt"] for row in rows], dtype=np.float64)
    tmi_maximum = np.asarray([row["tmi_max_nt"] for row in rows], dtype=np.float64)
    peak = np.asarray([row["peak_absolute_tmi_nt"] for row in rows], dtype=np.float64)
    return {
        "number_of_samples": len(rows),
        "split_sample_counts": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "validation", "test")
        },
        "actual_susceptibility_min_si": float(susceptibility.min()),
        "actual_susceptibility_max_si": float(susceptibility.max()),
        "susceptibility_decade_counts": {
            "0.0001_to_below_0.001_si": int(np.count_nonzero((susceptibility >= 1e-4) & (susceptibility < 1e-3))),
            "0.001_to_below_0.01_si": int(np.count_nonzero((susceptibility >= 1e-3) & (susceptibility < 1e-2))),
            "0.01_to_0.1_si_inclusive": int(np.count_nonzero((susceptibility >= 1e-2) & (susceptibility <= 1e-1))),
        },
        "global_tmi_min_nt": float(tmi_minimum.min()),
        "global_tmi_max_nt": float(tmi_maximum.max()),
        "peak_absolute_tmi_counts": {
            "below_1_nt": int(np.count_nonzero(peak < 1.0)),
            "1_to_below_10_nt": int(np.count_nonzero((peak >= 1.0) & (peak < 10.0))),
            "10_to_below_100_nt": int(np.count_nonzero((peak >= 10.0) & (peak < 100.0))),
            "100_to_below_1000_nt": int(np.count_nonzero((peak >= 100.0) & (peak < 1000.0))),
            "1000_nt_and_above": int(np.count_nonzero(peak >= 1000.0)),
        },
    }

def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def generate_dataset(output: Path, counts: tuple[int, int, int], seed: int, overwrite: bool) -> Path:
    if any(count < 1 for count in counts): raise ValueError("All split counts must be positive.")
    output = output.resolve()
    if output.exists():
        if not overwrite: raise FileExistsError(f"Dataset already exists: {output}")
        shutil.rmtree(output)
    samples_dir = output / "samples"; samples_dir.mkdir(parents=True)
    config = E02DatasetConfig(random_seed=seed)
    config.validate()
    survey = config.survey
    grid = make_tensor_grid([0,640,0,640,0,240], [10,10], 10)
    forward = TMIForwardModel(grid, survey.receiver_xyz, survey.field_strength_nt,
                              survey.inclination_deg, survey.declination_deg, survey.azimuth_deg)
    rng = np.random.default_rng(seed); rows: list[dict[str, object]] = []
    split_names = ("train", "validation", "test"); boundaries = np.cumsum(counts)
    total = sum(counts)
    model_batch_size = 16
    for batch_start in range(0, total, model_batch_size):
        batch_stop = min(batch_start + model_batch_size, total)
        sampled = [sample_susceptibility(rng, config) for _ in range(batch_stop - batch_start)]
        model_vectors = np.stack([
            np.transpose(model, (2, 1, 0)).ravel(order="F")
            for model, _ in sampled
        ])
        tmi_batch = forward.predict_many(model_vectors).reshape(
            -1, *survey.receiver_map_shape
        )
        for offset, ((susceptibility_zyx, body), tmi) in enumerate(zip(sampled, tmi_batch)):
            index = batch_start + offset
            sample_id = f"sample_{index:06d}"; relative_path = f"samples/{sample_id}.npz"
            np.savez_compressed(samples_dir/f"{sample_id}.npz", tmi=tmi.astype(np.float32), susceptibility=susceptibility_zyx)
            split = split_names[int(index >= boundaries[0]) + int(index >= boundaries[1])]
            rows.append({"sample_id": sample_id, "relative_path": relative_path, "split": split, **body,
                         "tmi_min_nt": float(tmi.min()), "tmi_max_nt": float(tmi.max()),
                         "peak_absolute_tmi_nt": float(np.max(np.abs(tmi)))})
        print(f"Generated {batch_stop}/{total} TMI samples", flush=True)
    for split in split_names: _write_manifest(output/f"{split}_manifest.csv", [r for r in rows if r["split"] == split])
    summary = build_generation_summary(rows)
    metadata = {"experiment_name":config.experiment_name, "dataset_version":config.dataset_version,
                "physics":"scalar induced magnetics", "observed_component":"TMI anomaly", "tmi_unit":"nT",
                "model_property":"magnetic susceptibility", "susceptibility_unit":"dimensionless SI",
                "susceptibility_sampling":{"distribution":config.susceptibility_distribution,
                    "log10_exponent_min":config.log10_susceptibility_min,
                    "log10_exponent_max":config.log10_susceptibility_max,
                    "minimum_si":config.susceptibility_min_si,"maximum_si":config.susceptibility_max_si,
                    "one_homogeneous_value_per_body":config.homogeneous_body_susceptibility},
                "tmi_input_normalization":{"operation":"divide","scale_nt":config.tmi_normalization_nt,
                    "clipping":False,"per_sample_normalization":False},
                "remanent_magnetization":config.remanent_magnetization,
                "observational_noise":config.observational_noise,
                "model_order":"z,y,x", "field_strength_nt":survey.field_strength_nt,
                "inclination_deg":survey.inclination_deg, "declination_deg":survey.declination_deg,
                "azimuth_deg":survey.azimuth_deg, "model_shape":list(MODEL_SHAPE_ZYX),
                "tmi_shape":list(survey.receiver_map_shape), "seed":seed,
                "generation_summary":summary}
    (output/"metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output/"generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("E02 generation summary")
    print(json.dumps(summary, indent=2), flush=True)
    return output

def main() -> None:
    args = build_argument_parser().parse_args(); root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root/args.output
    print(f"Generated TMI dataset: {generate_dataset(output,(args.train_count,args.validation_count,args.test_count),args.seed,args.overwrite)}")

if __name__ == "__main__": main()

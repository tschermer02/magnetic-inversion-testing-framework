"""Generate canonical TMI/susceptibility pairs for 3-D CNN inversion."""
from __future__ import annotations
import argparse, csv, json, shutil
from pathlib import Path
import numpy as np
from e01_magnetic.config import MagneticSurveyConfig
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

def sample_susceptibility(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, int | float]]:
    """Sample one positive, induced-magnetization rectangular body."""
    width_x, width_y = (int(rng.integers(4, 17)) for _ in range(2))
    thickness = int(rng.integers(2, 9))
    top = int(rng.integers(2, min(17, 24 - thickness + 1)))
    x_start = int(rng.integers(8, 64 - 8 - width_x + 1))
    y_start = int(rng.integers(8, 64 - 8 - width_y + 1))
    susceptibility_si = float(rng.uniform(0.001, 0.1))
    model = np.zeros(MODEL_SHAPE_ZYX, dtype=np.float32)
    model[top:top+thickness, y_start:y_start+width_y, x_start:x_start+width_x] = susceptibility_si
    return model, {"x_start": x_start, "x_end": x_start+width_x, "y_start": y_start,
                   "y_end": y_start+width_y, "z_start": top, "z_end": top+thickness,
                   "susceptibility_si": susceptibility_si}

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
    survey = MagneticSurveyConfig()
    grid = make_tensor_grid([0,640,0,640,0,240], [10,10], 10)
    forward = TMIForwardModel(grid, survey.receiver_xyz, survey.field_strength_nt,
                              survey.inclination_deg, survey.declination_deg, survey.azimuth_deg)
    rng = np.random.default_rng(seed); rows: list[dict[str, object]] = []
    split_names = ("train", "validation", "test"); boundaries = np.cumsum(counts)
    for index in range(sum(counts)):
        susceptibility_zyx, body = sample_susceptibility(rng)
        tmi = forward.predict(np.transpose(susceptibility_zyx, (2,1,0))).reshape(survey.receiver_map_shape)
        sample_id = f"sample_{index:06d}"; relative_path = f"samples/{sample_id}.npz"
        np.savez_compressed(samples_dir/f"{sample_id}.npz", tmi=tmi.astype(np.float32), susceptibility=susceptibility_zyx)
        split = split_names[int(index >= boundaries[0]) + int(index >= boundaries[1])]
        rows.append({"sample_id": sample_id, "relative_path": relative_path, "split": split, **body,
                     "tmi_min_nt": float(tmi.min()), "tmi_max_nt": float(tmi.max())})
    for split in split_names: _write_manifest(output/f"{split}_manifest.csv", [r for r in rows if r["split"] == split])
    metadata = {"physics":"scalar induced magnetics", "observed_component":"TMI", "tmi_unit":"nT",
                "model_property":"magnetic susceptibility", "susceptibility_unit":"dimensionless SI",
                "model_order":"z,y,x", "field_strength_nt":survey.field_strength_nt,
                "inclination_deg":survey.inclination_deg, "declination_deg":survey.declination_deg,
                "azimuth_deg":survey.azimuth_deg, "model_shape":list(MODEL_SHAPE_ZYX),
                "tmi_shape":list(survey.receiver_map_shape), "seed":seed}
    (output/"metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output

def main() -> None:
    args = build_argument_parser().parse_args(); root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root/args.output
    print(f"Generated TMI dataset: {generate_dataset(output,(args.train_count,args.validation_count,args.test_count),args.seed,args.overwrite)}")

if __name__ == "__main__": main()

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from e01_magnetic.config import E01Config
from e01_magnetic.physics import MagneticTMIModel
from forward_modeling.forward_model import make_tensor_grid


@dataclass(frozen=True)
class DiagnosticDataset:
    susceptibility: np.ndarray
    tmi: np.ndarray
    metadata: dict


def build_diagnostic_dataset(config: E01Config, *, sample_count: int = 3) -> list[tuple[np.ndarray, np.ndarray, dict]]:
    cfg = config
    cfg.validate()
    rng = np.random.default_rng(cfg.seed)
    grid = make_tensor_grid(bounds=[0.0, 640.0, 0.0, 640.0, 0.0, 240.0], xy_size=[10.0, 10.0], dz=10.0)
    survey = cfg.survey
    forward = MagneticTMIModel(
        grid,
        survey.receiver_xyz,
        field_strength_nt=survey.field_strength_nt,
        inclination_deg=survey.inclination_deg,
        declination_deg=survey.declination_deg,
        azimuth_deg=survey.azimuth_deg,
    )
    samples = []
    for _ in range(sample_count):
        model = rng.uniform(0.0, 0.2, size=(grid.x.size, grid.y.size, grid.z.size)).astype(np.float64)
        body_mask = np.zeros_like(model, dtype=bool)
        body_mask[1:4, 1:4, 1:3] = True
        body_model = np.zeros_like(model, dtype=np.float64)
        body_model[body_mask] = rng.uniform(0.05, 0.2, size=body_model[body_mask].shape)
        model = model + body_model
        flat = model.ravel(order="F")
        tmi = forward.predict(flat)
        samples.append((model, tmi, {"seed": cfg.seed, "grid_shape": grid.shape, "receiver_count": forward.n_receivers}))
    return samples


def generate_diagnostic_dataset(config: E01Config, *, output_dir: Path | str | None = None) -> Path:
    output_path = Path(output_dir) if output_dir is not None else Path(config.output_directory) / "diagnostic"
    output_path.mkdir(parents=True, exist_ok=True)
    samples = build_diagnostic_dataset(config, sample_count=3)
    for idx, (sus, tmi, metadata) in enumerate(samples):
        np.savez_compressed(output_path / f"sample_{idx:03d}.npz", susceptibility=sus, tmi=tmi, metadata=np.array([0], dtype=np.int32))
    return output_path

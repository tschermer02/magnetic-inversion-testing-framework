from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from forward_modeling.forward_model import TMIForwardModel, TensorGrid, make_tensor_grid


@dataclass(frozen=True)
class MagneticTensorGrid:
    """Convenience wrapper around the MATLAB-compatible grid logic used by TMIForwardModel."""

    grid: TensorGrid

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.grid.shape

    @property
    def n_cells(self) -> int:
        return self.grid.n_cells


class MagneticTMIModel(TMIForwardModel):
    """Thin wrapper around the supplied scalar induced-magnetics forward model."""

    def __init__(
        self,
        grid: TensorGrid,
        receiver_xyz: Sequence[Sequence[float]] | np.ndarray,
        *,
        field_strength_nt: float,
        inclination_deg: float,
        declination_deg: float,
        azimuth_deg: float = 0.0,
        receiver_chunk_size: int = 128,
    ) -> None:
        super().__init__(
            grid,
            receiver_xyz,
            field_strength_nt=field_strength_nt,
            inclination_deg=inclination_deg,
            declination_deg=declination_deg,
            azimuth_deg=azimuth_deg,
            receiver_chunk_size=receiver_chunk_size,
        )

    def sensitivity_matrix_chunked(self) -> np.ndarray:
        """Return the full dense TMI sensitivity matrix with MATLAB-compatible ordering."""
        rows = []
        for start in range(0, self.n_receivers, self.receiver_chunk_size):
            stop = min(start + self.receiver_chunk_size, self.n_receivers)
            kernel = self._kernel_chunk(self.receiver_xyz[start:stop])
            rows.append(kernel)
        return np.vstack(rows)

    def sensitivity_matrix(self) -> np.ndarray:
        return self.sensitivity_matrix_chunked()


def build_sensitivity_matrix(
    *,
    grid: TensorGrid,
    receiver_xyz: np.ndarray,
    field_strength_nt: float,
    inclination_deg: float,
    declination_deg: float,
    azimuth_deg: float = 0.0,
    receiver_chunk_size: int = 128,
) -> np.ndarray:
    forward = MagneticTMIModel(
        grid,
        receiver_xyz,
        field_strength_nt=field_strength_nt,
        inclination_deg=inclination_deg,
        declination_deg=declination_deg,
        azimuth_deg=azimuth_deg,
        receiver_chunk_size=receiver_chunk_size,
    )
    return forward.sensitivity_matrix()


def make_default_grid() -> TensorGrid:
    return make_tensor_grid(
        bounds=[0.0, 640.0, 0.0, 640.0, 0.0, 240.0],
        xy_size=[10.0, 10.0],
        dz=10.0,
    )

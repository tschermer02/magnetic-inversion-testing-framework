"""TMI-only Python translation of the supplied Joint3D MATLAB forward model.

This module translates the scalar induced-magnetics path formed by:

    getGpars.m -> getQuadPoints.m (pulse basis) -> getPredMag.m (channel 1)

The formulation treats every model cell as a susceptibility-weighted point
dipole at the cell center, multiplied by cell volume. It intentionally does
not replace the supplied MATLAB formulation with an exact rectangular-prism
integral.

Coordinate and unit conventions
-------------------------------
* X: east, Y: north, Z: positive down (left-handed system)
* Field inclination: positive down, in degrees
* Field declination and model/survey azimuth: degrees
* Cell and receiver coordinates: metres
* Susceptibility: dimensionless SI
* Inducing-field magnitude: nT
* Returned total magnetic intensity (TMI): nT

Model ordering
--------------
MATLAB ``ndgrid(x, y, z)`` followed by ``(:)`` makes X the fastest-changing
index, then Y, then Z. A 3-D NumPy model is therefore expected to have shape
``(nx, ny, nz)`` and is flattened with ``order="F"``.

The implementation is NumPy-only and receiver-chunked to control temporary
memory. For a fixed grid and survey, :class:`TMIForwardModel` can be reused for
many susceptibility models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TensorGrid:
    """Cell centers and dimensions matching the supplied MATLAB grid logic."""

    x: FloatArray
    y: FloatArray
    z: FloatArray
    dx: float
    dy: float
    dz: FloatArray

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.x.size, self.y.size, self.z.size)

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.shape))

    def quadrature(self) -> tuple[FloatArray, FloatArray]:
        """Return cell-center coordinates and pulse-basis cell volumes.

        This reproduces ``ndgrid(...); xg(:); yg(:); zg(:)`` and
        ``w = dx .* dy .* dzg`` from the MATLAB files.
        """

        xg, yg, zg = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        _, _, dzg = np.meshgrid(self.x, self.y, self.dz, indexing="ij")
        centers = np.column_stack(
            (
                xg.ravel(order="F"),
                yg.ravel(order="F"),
                zg.ravel(order="F"),
            )
        )
        volumes = self.dx * self.dy * dzg.ravel(order="F")
        return centers, volumes


def make_tensor_grid(
    bounds: Sequence[float],
    xy_size: Sequence[float],
    dz: float | ArrayLike,
) -> TensorGrid:
    """Translate ``getGpars.m`` for a Cartesian tensor grid.

    Parameters
    ----------
    bounds
        ``(xmin, xmax, ymin, ymax, zmin, zmax)`` in metres.
    xy_size
        ``(dx, dy)`` in metres.
    dz
        One constant vertical cell size or a sequence of layer thicknesses.
    """

    bounds_array = np.asarray(bounds, dtype=np.float64)
    xy_array = np.asarray(xy_size, dtype=np.float64)
    if bounds_array.shape != (6,):
        raise ValueError("bounds must contain [xmin, xmax, ymin, ymax, zmin, zmax]")
    if xy_array.shape != (2,) or np.any(xy_array <= 0.0):
        raise ValueError("xy_size must contain positive [dx, dy]")
    if not (bounds_array[1] > bounds_array[0] and bounds_array[3] > bounds_array[2]):
        raise ValueError("xmax and ymax must be greater than xmin and ymin")
    if bounds_array[5] <= bounds_array[4]:
        raise ValueError("zmax must be greater than zmin")

    dx, dy = float(xy_array[0]), float(xy_array[1])
    nx = int(np.floor((bounds_array[1] - bounds_array[0]) / dx))
    ny = int(np.floor((bounds_array[3] - bounds_array[2]) / dy))
    if nx < 1 or ny < 1:
        raise ValueError("bounds must contain at least one horizontal cell")
    x = bounds_array[0] + dx / 2.0 + dx * np.arange(nx, dtype=np.float64)
    y = bounds_array[2] + dy / 2.0 + dy * np.arange(ny, dtype=np.float64)

    dz_array = np.atleast_1d(np.asarray(dz, dtype=np.float64))
    if dz_array.ndim != 1 or dz_array.size == 0 or np.any(dz_array <= 0.0):
        raise ValueError("dz must be a positive scalar or one-dimensional sequence")

    if dz_array.size == 1:
        nz = int(np.floor((bounds_array[5] - bounds_array[4]) / dz_array[0]))
        if nz < 1:
            raise ValueError("bounds must contain at least one vertical cell")
        dz_array = np.full(nz, dz_array[0], dtype=np.float64)

    z = np.empty(dz_array.size, dtype=np.float64)
    z[0] = bounds_array[4] + dz_array[0] / 2.0
    if dz_array.size > 1:
        z[1:] = z[0] + np.cumsum((dz_array[:-1] + dz_array[1:]) / 2.0)

    keep = z <= bounds_array[5]
    z = z[keep]
    dz_array = dz_array[keep]
    if z.size == 0:
        raise ValueError("no vertical cell centers fall inside the bounds")

    return TensorGrid(x=x, y=y, z=z, dx=dx, dy=dy, dz=dz_array)


def inducing_field_direction(
    inclination_deg: float,
    declination_deg: float,
    azimuth_deg: float = 0.0,
) -> FloatArray:
    """Return ``[lx, ly, lz]`` exactly as calculated by ``getPredMag.m``."""

    inc = np.deg2rad(inclination_deg)
    dec_minus_az = np.deg2rad(declination_deg - azimuth_deg)
    return np.array(
        [
            np.cos(inc) * np.sin(dec_minus_az),
            np.cos(inc) * np.cos(dec_minus_az),
            np.sin(inc),
        ],
        dtype=np.float64,
    )


def _as_receiver_array(receiver_xyz: ArrayLike) -> FloatArray:
    receivers = np.asarray(receiver_xyz, dtype=np.float64)
    if receivers.ndim == 1:
        receivers = receivers.reshape(1, -1)
    if receivers.ndim != 2 or receivers.shape[1] != 3:
        raise ValueError("receiver_xyz must have shape (n_receivers, 3)")
    if not np.all(np.isfinite(receivers)):
        raise ValueError("receiver_xyz contains a non-finite value")
    return receivers


def _as_model_vector(susceptibility: ArrayLike, grid: TensorGrid) -> FloatArray:
    model = np.asarray(susceptibility, dtype=np.float64)
    if model.shape == grid.shape:
        vector = model.ravel(order="F")
    elif model.ndim == 1 and model.size == grid.n_cells:
        vector = model
    else:
        raise ValueError(
            f"susceptibility must have shape {grid.shape} or be a vector of "
            f"length {grid.n_cells}; received shape {model.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("susceptibility contains a non-finite value")
    return vector


class TMIForwardModel:
    """Reusable scalar induced-magnetics TMI forward operator.

    The TMI sensitivity for receiver ``i`` and model cell ``k`` is

    ``A[i,k] = B0*V[k]/(4*pi) * (3*(l dot r)^2/R^2 - 1)/R^3``.

    Here ``r`` points from the receiver to the cell center. This is a direct
    transcription of magnetic channel 1 in ``getPredMag.m``.
    """

    def __init__(
        self,
        grid: TensorGrid,
        receiver_xyz: ArrayLike,
        field_strength_nt: float,
        inclination_deg: float,
        declination_deg: float,
        azimuth_deg: float = 0.0,
        receiver_chunk_size: int = 128,
    ) -> None:
        if not np.isfinite(field_strength_nt):
            raise ValueError("field_strength_nt must be finite")
        if receiver_chunk_size < 1:
            raise ValueError("receiver_chunk_size must be at least 1")

        self.grid = grid
        self.receiver_xyz = _as_receiver_array(receiver_xyz)
        self.field_strength_nt = float(field_strength_nt)
        self.direction = inducing_field_direction(
            inclination_deg, declination_deg, azimuth_deg
        )
        self.receiver_chunk_size = int(receiver_chunk_size)
        self.cell_xyz, volumes = grid.quadrature()
        self.gamma = self.field_strength_nt * volumes / (4.0 * np.pi)

        # A receiver at a cell center makes the supplied MATLAB expression
        # singular. Catch it explicitly so the failure is understandable.
        for start in range(0, self.receiver_xyz.shape[0], self.receiver_chunk_size):
            rec = self.receiver_xyz[start : start + self.receiver_chunk_size]
            delta = self.cell_xyz[None, :, :] - rec[:, None, :]
            if np.any(np.einsum("rci,rci->rc", delta, delta) == 0.0):
                raise ValueError("a receiver coincides with a model-cell center")

    @property
    def n_receivers(self) -> int:
        return self.receiver_xyz.shape[0]

    def _kernel_chunk(self, receivers: FloatArray) -> FloatArray:
        delta = self.cell_xyz[None, :, :] - receivers[:, None, :]
        r_squared = np.einsum("rci,rci->rc", delta, delta)
        l_dot_r = np.einsum("rci,i->rc", delta, self.direction)
        return self.gamma[None, :] * (
            3.0 * l_dot_r**2 / r_squared - 1.0
        ) / (r_squared * np.sqrt(r_squared))

    def predict(self, susceptibility: ArrayLike) -> FloatArray:
        """Calculate one TMI value per receiver, in nT."""

        model = _as_model_vector(susceptibility, self.grid)
        tmi = np.empty(self.n_receivers, dtype=np.float64)
        for start in range(0, self.n_receivers, self.receiver_chunk_size):
            stop = min(start + self.receiver_chunk_size, self.n_receivers)
            kernel = self._kernel_chunk(self.receiver_xyz[start:stop])
            tmi[start:stop] = kernel @ model
        return tmi

    def predict_many(self, susceptibility_models: ArrayLike) -> FloatArray:
        """Calculate TMI for several flattened models.

        ``susceptibility_models`` must have shape ``(n_models, n_cells)``.
        Models use the same MATLAB-compatible X-fastest ordering as
        :meth:`predict`. To convert a 3-D model, use
        ``model.ravel(order="F")``.
        """

        models = np.asarray(susceptibility_models, dtype=np.float64)
        if models.ndim != 2 or models.shape[1] != self.grid.n_cells:
            raise ValueError(
                "susceptibility_models must have shape "
                f"(n_models, {self.grid.n_cells})"
            )
        if not np.all(np.isfinite(models)):
            raise ValueError("susceptibility_models contains a non-finite value")

        tmi = np.empty((models.shape[0], self.n_receivers), dtype=np.float64)
        for start in range(0, self.n_receivers, self.receiver_chunk_size):
            stop = min(start + self.receiver_chunk_size, self.n_receivers)
            kernel = self._kernel_chunk(self.receiver_xyz[start:stop])
            tmi[:, start:stop] = models @ kernel.T
        return tmi

    def sensitivity_matrix(self) -> FloatArray:
        """Build the full TMI Frechet matrix from ``getFrechetMag.m``.

        Warning: this requires ``8*n_receivers*n_cells`` bytes in float64.
        For large ML grids, prefer :meth:`predict` or :meth:`predict_many`,
        which construct only one receiver chunk at a time.
        """

        matrix = np.empty((self.n_receivers, self.grid.n_cells), dtype=np.float64)
        for start in range(0, self.n_receivers, self.receiver_chunk_size):
            stop = min(start + self.receiver_chunk_size, self.n_receivers)
            matrix[start:stop] = self._kernel_chunk(self.receiver_xyz[start:stop])
        return matrix


def forward_tmi(
    susceptibility: ArrayLike,
    grid: TensorGrid,
    receiver_xyz: ArrayLike,
    field_strength_nt: float,
    inclination_deg: float,
    declination_deg: float,
    azimuth_deg: float = 0.0,
    receiver_chunk_size: int = 128,
) -> FloatArray:
    """Convenience wrapper for a single scalar-susceptibility model."""

    operator = TMIForwardModel(
        grid=grid,
        receiver_xyz=receiver_xyz,
        field_strength_nt=field_strength_nt,
        inclination_deg=inclination_deg,
        declination_deg=declination_deg,
        azimuth_deg=azimuth_deg,
        receiver_chunk_size=receiver_chunk_size,
    )
    return operator.predict(susceptibility)


def compare_with_matlab_obsdata2(
    obsdata2_path: str,
    python_tmi: ArrayLike,
    receiver_xyz: ArrayLike,
    *,
    rtol: float = 1e-10,
    atol_nt: float = 1e-8,
) -> dict[str, float | bool]:
    """Compare Python TMI with a TMI-only MATLAB ``obsData2.dat`` file.

    Before running MATLAB, set ``rc{2}=1`` and ``compFlag=[0 1 0 0]`` in
    ``inpt3.m``. The resulting file should contain columns
    ``X, Y, Z, channel_code, TMI_nT``. Rows are sorted by coordinates here, so
    the comparison does not depend on the receiver ordering used by the caller.
    """

    matlab_data = np.loadtxt(obsdata2_path, ndmin=2)
    if matlab_data.shape[1] < 5:
        raise ValueError("obsData2.dat must contain at least five columns")
    if not np.all(matlab_data[:, 3] == 1):
        raise ValueError("obsData2.dat contains a magnetic channel other than TMI (1)")

    receivers = _as_receiver_array(receiver_xyz)
    predicted = np.asarray(python_tmi, dtype=np.float64).reshape(-1)
    if predicted.size != receivers.shape[0]:
        raise ValueError("python_tmi must have one value per receiver")

    python_data = np.column_stack((receivers, predicted))
    matlab_order = np.lexsort(
        (matlab_data[:, 2], matlab_data[:, 1], matlab_data[:, 0])
    )
    python_order = np.lexsort(
        (python_data[:, 2], python_data[:, 1], python_data[:, 0])
    )
    matlab_sorted = matlab_data[matlab_order]
    python_sorted = python_data[python_order]
    if matlab_sorted.shape[0] != python_sorted.shape[0]:
        raise ValueError("MATLAB and Python receiver counts differ")
    if not np.array_equal(matlab_sorted[:, :3], python_sorted[:, :3]):
        raise ValueError("MATLAB and Python receiver coordinates differ")

    residual = python_sorted[:, 3] - matlab_sorted[:, 4]
    return {
        "passed": bool(
            np.allclose(
                python_sorted[:, 3], matlab_sorted[:, 4], rtol=rtol, atol=atol_nt
            )
        ),
        "max_abs_error_nt": float(np.max(np.abs(residual))),
        "rms_error_nt": float(np.sqrt(np.mean(residual**2))),
    }


def matlab_example() -> tuple[TensorGrid, FloatArray, FloatArray]:
    """Run the magnetic slot from the supplied ``inpt3.m`` example.

    Returns
    -------
    grid, receivers, tmi
        The 20 x 20 x 5 model grid, 121 receiver coordinates, and TMI in nT.
    """

    grid = make_tensor_grid(
        bounds=(-100.0, 100.0, -100.0, 100.0, 50.0, 100.0),
        xy_size=(10.0, 10.0),
        dz=10.0,
    )
    receiver_x = np.tile(np.arange(-250.0, 251.0, 50.0), 11)
    receiver_y = np.repeat(np.arange(-250.0, 251.0, 50.0), 11)
    receivers = np.column_stack(
        (receiver_x, receiver_y, np.full(receiver_x.size, -30.0))
    )
    susceptibility = np.full(grid.shape, 0.06, dtype=np.float64)
    tmi = forward_tmi(
        susceptibility=susceptibility,
        grid=grid,
        receiver_xyz=receivers,
        field_strength_nt=50_000.0,
        inclination_deg=75.0,
        declination_deg=25.0,
        azimuth_deg=0.0,
    )
    return grid, receivers, tmi


if __name__ == "__main__":
    example_grid, example_receivers, example_tmi = matlab_example()
    print(f"Grid shape: {example_grid.shape} ({example_grid.n_cells} cells)")
    print(f"Receivers: {example_receivers.shape[0]}")
    print(f"TMI range: {example_tmi.min():.12g} to {example_tmi.max():.12g} nT")
    center = np.flatnonzero(
        (example_receivers[:, 0] == 0.0) & (example_receivers[:, 1] == 0.0)
    )[0]
    print(f"TMI at X=0, Y=0, Z=-30 m: {example_tmi[center]:.12g} nT")

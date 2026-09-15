# Magnetic inversion dataset

The canonical generator is `generate_single_plane_dataset.py`. It writes
`tmi` arrays in nT and `susceptibility` volumes in dimensionless SI units.
Receiver maps use `(y, x)` order; model volumes use `(z, y, x)` order. Before
calling the protected forward operator, volumes are transposed to `(x, y, z)`
as required by its MATLAB-compatible Fortran flattening.

Older files retain some gravity-named compatibility APIs for loading previous
experiments. They are not used by the canonical magnetic generator.

# E05 experiment suite

E05 tests stronger balanced-susceptibility supervision, supervised vertical
gradient matching, and a smaller absolute TMI-consistency term while retaining
the E01 dataset, architecture, SI/nT units, scaling, occupancy, optimizer,
learning rate, batch size, epoch budget, seed, and data order.

| CLI | Label | susceptibility weight | vertical weight | TMI weight |
|---|---|---:|---:|---:|
| e05a | E05a | 1 | 0 | 0 |
| e05b | E05b | 10 | 0 | 0 |
| e05c | E05c | 100 | 0 | 0 |
| e05d | E05d | selected parent | 0.1 | 0 |
| e05e | E05e | selected parent | 0 | 0.0001 |
| e05f | E05f | selected parent | 0.1 | 0.0001 |

The parent is frozen from validation only: rank E05a-c by lower true-body MAE
and higher support IoU, minimize rank sum, then break ties by lower body MAE and
lower susceptibility multiplier. Downstream models start from the same fresh
seed, never from the parent's checkpoint.

The vertical term is the per-sample mean absolute error between adjacent-Z
differences of `chi/0.1 SI`, followed by a batch mean. It matches supervised
boundaries rather than penalizing roughness alone. Adjacent-slice differences
assume the current uniform 10 m depth grid; no hidden spacing factor is used.

Every requested evaluation sample receives numerical amplitude, support, depth,
and TMI metrics. `--plots all` also produces fixed-coordinate susceptibility
slices and TMI comparisons with shared true/predicted color limits and separate
symmetric residual limits.

Full unattended workflow:

```bash
python -u -m cnn_inversion_3d.run_e05_suite --variants all --seed 20260727 --resume --plots all
```

Resume the same command after interruption. Run one independent first-stage
variant with `--variants e05b`; run only training with `--stage train`; evaluate
an existing checkpoint with `--stage analyze`. Requesting a dependent training
variant automatically resolves or runs the missing first-stage prerequisite.
An optional untouched final manifest can be supplied with
`--evaluation-manifest datasets/E01_soft_tversky_full/fresh_test_manifest.csv`;
its filename stem receives a separate output directory and never replaces the
existing test evaluation.

Outputs are separated under `outputs/E05/seed_20260727`,
`prediction_outputs/E05/seed_20260727`, and `analysis_outputs/E05/seed_20260727`.
Resume rejects incompatible configurations or split hashes. TensorFlow checkpoint
resume restores model and Adam state; the `tf.data` reshuffle sequence restarts
at a process boundary and is explicitly recorded.

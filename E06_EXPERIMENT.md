# E06 experiment suite

E06 starts from the actual E05E objective:

`100 L_chi + 2 L_depth + L_sensitivity + L_amplitude + 0.1 L_Tversky + 0.0001 L_TMI`.

There is no vertical-gradient term and no body-only auxiliary loss. Susceptibility
is represented in SI throughout; the `100` coefficient is an objective weight,
not an additional normalization. The architecture retains its sigmoid times
`0.1 SI` output, input TMI is divided by `100 nT`, and continuous susceptibility
reaches regression and forward-model losses unchanged.

| Variant | Body/background MSE | Tversky prediction occupancy |
|---|---:|---|
| E06A | 0.50 / 0.50 | E05E baseline-corrected sigmoid |
| E06B | 0.75 / 0.25 | E05E baseline-corrected sigmoid |
| E06C | 0.50 / 0.50 | `chi/(chi+0.001 SI)` |
| E06D | 0.75 / 0.25 | `chi/(chi+0.001 SI)` |

Both mappings retain the E05 truth mask `truth >= 0.001 SI`. Evaluation uses
the same `prediction >= 0.001 SI` support threshold. In Tversky, alpha `0.7`
multiplies false positives and beta `0.3` multiplies false negatives.

The rational mapping changes both occupancy shape and threshold calibration. It
has a nonzero susceptibility derivative at high values but the output sigmoid
still affects end-to-end gradients; it can reward larger in-body susceptibility
and contains no special vertical/horizontal distinction.

Every variant trains from the same reset seed. Primary selection and early
stopping minimize a fixed per-sample 50/50 validation balanced MAE:

`0.5 * MAE_body / 0.1 + 0.5 * MAE_background / 0.1`.

The geometry checkpoint maximizes validation IoU. Primary ties use higher IoU;
geometry ties use lower balanced MAE. Missing mask components contribute zero
and valid counts are recorded. Geometry errors for empty predictions remain
undefined and aggregate reports record valid and failure counts.

Full run:

```bash
python -u -m cnn_inversion_3d.run_e06_suite --variants all --seed 42 --resume --plots all
```

Resume uses the same command. Run one variant with `--variants e06b`; train only
with `--stage train`; evaluate existing checkpoints with `--stage analyze`.
Outputs are stored under `outputs/E06/seed_42`,
`prediction_outputs/E06/seed_42`, and `analysis_outputs/E06/seed_42`, separated
by variant and primary/geometry checkpoint.

E05C/E comparisons are valid only for identical test IDs and definitions, and
must be labeled as using the older body-only checkpoint-selection procedure.
Single-seed findings are preliminary.

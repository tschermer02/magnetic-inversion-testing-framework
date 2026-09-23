# E04 source-of-truth audit of E01

E04 is based only on the E01 implementation and `datasets/E01_soft_tversky_full`.
That dataset contains 1000 training, 100 validation, and 100 test samples and
uses seed 20260727. E02 and E03 data and losses are excluded.

## Data and geometry

- TMI input: 81 x 81 receivers, divided by the fixed 100 nT scale; no clipping
  and no per-sample normalization.
- Susceptibility target/output: 24 x 64 x 64 x 1, SI; targets are unscaled and
  the CNN sigmoid is multiplied by 0.1 SI.
- Model domain: X east 0-640 m, Y north 0-640 m, Z positive downward 0-240 m;
  10 m cubic cells. Forward vectors use X fastest, then Y, then Z.
- Receivers: 0-800 m east and north at 10 m spacing, elevation z=-10 m.
- Inducing field: 50,000 nT, inclination +75 degrees, declination 25 degrees,
  survey azimuth 0 degrees. Pure induced scalar susceptibility; TMI is nT.

## Architecture and training

The model is `build_e01_model(ModelConfig(base_filters=8))`: the existing
asymmetric 2-D U-Net with 81-to-96 padding, three encoder/pooling stages,
bottleneck, three transpose-convolution/skip decoder stages, a 33 x 33
96-to-64 spatial transform, and 24 sigmoid depth channels reshaped to the 3-D
volume. Training uses fresh seed/data-order seed 20260727, Adam at 1e-3, batch
size 2, at most 50 epochs, and validation-loss early stopping with patience 10
and minimum delta 1e-5.

## Exact active E01 objective

The active configuration in `cnn_inversion_3d.train` is:

- balanced true-body/background susceptibility MSE: coefficient 1;
- depth loss (depth-profile MSE plus normalized center-depth error): coefficient 2;
- integrated-sensitivity compensated MSE: coefficient 1, depth weights bounded
  0.5-5 with gamma 0.5 and mean one;
- true-body mean susceptibility-amplitude MSE: coefficient 1;
- legacy soft-Tversky: coefficient 0.1, threshold 0.001 SI, sharpness 1000,
  alpha 0.7 and beta 0.3;
- body-only auxiliary coefficient 0 and TMI coefficient 0.

The pulled E01 model directory contains the exported `e01.keras` but no matching
training history/run metadata. Dataset physics, shapes, and seed are recorded in
the dataset metadata; the remaining locked settings above come from the actual
E01 training entry point and command used for the full run.

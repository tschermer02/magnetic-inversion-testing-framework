# E04: controlled E01 plus magnetic consistency

Scientific question: does adding differentiable magnetic forward consistency
to the original E01 objective improve TMI-to-3-D-susceptibility inversion?

E04 retains every item audited in `E04_E01_BASELINE_AUDIT.md`. It initializes
from scratch and adds only

`L_E04 = L_E01 + lambda_TMI * L_TMI`,

where

`L_TMI = mean(((F(chi_pred) - TMI_obs_nt) / 100 nT)^2)`.

The normalized network input is multiplied by 100 nT exactly once. `F` is the
E01 point-dipole operator implemented entirely in TensorFlow, receiver-chunked,
and applied to unthresholded predicted SI susceptibility. There is no per-sample
normalization, RMS floor, relative loss, adaptive weighting, or gradient clipping.

Five stored E01 training models were reproduced with maximum TMI RMSE
2.42e-6 nT, maximum absolute residual 3.05e-5 nT, correlation effectively one,
and relative L2 error below 1.15e-7.

At fresh seeded initialization on fixed 10th/50th/90th-percentile training
susceptibilities, E01 loss/gradient norm were 0.137841/0.030089. Raw TMI
loss/gradient norm were 32.6618/13.6523. The fixed starting `lambda_TMI=0.001`
gives weighted values 0.032662/0.013652: influential but below the unchanged
E01 loss and gradient. This coefficient was frozen before validation and is
not optimized.

Train:

`python -u -m cnn_inversion_3d.train_e04 --dataset datasets/E01_soft_tversky_full --output outputs/E04`

Evaluate the same E01 test split:

`python -m cnn_inversion_3d.evaluate --model outputs/E04/e04.keras --dataset datasets/E01_soft_tversky_full --output prediction_outputs/E04 --tmi-scale 100 --threshold-si 0.001`

For the retained E01 export, run the same evaluator with
`outputs/E01_soft_tversky/e01.keras` and a separate output directory. Compare
the resulting `combined_test_metrics.csv` files; do not change thresholds.

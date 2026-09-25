# E05 audit of inherited E01 losses

E05 preserves the runtime E01 definitions below. All reductions first produce
one value per sample; the training wrapper then averages those values across the
batch. This prevents a smaller final batch from changing sample weighting.

- Balanced susceptibility MSE: within each sample, mean squared SI error is
  calculated separately over exact-positive true-body cells and background
  cells, then `0.5 * body_MSE + 0.5 * background_MSE`. E05a/b/c multiply only
  this scalar by 1/10/100; inputs are not divided by 0.1.
- Depth profile: sum continuous susceptibility over X/Y for each Z layer,
  normalize each profile by its own total, then mean squared profile difference.
- Center depth: susceptibility-weighted centers use layer centers 5,15,...,235 m;
  their difference is divided by 230 m and squared.
- Depth loss: profile loss plus `alpha_center=1` times center-depth loss. Its
  inherited total-loss coefficient is 2.
- Integrated sensitivity: weighted voxel MSE with fixed depth sensitivity
  weights, gamma 0.5, clipped to 0.5-5 and normalized to mean one.
- Amplitude: squared difference between true and predicted mean susceptibility
  inside the exact-positive true-body mask. Its coefficient is 1.
- Soft Tversky: truth is `chi_true >= 0.001 SI`; predicted occupancy is the
  baseline-corrected sigmoid with threshold 0.001 and sharpness 1000. Alpha 0.7
  multiplies false positives; beta 0.3 multiplies false negatives. Total weight 0.1.
- E04-style TMI, when enabled: mean squared physical-nT residual divided by
  `(100 nT)^2`, with unthresholded predictions. E05e/f coefficient is 0.0001.

Historical E01-E04 checkpointing monitored composite `val_loss`. E05 retains a
separate best-composite checkpoint only for diagnostics. The primary checkpoint
and early stopping use minimum per-sample validation true-body susceptibility
MAE, tie-broken by higher validation support IoU. Patience is 10 epochs.

For an empty true body, validation body MAE is defined as zero. Support follows:
both empty = 1; only one empty = 0. Generated E01 samples are nonempty.

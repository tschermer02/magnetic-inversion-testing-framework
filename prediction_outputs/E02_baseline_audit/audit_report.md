# E02 baseline audit

- 100 test samples; dataset truth and TMI exactly match saved prediction arrays.
- Corrected mean geological IoU=0.0815; mean symmetric 0.001 SI IoU=0.0731.
- Saved predictions have mean susceptibility MAE=0.000891078 SI and cached-forward TMI MAE=18.53 nT; zero predictions score 6.37172e-05 SI and 1.177 nT, respectively.
- 29 test bodies have susceptibility below 0.001 SI; the symmetric thresholded diagnostic correctly excludes them, while geological support retains them.
- 100 historical CSV rows have occupied-cell counts inconsistent with the current symmetric >0.001 SI evaluator applied to saved arrays. No historical threshold is inferred.
  Truth counts differ in 29 rows; predicted counts differ in 100 rows.
- Training inconsistency remains: Tversky target uses 0.001 SI whereas susceptibility/depth losses use all positive-susceptibility bodies. Training is unchanged.
- Checkpoint provenance: unverified_saved_prediction_arrays. Recorded best validation epoch 11 does not establish the source of saved predictions.
- Historical artifacts were not overwritten. Cached forward-TMI maps are reused when no checkpoint is available; their correspondence to saved prediction arrays remains unverified.
- Geological prediction threshold 0.00005 SI is provisional, not a detection limit and not test-optimized.

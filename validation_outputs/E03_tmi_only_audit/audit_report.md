# E02 validation evaluation audit

- 100 test samples evaluated from the dataset manifest; predictions were regenerated from the explicit checkpoint.
- Corrected mean geological IoU=0.0031; mean symmetric 0.001 SI IoU=0.0024.
- Regenerated best-checkpoint predictions have mean susceptibility MAE=0.0230352 SI and fresh-forward TMI MAE=275.1 nT; zero predictions score 9.07472e-05 SI and 1.709 nT, respectively.
- 26 validation bodies have susceptibility below 0.001 SI; the symmetric thresholded diagnostic correctly excludes them, while geological support retains them.
- 0 historical CSV rows have occupied-cell counts inconsistent with the current symmetric >0.001 SI evaluator applied to saved arrays. No historical threshold is inferred.
  Truth counts differ in 0 rows; predicted counts differ in 0 rows.
- Corrected training supervision used exact-positive geological truth and exponential soft predicted occupancy.
- Checkpoint provenance verified explicitly at recorded best validation epoch 2.
- Historical artifacts were not overwritten. TMI was freshly forward-modeled from regenerated checkpoint predictions.
- Geological prediction threshold 0.00005 SI is provisional, not a detection limit and not test-optimized.

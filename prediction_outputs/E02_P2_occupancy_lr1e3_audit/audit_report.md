# E02 evaluation audit

- 100 test samples were evaluated from the dataset manifest; predictions were regenerated from the explicit checkpoint.
- Corrected mean geological IoU=0.3441; mean symmetric 0.001 SI IoU=0.2698.
- Regenerated best-checkpoint predictions have mean susceptibility MAE=0.000260428 SI and fresh-forward TMI MAE=3.75 nT; zero predictions score 6.37172e-05 SI and 1.177 nT, respectively.
- 29 test bodies have susceptibility below 0.001 SI; the symmetric thresholded diagnostic correctly excludes them, while geological support retains them.
- 100 historical CSV rows have occupied-cell counts inconsistent with the current symmetric >0.001 SI evaluator applied to saved arrays. No historical threshold is inferred.
  Truth counts differ in 29 rows; predicted counts differ in 100 rows.
- Corrected training supervision used exact-positive geological truth and exponential soft predicted occupancy.
- Checkpoint provenance was verified explicitly at recorded best validation epoch 49.
- Historical artifacts were not overwritten. TMI was freshly forward-modeled from regenerated checkpoint predictions.
- Geological prediction threshold 0.00005 SI is provisional, not a detection limit and not test-optimized.

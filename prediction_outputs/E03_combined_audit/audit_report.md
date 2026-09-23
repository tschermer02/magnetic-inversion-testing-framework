# E02 test evaluation audit

- 100 test samples evaluated from the dataset manifest; predictions were regenerated from the explicit checkpoint.
- Corrected mean geological IoU=0.0047; mean symmetric 0.001 SI IoU=0.0041.
- Regenerated best-checkpoint predictions have mean susceptibility MAE=0.00715803 SI and fresh-forward TMI MAE=79.44 nT; zero predictions score 6.37172e-05 SI and 1.177 nT, respectively.
- 29 test bodies have susceptibility below 0.001 SI; the symmetric thresholded diagnostic correctly excludes them, while geological support retains them.
- 100 historical CSV rows have occupied-cell counts inconsistent with the current symmetric >0.001 SI evaluator applied to saved arrays. No historical threshold is inferred.
  Truth counts differ in 29 rows; predicted counts differ in 100 rows.
- Corrected training supervision used exact-positive geological truth and exponential soft predicted occupancy.
- Checkpoint provenance verified explicitly at recorded best validation epoch 2.
- Historical artifacts were not overwritten. TMI was freshly forward-modeled from regenerated checkpoint predictions.
- Geological prediction threshold 0.00005 SI is provisional, not a detection limit and not test-optimized.

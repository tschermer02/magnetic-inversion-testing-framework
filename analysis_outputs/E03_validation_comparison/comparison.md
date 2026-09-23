# E03 common validation metrics

Total validation loss is intentionally excluded because the ablations use different objectives.

| experiment | geological_iou | geological_dice | true_body_susceptibility_mae_si | background_susceptibility_mae_si | tmi_rmse_nt | tmi_correlation | predicted_to_observed_tmi_rms_ratio |
|---|---|---|---|---|---|---|---|
| E03_susceptibility_only_audit | 0.08025854444857805 | 0.1432441205553411 | 0.024411161251811662 | 0.0005532587626713689 | 26.501741646726558 | 0.636460777288433 | 173.26047342126833 |
| E03_tmi_only_audit | 0.0030979314984823153 | 0.006159574747561946 | 0.027181746630707684 | 0.023012571566044446 | 307.0636812924485 | 0.1728274707301668 | 3502.8591863521474 |
| E03_combined_audit | 0.00492211595803026 | 0.009770472431125547 | 0.01618283156206445 | 0.008718063266934542 | 114.7502570641821 | 0.24563482886251753 | 411.8531504332318 |

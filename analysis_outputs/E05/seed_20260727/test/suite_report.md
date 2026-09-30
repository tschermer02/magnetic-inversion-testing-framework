# E05 suite comparison

Split: `test`; 100 identical sample IDs; parent: `e05c`.

Composite validation losses are not compared because objectives differ.

| Variant | Metric | Mean | Median | Q10 | Q90 |
|---|---|---:|---:|---:|---:|
| e05a | true_body_susceptibility_mae_si | 0.028167 | 0.0274873 | 0.00630162 | 0.0501332 |
| e05a | center_depth_absolute_error_m | 29.5013 | 27.5534 | 3.8757 | 56.0611 |
| e05a | tmi_rmse_nt | 593.123 | 548.438 | 546.415 | 732.151 |
| e05b | true_body_susceptibility_mae_si | 0.028293 | 0.0272329 | 0.00616237 | 0.0511607 |
| e05b | center_depth_absolute_error_m | 19.1615 | 16.1938 | 3.44575 | 41.2007 |
| e05b | tmi_rmse_nt | 414.042 | 402.462 | 360.742 | 489.479 |
| e05c | true_body_susceptibility_mae_si | 0.0157315 | 0.013851 | 0.00497737 | 0.0297248 |
| e05c | center_depth_absolute_error_m | 9.16303 | 5.79901 | 1.01909 | 23.3619 |
| e05c | tmi_rmse_nt | 33.5165 | 24.7504 | 1.56603 | 77.0619 |
| e05d | true_body_susceptibility_mae_si | 0.0157785 | 0.014022 | 0.00531505 | 0.0297633 |
| e05d | center_depth_absolute_error_m | 7.18673 | 5.61626 | 1.54962 | 15.7276 |
| e05d | tmi_rmse_nt | 18.6535 | 10.4417 | 0.528451 | 52.1732 |
| e05e | true_body_susceptibility_mae_si | 0.0161363 | 0.0158638 | 0.00683998 | 0.0247828 |
| e05e | center_depth_absolute_error_m | 4.90572 | 3.4231 | 0.64626 | 11.3618 |
| e05e | tmi_rmse_nt | 10.1835 | 6.4442 | 1.50394 | 23.9081 |
| e05f | true_body_susceptibility_mae_si | 0.0249293 | 0.0263915 | 0.00562018 | 0.045241 |
| e05f | center_depth_absolute_error_m | 36.6083 | 35 | 3.81933 | 69.0555 |
| e05f | tmi_rmse_nt | 492.096 | 492.605 | 489.424 | 494.044 |

Interpret amplitude, geometry, depth fragmentation and magnetic consistency separately; do not declare improvement from mean relative L2 alone.

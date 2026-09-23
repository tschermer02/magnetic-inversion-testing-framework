# E03 amplitude-aware and TMI-consistent inversion

E03 keeps the E02 dataset, P2 geological occupancy, U-Net, fixed 100 nT input
scale, Adam learning rate 1e-3, batch size 2, seed 20260727, and early stopping.
No ablation is warm-started.

For each sample, `s=max(mean(true susceptibility in the generated body),1e-4 SI)`.
The new susceptibility term is the equally body/background-balanced mean of
`((prediction-truth)/s)^2`. Its coefficient is `1e-6`; the existing absolute
susceptibility coefficient remains 1. This deliberately makes leakage more
expensive for weak bodies and can create large gradients for badly
overestimated weak samples.

The differentiable operator implements the dataset's point-dipole matrix in
TensorFlow, in physical nT, with receiver chunking and no susceptibility
threshold. Observed data are restored exactly once by multiplying normalized
inputs by 100 nT. The absolute TMI term uses `(prediction-observation)/100 nT`.
The relative term uses the larger of observed RMS and `0.03296747710542477 nT`.
That floor is the 10th percentile of positive RMS values in the 1000-sample
E02 training split; it is a numerical weighting floor, not instrument noise.

The training-only, seeded three-decade diagnostic at fresh initialization gave:

| term | raw loss | raw gradient norm | coefficient | weighted loss |
|---|---:|---:|---:|---:|
| relative susceptibility | 9512.86 | 2807.79 | 1e-6 | 0.00951 |
| absolute TMI | 32.7795 | 8.18596 | 1e-4 | 0.00328 |
| relative TMI | 17543426 | 4373919 | 1e-9 | 0.01754 |

These are conservative frozen starting coefficients, not validation-optimized
values. A P2-checkpoint diagnostic was not possible locally because its weights
were unavailable. There is no gradient clipping, adaptive weighting, or learning
rate change.

The combined objective is

`L = E_abs + 1e-6 E_rel + 2 L_depth + L_sensitivity + L_amplitude`
`    + 0.1 L_Tversky + 1e-4 E_TMI_abs + 1e-9 E_TMI_rel`.

`E03_susceptibility_only`, `E03_tmi_only`, and `E03_combined` selectively enable
the shared coefficients above. `E02_P2_control` disables all three new terms.
Run `python -m evaluation.diagnose_e03_scales` to reproduce the saved diagnostic.
Compare ablations using common validation metrics, not their different total losses.

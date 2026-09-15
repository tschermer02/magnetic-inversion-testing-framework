# E01 magnetic inversion

This directory contains one architecture and one loss framework: **E01**, the
former E09B-12 asymmetric 2-D U-Net with depth, integrated-sensitivity,
amplitude, body-support, optional TMI-forward, and soft-Tversky terms.

Data use `tmi` in nT with shape `(81,81,1)` and `susceptibility` in
dimensionless SI with shape `(24,64,64,1)`. Run `python -m
cnn_inversion_3d.train --help` for the consolidated training entry point.

Use `python -m cnn_inversion_3d.analyze --help` to evaluate susceptibility
recovery and forward-model consistency in TMI space.

`python run_e01_pipeline.py` runs generation, training, prediction, and
analysis for every test sample. Results follow the established experiment
layout beneath `prediction_outputs/E01_soft_tversky/`.

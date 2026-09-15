# E01 magnetic inversion

This directory contains one architecture and one loss framework: **E01**, the
former E09B-12 asymmetric 2-D U-Net with depth, integrated-sensitivity,
amplitude, body-support, optional TMI-forward, and soft-Tversky terms.

Data use `tmi` in nT with shape `(81,81,1)` and `susceptibility` in
dimensionless SI with shape `(24,64,64,1)`. Run `python -m
cnn_inversion_3d.train --help` for the consolidated training entry point.

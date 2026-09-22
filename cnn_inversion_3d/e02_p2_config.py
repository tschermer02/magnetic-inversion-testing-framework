"""Controlled Priority-2 E02 occupancy-supervision experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class E02P2RunConfig:
    name: str
    learning_rate: float
    occupancy_mode: str = "geological_exponential"
    occupancy_tau_si: float = 1.0e-4 / math.log(100.0)
    initialization_seed: int = 20260727
    data_order_seed: int = 20260727
    batch_size: int = 2
    epochs: int = 50
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1.0e-5
    base_filters: int = 8
    tmi_scale_nt: float = 100.0
    lambda_depth: float = 2.0
    lambda_amplitude: float = 1.0
    lambda_tmi: float = 0.0
    lambda_tversky: float = 0.1
    tversky_alpha: float = 0.7
    tversky_beta: float = 0.3
    evaluation_prediction_threshold_si: float = 0.00005

    def to_dict(self) -> dict:
        return asdict(self)


RUNS = {
    "E02_P2_occupancy_lr1e3": E02P2RunConfig("E02_P2_occupancy_lr1e3", 1.0e-3),
    "E02_P2_occupancy_lr1e4": E02P2RunConfig("E02_P2_occupancy_lr1e4", 1.0e-4),
}


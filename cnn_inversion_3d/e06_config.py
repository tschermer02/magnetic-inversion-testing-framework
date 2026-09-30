"""Locked definitions for the controlled E06 experiment suite."""
from __future__ import annotations
from dataclasses import asdict,dataclass,replace
from pathlib import Path

@dataclass(frozen=True)
class E06Variant:
    identifier:str
    label:str
    body_fraction:float
    occupancy_mode:str
    def to_dict(self):return asdict(self)

VARIANTS={
    "e06a":E06Variant("e06a","E06A",.50,"legacy_threshold_sigmoid"),
    "e06b":E06Variant("e06b","E06B",.75,"legacy_threshold_sigmoid"),
    "e06c":E06Variant("e06c","E06C",.50,"rational"),
    "e06d":E06Variant("e06d","E06D",.75,"rational"),
}

@dataclass(frozen=True)
class E06SuiteConfig:
    dataset:Path=Path("datasets/E01_soft_tversky_full")
    seed:int=42
    batch_size:int=2
    epochs:int=50
    learning_rate:float=1e-3
    base_filters:int=8
    tmi_scale_nt:float=100.
    susceptibility_scale_si:float=1.
    support_threshold_si:float=.001
    early_stopping_patience:int=10
    diagnostic_interval:int=5
    diagnostic_samples:int=1
    def to_dict(self):
        value=asdict(self);value["dataset"]=str(self.dataset);return value

def with_seed(config:E06SuiteConfig,seed:int):return replace(config,seed=seed)

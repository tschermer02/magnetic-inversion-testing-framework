"""Locked definitions for the controlled E05 experiment suite."""
from __future__ import annotations
from dataclasses import asdict,dataclass,replace
from pathlib import Path

@dataclass(frozen=True)
class E05Variant:
    identifier:str
    label:str
    susceptibility_multiplier:float
    vertical_gradient_coefficient:float
    tmi_coefficient:float
    parent_identifier:str|None=None
    def to_dict(self): return asdict(self)

FIRST_STAGE={
    "e05a":E05Variant("e05a","E05a",1,0,0),
    "e05b":E05Variant("e05b","E05b",10,0,0),
    "e05c":E05Variant("e05c","E05c",100,0,0),
}

def dependent_variants(parent:E05Variant):
    common={"susceptibility_multiplier":parent.susceptibility_multiplier,
            "parent_identifier":parent.identifier}
    return {
        "e05d":E05Variant("e05d","E05d",vertical_gradient_coefficient=.1,tmi_coefficient=0,**common),
        "e05e":E05Variant("e05e","E05e",vertical_gradient_coefficient=0,tmi_coefficient=1e-4,**common),
        "e05f":E05Variant("e05f","E05f",vertical_gradient_coefficient=.1,tmi_coefficient=1e-4,**common),
    }

@dataclass(frozen=True)
class E05SuiteConfig:
    dataset:Path=Path("datasets/E01_soft_tversky_full")
    seed:int=20260727
    batch_size:int=2
    epochs:int=50
    learning_rate:float=1e-3
    base_filters:int=8
    tmi_scale_nt:float=100
    susceptibility_scale_si:float=1
    output_maximum_si:float=.1
    support_threshold_si:float=.001
    early_stopping_patience:int=10
    diagnostic_interval:int=5
    diagnostic_samples:int=1
    def to_dict(self):
        result=asdict(self);result["dataset"]=str(self.dataset);return result

def with_seed(config:E05SuiteConfig,seed:int): return replace(config,seed=seed)

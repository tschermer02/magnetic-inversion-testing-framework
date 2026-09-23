"""Locked E04 configuration: E01 plus one absolute TMI-consistency term."""
from __future__ import annotations
from dataclasses import asdict,dataclass
from pathlib import Path

@dataclass(frozen=True)
class E04Config:
    experiment_name:str="E04"
    dataset_directory:Path=Path("datasets/E01_soft_tversky_full")
    seed:int=20260727
    batch_size:int=2
    epochs:int=50
    early_stopping_patience:int=10
    early_stopping_min_delta:float=1e-5
    base_filters:int=8
    learning_rate:float=1e-3
    tmi_scale_nt:float=100.0
    susceptibility_scale_si:float=1.0
    lambda_tmi:float=1e-3
    def to_dict(self):
        result=asdict(self); result["dataset_directory"]=str(self.dataset_directory); return result

E04_CONFIG=E04Config()

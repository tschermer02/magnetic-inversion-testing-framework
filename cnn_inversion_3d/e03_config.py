"""Frozen starting configurations for the controlled E03 ablations."""
from __future__ import annotations
from dataclasses import asdict,dataclass
import math

@dataclass(frozen=True)
class E03RunConfig:
    name:str
    lambda_susceptibility_relative:float
    lambda_tmi_absolute:float
    lambda_tmi_relative:float
    susceptibility_floor_si:float=1e-4
    tmi_rms_floor_nt:float=0.03296747710542477
    tmi_floor_derivation:str="10th percentile of positive RMS(TMI) over 1000 E02 training samples"
    tmi_reference_nt:float=100.0
    occupancy_mode:str="geological_exponential"
    occupancy_tau_si:float=1e-4/math.log(100)
    learning_rate:float=1e-3
    seed:int=20260727
    batch_size:int=2
    epochs:int=50
    early_stopping_patience:int=10
    early_stopping_min_delta:float=1e-5
    base_filters:int=8
    tmi_scale_nt:float=100.0
    evaluation_prediction_threshold_si:float=5e-5
    def to_dict(self): return asdict(self)

# Conservative training-only starting weights. Raw-scale diagnostics are saved
# by ``diagnose_e03_scales`` before full training; these values are not tuned on validation/test.
RELATIVE_WEIGHT=1e-6
TMI_ABSOLUTE_WEIGHT=1e-4
# The training-only scale diagnostic measured a raw initial loss of 1.754e7 and
# gradient norm of 4.374e6.  At 1e-9 these become 1.754e-2 and 4.374e-3,
# respectively, so the relative term is useful without dominating P2.
TMI_RELATIVE_WEIGHT=1e-9

RUNS={
 "E02_P2_control":E03RunConfig("E02_P2_control",0,0,0),
 "E03_susceptibility_only":E03RunConfig("E03_susceptibility_only",RELATIVE_WEIGHT,0,0),
 "E03_tmi_only":E03RunConfig("E03_tmi_only",0,TMI_ABSOLUTE_WEIGHT,TMI_RELATIVE_WEIGHT),
 "E03_combined":E03RunConfig("E03_combined",RELATIVE_WEIGHT,TMI_ABSOLUTE_WEIGHT,TMI_RELATIVE_WEIGHT),
}

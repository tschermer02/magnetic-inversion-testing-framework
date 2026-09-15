"""Public compatibility import for the repository's TMI forward model.

The implementation lives in :mod:`forward_modeling.forward_model`; keeping
this tiny module avoids duplicating or modifying that implementation.
"""

from forward_modeling.forward_model import (
    TMIForwardModel,
    TensorGrid,
    compare_with_matlab_obsdata2,
    forward_tmi,
    inducing_field_direction,
    make_tensor_grid,
    matlab_example,
)

__all__ = [
    "TMIForwardModel",
    "TensorGrid",
    "compare_with_matlab_obsdata2",
    "forward_tmi",
    "inducing_field_direction",
    "make_tensor_grid",
    "matlab_example",
]

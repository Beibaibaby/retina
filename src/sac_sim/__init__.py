"""
Two-SAC retinal microcircuit – generative model.

Import as:
    from sac_sim.core   import run_two_SAC_simulation
    from sac_sim.params import BASE_PARAMS, make_shifted
"""
from .params import BASE_PARAMS, make_shifted        # noqa: F401
from .core   import (                               # noqa: F401
    precompute_bipolar_inputs,
    run_two_SAC_simulation,
    detect_spikes,
)


"""Run many Monte-Carlo repeats of the model."""
from __future__ import annotations
import numpy as np
from sac_sim.params import BASE_PARAMS, make_shifted
from sac_sim.core   import precompute_bipolar_inputs, run_two_SAC_simulation


def _single(trial_params, bip_data, inhibit):
    _, V1, Ca1, V2, Ca2, IPSC, *_ = run_two_SAC_simulation(
        inhibit_connection=inhibit,
        params=trial_params,
        bipolar_data=bip_data,
    )
    return V1, Ca1, V2, Ca2, IPSC


def run_batch(N=20, *, seed=0):
    """
    Return (t_vector, summary_dict) where summary_dict[condition][trace_key]
    is (mean, std)   with trace_key ∈ {"V1","Ca1","V2","Ca2","IPSC"}.
    """
    rng = np.random.default_rng(seed)
    holders = {c: [] for c in ("baseline-intact", "baseline-cut", "shift-intact")}

    for _ in range(N):
        t, bip1, bip2 = precompute_bipolar_inputs(BASE_PARAMS)
        bip = {"SAC1": bip1, "SAC2": bip2}

        # baseline-intact
        holders["baseline-intact"].append(_single(BASE_PARAMS, bip, True))
        # baseline-cut
        holders["baseline-cut"].append(_single(BASE_PARAMS, bip, False))
        # shift-intact
        p_shift = make_shifted(4.0)
        holders["shift-intact"].append(_single(p_shift, bip, True))

        rng.random()      # move RNG state

    # aggregate
    summary = {}
    for cond, runs in holders.items():
        V1s, Ca1s, V2s, Ca2s, IPSCs = map(np.stack, zip(*runs))
        summary[cond] = {
            "V1":   (V1s.mean(0),   V1s.std(0)),
            "Ca1":  (Ca1s.mean(0),  Ca1s.std(0)),
            "V2":   (V2s.mean(0),   V2s.std(0)),
            "Ca2":  (Ca2s.mean(0),  Ca2s.std(0)),
            "IPSC": (IPSCs.mean(0), IPSCs.std(0)),
        }
    return t, summary

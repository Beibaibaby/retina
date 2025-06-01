#!/usr/bin/env python3
"""
Quick visual sanity check – reproduces the 3×3 grid for one run.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from sac_sim.params import BASE_PARAMS, make_shifted
from sac_sim.core   import precompute_bipolar_inputs, run_two_SAC_simulation

def _run(params, bip_data, inhibit):
    return run_two_SAC_simulation(inhibit_connection=inhibit,
                                  params=params,
                                  bipolar_data=bip_data)

if __name__ == "__main__":
    t, bip1, bip2 = precompute_bipolar_inputs(BASE_PARAMS)
    bip = {"SAC1": bip1, "SAC2": bip2}

    # three conditions
    resA_int = _run(BASE_PARAMS, bip, True)
    resA_cut = _run(BASE_PARAMS, bip, False)
    resB_int = _run(make_shifted(4.0), bip, True)

    fig, axs = plt.subplots(3, 3, figsize=(16, 10), sharex=True, sharey='row')
    fig.suptitle("Single run – baseline & threshold shift", fontsize=12)

    for col, (res, title) in enumerate(
        [(resA_int, "Baseline – intact"),
         (resA_cut, "Baseline – cut"),
         (resB_int, "Shift – intact")]):
        t, V1, Ca1, V2, Ca2, IPSC, *_ = res
        axs[0, col].plot(t, V1, 'b-', label='V1')
        axs[0, col].plot(t, Ca1, 'r-', label='Ca1')
        axs[0, col].set_title(f"{title}: SAC1")
        axs[0, col].legend(fontsize=8)

        axs[1, col].plot(t, V2, 'b-', label='V2')
        axs[1, col].plot(t, Ca2, 'r-', label='Ca2')
        axs[1, col].set_title(f"{title}: SAC2")
        axs[1, col].legend(fontsize=8)

        axs[2, col].plot(t, IPSC, 'k-', label='IPSC')
        axs[2, col].set_title(f"{title}: DSGC")
        axs[2, col].legend(fontsize=8)
        axs[2, col].set_xlabel("Time (ms)")

    axs[1,0].set_ylabel("Voltage / Ca (a.u.)")
    axs[2,0].set_ylabel("IPSC (a.u.)")
    plt.tight_layout()
    plt.show()

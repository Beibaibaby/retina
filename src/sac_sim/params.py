"""Default parameters and simple helpers."""

# ------------------ base dictionary ------------------
BASE_PARAMS = {
    # Simulation timing
    "t_max": 8000.0,
    "dt": 1.0,

    # Bipolar & Noise
    "n_bipolars": 10,
    "active_bipolars": 5,
    "refresh_rate": 15,
    "I_exc_per_bipolar": 0.08,

    # Passive membrane
    "C_mem": 1.0,
    "g_leak": 0.05,
    "E_leak": -70.0,

    # HH-like gating
    "threshold_base": -57.5,
    "threshold_shift_LY": 0.0,      # (δV you’ll override)
    "k_gate": 0.8,
    "tau_m": 10.0,
    "I_Ca_in": 0.2,

    # Calcium dynamics
    "tau_Ca": 200.0,
    "plot_range": (-80, 50, -1.0, 50.0),

    # Disinhibitory circuit
    "I_inh_strength": 0.18,
    "spike_threshold": 4,
    "soft_inh_threshold": 30.0,
    "inh_slope": 0.3,

    # Short-term plasticity SAC1→SAC2
    "SAC1toSAC2_U": 0.02,
    "SAC1toSAC2_tau_rec": 50.0,

    # Short-term plasticity SAC2→DSGC
    "tau_rec": 1000.0,
    "U": 0.10,
    "w_syn0": 1.0,

    # Dual-exponential kernel
    "tau_rise": 20.0,
    "tau_decay": 100.0,
    "t_kernel": 150.0,

    # Probabilistic release factor
    "k_release": 0.005,

    # Post-release non-linearity
    "post_nlin_threshold": 30.0,
    "post_nlin_scale_below": 0.5,
}


# ------------------ convenience helper ------------------
def make_shifted(delta_v: float):
    """Return a *new* param-dict with threshold shifted by `delta_v` mV."""
    p = dict(BASE_PARAMS)
    p["threshold_shift_LY"] = delta_v
    return p

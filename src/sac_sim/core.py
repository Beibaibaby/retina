"""
Core numerical engine for the two-SAC model.
Pure functions only – no plotting, no CLI.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks
from .params import BASE_PARAMS

# ------------------------------------------------------------------ helpers
def m_inf(V: float, V_half: float, k_gate: float) -> float:
    """Steady-state activation (logistic)."""
    return 1.0 / (1.0 + np.exp(-(V - V_half) / k_gate))


# --------------------- bipolar input generator ----------------------------
def _get_bipolar_count(cell: str, t: float, p: dict) -> int:
    """Time-dependent drive profile (piece-wise)."""
    if 0 <= t < 2000:
        return int(np.clip(round(np.random.normal(5.0, 5.0)), 0, p["n_bipolars"]))
    elif 2000 <= t < 4000:
        return p["n_bipolars"] if cell == "SAC1" else \
               int(np.clip(round(np.random.normal(5.0, 5.0)), 0, p["n_bipolars"]))
    elif 4000 <= t < 6000:
        return p["n_bipolars"] if cell == "SAC2" else \
               int(np.clip(round(np.random.normal(5.0, 5.0)), 0, p["n_bipolars"]))
    else:
        return int(np.clip(round(np.random.normal(5.0, 5.0)), 0, p["n_bipolars"]))


def precompute_bipolar_inputs(p: dict = BASE_PARAMS):
    """Return (t_arr, bip_SAC1, bip_SAC2) arrays."""
    dt = p["dt"]
    steps = int(p["t_max"] // dt)
    t_arr = np.arange(0, p["t_max"], dt)

    bip1 = np.zeros(steps)
    bip2 = np.zeros(steps)

    frame_period      = 1000.0 / p["refresh_rate"]
    frame_step_count  = int(frame_period // dt)

    for i, t in enumerate(t_arr):
        if i % frame_step_count == 0:
            bip1[i] = _get_bipolar_count("SAC1", t, p)
            bip2[i] = _get_bipolar_count("SAC2", t, p)
        else:
            bip1[i] = bip1[i - 1]
            bip2[i] = bip2[i - 1]

    return t_arr, bip1, bip2


# --------------------------- main integrator ------------------------------
def run_two_SAC_simulation(*,
                           inhibit_connection: bool = True,
                           params: dict = BASE_PARAMS,
                           bipolar_data: dict | None = None):
    """
    Simulate SAC1 & SAC2 membrane voltage / Ca²⁺ plus downstream DSGC IPSC.
    Returns the full set of traces.
    """
    p   = params
    dt  = p["dt"]
    t_arr, bip1, bip2 = (precompute_bipolar_inputs(p)
                         if bipolar_data is None
                         else (None,
                               bipolar_data["SAC1"],
                               bipolar_data["SAC2"]))

    steps = len(bip1)
    # allocate
    V1  = np.zeros(steps);  V2  = np.zeros(steps)
    Ca1 = np.zeros(steps);  Ca2 = np.zeros(steps)
    m1  = np.zeros(steps);  m2  = np.zeros(steps)

    # initial conditions
    V1[0] = V2[0] = p["E_leak"]

    # resource variables
    R_syn12        = 1.0                # SAC1→SAC2 depression
    R_syn12_trace  = np.zeros(steps)
    R_syn12_trace[0] = R_syn12

    # cached constants
    V_half = p["threshold_base"] + p["threshold_shift_LY"]

    for i in range(steps - 1):
        # ---------- excitatory drive ----------
        I_exc1 = bip1[i] * p["I_exc_per_bipolar"]
        I_exc2 = bip2[i] * p["I_exc_per_bipolar"]

        # ---------- SAC1 ----------
        V1[i+1] = V1[i] + dt/p["C_mem"] * (-p["g_leak"]*(V1[i]-p["E_leak"]) + I_exc1)
        m1_inf  = m_inf(V1[i], V_half, p["k_gate"])
        m1[i+1] = m1[i] + dt*(m1_inf - m1[i])/p["tau_m"]
        Ca1[i+1]= Ca1[i] + dt*(-Ca1[i]/p["tau_Ca"] + p["I_Ca_in"]*m1[i])

        # ---------- Ca-gated inhibition ----------
        if inhibit_connection:
            I_inh_base = p["I_inh_strength"] / (1 + np.exp(-(Ca1[i]-p["soft_inh_threshold"])/p["inh_slope"]))
            I_inh = I_inh_base * R_syn12
        else:
            I_inh_base = I_inh = 0.0

        # ---------- SAC2 ----------
        I_tot2   = I_exc2 - I_inh
        V2[i+1]  = V2[i] + dt/p["C_mem"] * (-p["g_leak"]*(V2[i]-p["E_leak"]) + I_tot2)
        m2_inf   = m_inf(V2[i], V_half, p["k_gate"])
        m2[i+1]  = m2[i] + dt*(m2_inf - m2[i])/p["tau_m"]
        Ca2[i+1] = Ca2[i] + dt*(-Ca2[i]/p["tau_Ca"] + p["I_Ca_in"]*m2[i])

        # ---------- STD on SAC1→SAC2 ----------
        dR12 = (1 - R_syn12)/p["SAC1toSAC2_tau_rec"] - p["SAC1toSAC2_U"]*R_syn12*I_inh_base
        R_syn12 = np.clip(R_syn12 + dt*dR12, 0.0, 1.0)
        R_syn12_trace[i+1] = R_syn12

    # ---------- SAC2→DSGC release ----------
    release_raw = np.zeros(steps)
    R_syn = 1.0
    R_trace = np.zeros(steps)
    for i in range(steps):
        p_rel = p["k_release"] * Ca2[i] * dt
        if np.random.rand() < p_rel:
            release_raw[i] = p["w_syn0"] * R_syn
            R_syn *= (1 - p["U"])
        R_syn += dt*(1-R_syn)/p["tau_rec"]
        R_syn = min(R_syn, 1.0)
        R_trace[i] = R_syn

    # post-nonlinearity
    release_final = release_raw.copy()
    below = Ca2 < p["post_nlin_threshold"]
    release_final[below] *= p["post_nlin_scale_below"]

    # ---------- convolution kernel ----------
    def _dual_exp_kernel(dt, t_max, tau_rise, tau_decay):
        t_k = np.arange(0, t_max, dt)
        k   = np.exp(-t_k/tau_decay) - np.exp(-t_k/tau_rise)
        return k / k.max()

    kernel = _dual_exp_kernel(dt, p["t_kernel"], p["tau_rise"], p["tau_decay"])
    IPSC   = np.convolve(release_final, kernel, mode="full")[:steps]

       # --- guarantee we always return a usable t-vector ---
    if t_arr is None:                       # happened because caller passed bipolar_data
        steps = int(p["t_max"] // p["dt"])
        t_arr = np.arange(0, p["t_max"], p["dt"])

    return (
        t_arr,          # <-- now never None
        V1, Ca1,
        V2, Ca2,
        IPSC,
        R_trace,
        R_syn12_trace,
    )



# --------------------------- simple spike-finder ---------------------------
def detect_spikes(signal, threshold=4.0):
    """Return indices of peaks ≥ threshold."""
    peaks, _ = find_peaks(signal, height=threshold)
    return peaks

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

###############################################################################
#                             MODEL PARAMETERS                                #
###############################################################################
params = {
    # Simulation timing
    "t_max": 8000.0,
    "dt": 1.0,

    # Bipolar & Noise
    "n_bipolars": 10,
    "active_bipolars": 5,
    "refresh_rate": 15,
    "I_exc_per_bipolar": 0.08,

    # Membrane Passive
    "C_mem": 1.0,
    "g_leak": 0.05,
    "E_leak": -70.0,

    # HH-like gating
    "threshold_base": -57.5,
    "threshold_shift_LY": 0.0,  # We'll override later for shift=+5 if needed
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

    # Short-term plasticity for SAC1->SAC2
    "SAC1toSAC2_U": 0.02,
    "SAC1toSAC2_tau_rec": 50.0,

    # Short-term plasticity for SAC2->DSGC
    "tau_rec": 1000.0,
    "U": 0.08,
    "w_syn0": 1.0,

    # Dual-exponential kernel
    "tau_rise": 20.0,
    "tau_decay": 100.0,
    "t_kernel": 150.0,

    # Scaling factor for probabilistic release (SAC2->DSGC)
    "k_release": 0.005,

    # NEW: We define a post-release nonlinearity threshold & scale
    # for the "effectiveness" of the release onto the DSGC.
    "post_nlin_threshold": 30.0,  # If Ca2 < 10, reduce or zero release
    "post_nlin_scale_below": 0.5  # e.g. 0 => completely block if below threshold
}

###############################################################################
#                       HH-LIKE GATING FUNCTION                               #
###############################################################################
def m_inf(V, V_half, k_gate):
    return 1.0 / (1.0 + np.exp(-(V - V_half)/k_gate))

###############################################################################
#                      BIPOLAR INPUT & PRECOMPUTATION                         #
###############################################################################
def get_bipolar_count(cell, t, params):
    if 0 <= t < 2000:
        return int(np.clip(round(np.random.normal(5.0, 5.0)), 0, params["n_bipolars"]))
    elif 2000 <= t < 4000:
        return params["n_bipolars"] if cell == "SAC1" else \
               int(np.clip(round(np.random.normal(5.0, 5.0)), 0, params["n_bipolars"]))
    elif 4000 <= t < 6000:
        return params["n_bipolars"] if cell == "SAC2" else \
               int(np.clip(round(np.random.normal(5.0, 5.0)), 0, params["n_bipolars"]))
    else:
        return int(np.clip(round(np.random.normal(5.0, 5.0)), 0, params["n_bipolars"]))

def precompute_bipolar_inputs(params):
    dt = params["dt"]
    steps = int(params["t_max"] // dt)
    t_arr = np.arange(0, params["t_max"], dt)

    bipolar_SAC1 = np.zeros(steps)
    bipolar_SAC2 = np.zeros(steps)
    frame_period = 1000.0 / params["refresh_rate"]
    frame_step_count = int(frame_period // dt)

    for i in range(steps):
        t = t_arr[i]
        if i % frame_step_count == 0:
            bipolar_SAC1[i] = get_bipolar_count("SAC1", t, params)
            bipolar_SAC2[i] = get_bipolar_count("SAC2", t, params)
        else:
            bipolar_SAC1[i] = bipolar_SAC1[i-1]
            bipolar_SAC2[i] = bipolar_SAC2[i-1]

    return t_arr, bipolar_SAC1, bipolar_SAC2

###############################################################################
#                           MAIN SIMULATION                                   #
###############################################################################
def run_two_SAC_simulation(inhibit_connection=True, params=params, bipolar_data=None):
    dt = params["dt"]
    steps = int(params["t_max"] // dt)
    t_arr = np.arange(0, params["t_max"], dt)

    # If needed, generate bipolar data
    if bipolar_data is None:
        t_arr, bip1, bip2 = precompute_bipolar_inputs(params)
        bipolar_data = {"SAC1": bip1, "SAC2": bip2}
    else:
        bip1 = bipolar_data["SAC1"]
        bip2 = bipolar_data["SAC2"]

    V1 = np.zeros(steps)
    Ca1 = np.zeros(steps)
    m1 = np.zeros(steps)
    V2 = np.zeros(steps)
    Ca2 = np.zeros(steps)
    m2 = np.zeros(steps)

    # Initial conditions
    V1[0] = params["E_leak"]
    V2[0] = params["E_leak"]
    Ca1[0] = 0.0
    Ca2[0] = 0.0
    m1[0] = 0.0
    m2[0] = 0.0

    # Resource for SAC1->SAC2
    R_syn12 = 1.0
    R_syn12_trace = np.zeros(steps)
    R_syn12_trace[0] = R_syn12

    # Gating threshold
    V_half = params["threshold_base"] + params["threshold_shift_LY"]
    k_gate = params["k_gate"]
    tau_m = params["tau_m"]
    tau_Ca = params["tau_Ca"]
    I_Ca_in = params["I_Ca_in"]

    # Inhibition
    soft_thresh = params["soft_inh_threshold"]
    inh_slope = params["inh_slope"]
    I_inh_strength = params["I_inh_strength"]

    # STD for SAC1->SAC2
    U_inh = params["SAC1toSAC2_U"]
    tau_rec_inh = params["SAC1toSAC2_tau_rec"]

    for i in range(steps - 1):
        I_exc1 = bip1[i] * params["I_exc_per_bipolar"]
        I_exc2 = bip2[i] * params["I_exc_per_bipolar"]

        #--- SAC1 ---
        dV1 = (-params["g_leak"] * (V1[i] - params["E_leak"]) + I_exc1) * (dt / params["C_mem"])
        V1[i+1] = V1[i] + dV1

        m_inf_val1 = m_inf(V1[i], V_half, k_gate)
        dm1 = (m_inf_val1 - m1[i]) / tau_m
        m1[i+1] = m1[i] + dm1 * dt

        dCa1 = -Ca1[i]/tau_Ca + I_Ca_in * m1[i]
        Ca1[i+1] = Ca1[i] + dCa1 * dt

        #--- Inhibition from SAC1->SAC2
        if inhibit_connection:
            I_inh_base = I_inh_strength / (1 + np.exp(-(Ca1[i] - soft_thresh) / inh_slope))
            I_inh = I_inh_base * R_syn12
        else:
            I_inh_base = 0.0
            I_inh = 0.0

        #--- SAC2 ---
        I_total2 = I_exc2 - I_inh
        dV2 = (-params["g_leak"] * (V2[i] - params["E_leak"]) + I_total2) * (dt / params["C_mem"])
        V2[i+1] = V2[i] + dV2

        m_inf_val2 = m_inf(V2[i], V_half, k_gate)
        dm2 = (m_inf_val2 - m2[i]) / tau_m
        m2[i+1] = m2[i] + dm2 * dt

        dCa2 = -Ca2[i]/tau_Ca + I_Ca_in * m2[i]
        Ca2[i+1] = Ca2[i] + dCa2 * dt

        #--- STD on SAC1->SAC2
        dR_syn12 = (1.0 - R_syn12) / tau_rec_inh - U_inh * R_syn12 * I_inh_base
        R_syn12 = R_syn12 + dR_syn12 * dt
        R_syn12 = max(min(R_syn12, 1.0), 0.0)
        R_syn12_trace[i+1] = R_syn12

    #--- SAC2->DSGC Release: short-term plasticity first (pre-nonlinearity)
    steps = len(t_arr)
    release_events_pre = np.zeros(steps)  # raw events
    R_syn = 1.0
    R_syn_trace = np.zeros(steps)
    for i in range(steps):
        p_release = params["k_release"] * Ca2[i] * dt
        if np.random.rand() < p_release:
            # If "success," short-term plasticity is used up
            release_events_pre[i] = params["w_syn0"] * R_syn
            R_syn *= (1 - params["U"])
        R_syn += dt * (1 - R_syn) / params["tau_rec"]
        R_syn = min(R_syn, 1.0)
        R_syn_trace[i] = R_syn

    #--- NOW apply the post-nonlinearity: final release events
    #    e.g. if Ca2 < threshold => scale or zero the amplitude
    release_events_postNL = np.copy(release_events_pre)
    Ca_threshold  = params.get("post_nlin_threshold", 10.0)
    scale_below   = params.get("post_nlin_scale_below", 0.0)
    for i in range(steps):
        if Ca2[i] < Ca_threshold:
            # Scale the event amplitude (if any)
            release_events_postNL[i] *= scale_below

    #--- Convolve => DSGC IPSC using final release
    def dual_exponential_kernel(dt, t_max, tau_rise, tau_decay):
        t_kernel = np.arange(0, t_max, dt)
        kernel = np.exp(-t_kernel/tau_decay) - np.exp(-t_kernel/tau_rise)
        if np.max(kernel) != 0:
            kernel /= np.max(kernel)
        return kernel, t_kernel

    kernel, t_kernel = dual_exponential_kernel(dt, params["t_kernel"],
                                               params["tau_rise"], params["tau_decay"])
    IPSC = np.convolve(release_events_postNL, kernel, mode="full")[:steps]

    return t_arr, V1, Ca1, V2, Ca2, IPSC, R_syn_trace, R_syn12_trace


def detect_spikes(signal, threshold):
    peaks, _ = find_peaks(signal, height=threshold)
    return peaks

###############################################################################
#                            MAIN SCRIPT                                      #
###############################################################################
if __name__ == "__main__":
    #-----------------------------------------------------------------
    # Example: 3 columns in a single figure:
    # (1) Baseline-Intact, (2) Baseline-Cut, (3) Shifted-Intact
    # to produce 3×3 subplots as you requested
    #-----------------------------------------------------------------

    # 1) Baseline condition
    params_baseline = dict(params)
    params_baseline["threshold_shift_LY"] = 0.0  # => e.g. -57.5
    t_arr, bip1, bip2 = precompute_bipolar_inputs(params_baseline)
    bip_data = {"SAC1": bip1, "SAC2": bip2}

    # (A) Baseline-Intact
    resA_intact = run_two_SAC_simulation(
        inhibit_connection=True,
        params=params_baseline,
        bipolar_data=bip_data
    )
    (tA, V1A_int, Ca1A_int, V2A_int, Ca2A_int,
     IPSC_A_int, R_synA_int, R_syn12A_int) = resA_intact

    # (B) Baseline-Cut
    resA_cut = run_two_SAC_simulation(
        inhibit_connection=False,
        params=params_baseline,
        bipolar_data=bip_data
    )
    (_, V1A_cut, Ca1A_cut, V2A_cut, Ca2A_cut,
     IPSC_A_cut, R_synA_cut, R_syn12A_cut) = resA_cut

    # 2) Shifted condition
    params_shift = dict(params)
    params_shift["threshold_shift_LY"] = 4.0  # => e.g. -52.5
    # Reuse the same bip_data for identical noise
    resB_intact = run_two_SAC_simulation(
        inhibit_connection=True,
        params=params_shift,
        bipolar_data=bip_data
    )
    (tB, V1B_int, Ca1B_int, V2B_int, Ca2B_int,
     IPSC_B_int, R_synB_int, R_syn12B_int) = resB_intact

    #----------------- Plot 3 columns × 3 rows in a single figure --------------
    fig, axs = plt.subplots(3, 3, figsize=(16, 10), sharex=True, sharey='row')
    fig.suptitle("3×3 subplots: (1) Baseline-Intact, (2) Baseline-Cut, (3) Shifted-Intact\n"
                 "Post-Nonlinearity after short-term plasticity for SAC2→DSGC",
                 fontsize=12)

    #--- Column 0: Baseline-Intact
    axs[0,0].plot(tA, V1A_int, 'b-', label="SAC1 Voltage")
    axs[0,0].plot(tA, Ca1A_int, 'r-', label="SAC1 Ca")
    axs[0,0].set_title("Baseline (Intact): SAC1")
    axs[0,0].legend(fontsize=8)

    axs[1,0].plot(tA, V2A_int, 'b-', label="SAC2 Voltage")
    axs[1,0].plot(tA, Ca2A_int, 'r-', label="SAC2 Ca")
    axs[1,0].set_title("Baseline (Intact): SAC2")
    axs[1,0].set_ylabel("Voltage / Ca (a.u.)")
    axs[1,0].legend(fontsize=8)

    axs[2,0].plot(tA, IPSC_A_int, 'k-', label="DSGC IPSC")
    axs[2,0].set_title("Baseline (Intact): DSGC")
    axs[2,0].set_xlabel("Time (ms)")
    axs[2,0].set_ylabel("IPSC (a.u.)")
    axs[2,0].legend(fontsize=8)

    #--- Column 1: Baseline-Cut
    axs[0,1].plot(tA, V1A_cut, 'b-', label="SAC1 Voltage")
    axs[0,1].plot(tA, Ca1A_cut, 'r-', label="SAC1 Ca")
    axs[0,1].set_title("Baseline (Cut): SAC1")
    axs[0,1].legend(fontsize=8)

    axs[1,1].plot(tA, V2A_cut, 'b-', label="SAC2 Voltage")
    axs[1,1].plot(tA, Ca2A_cut, 'r-', label="SAC2 Ca")
    axs[1,1].set_title("Baseline (Cut): SAC2")
    axs[1,1].legend(fontsize=8)

    axs[2,1].plot(tA, IPSC_A_cut, 'k-', label="DSGC IPSC")
    axs[2,1].set_title("Baseline (Cut): DSGC")
    axs[2,1].set_xlabel("Time (ms)")
    axs[2,1].legend(fontsize=8)

    #--- Column 2: Shifted-Intact
    axs[0,2].plot(tB, V1B_int, 'b-', label="SAC1 Voltage")
    axs[0,2].plot(tB, Ca1B_int, 'r-', label="SAC1 Ca")
    axs[0,2].set_title("Shifted (Intact): SAC1")
    axs[0,2].legend(fontsize=8)

    axs[1,2].plot(tB, V2B_int, 'b-', label="SAC2 Voltage")
    axs[1,2].plot(tB, Ca2B_int, 'r-', label="SAC2 Ca")
    axs[1,2].set_title("Shifted (Intact): SAC2")
    axs[1,2].legend(fontsize=8)

    axs[2,2].plot(tB, IPSC_B_int, 'k-', label="DSGC IPSC")
    axs[2,2].set_title("Shifted (Intact): DSGC")
    axs[2,2].set_xlabel("Time (ms)")
    axs[2,2].legend(fontsize=8)

    plt.tight_layout()
    plt.show()

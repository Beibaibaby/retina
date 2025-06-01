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
    "threshold_base": -58.0,
    "threshold_shift_LY": -5.0,
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

    # Short-term plasticity for SAC1->SAC2 (NEW parameters):
    "SAC1toSAC2_U": 0.02,         # e.g. 10% utilization for SAC1->SAC2
    "SAC1toSAC2_tau_rec": 50.0,  # faster or slower than the SAC2->DSGC if desired

    # Short-term plasticity for SAC2->DSGC (ORIGINAL parameters):
    "tau_rec": 1000.0,  # recovery time constant for synaptic resource
    "U": 0.04,          # utilization factor
    "w_syn0": 1.0,

    # Dual-exponential kernel parameters (unchanged)
    "tau_rise": 10.0,
    "tau_decay": 100.0,
    "t_kernel": 100.0,

    # Scaling factor for probabilistic release (SAC2->DSGC)
    "k_release": 0.005
}

###############################################################################
#              ORIGINAL HH-LIKE GATING FUNCTION (UNCHANGED)                   #
###############################################################################
def m_inf(V, V_half, k_gate):
    return 1.0 / (1.0 + np.exp(-(V - V_half)/k_gate))

###############################################################################
#             BIPOLAR INPUT MODULATION (time & cell dependent)                #
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

###############################################################################
#      PRECOMPUTE BIPOLAR INPUT FOR SAC1 AND SAC2 (for both conditions)       #
###############################################################################
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
            bipolar_SAC1[i] = bipolar_SAC1[i - 1]
            bipolar_SAC2[i] = bipolar_SAC2[i - 1]

    return t_arr, bipolar_SAC1, bipolar_SAC2

###############################################################################
#            TWO-SAC SIMULATION WITH DISINHIBITORY CONNECTION                 #
#           (NOW WITH DIFFERENT STD FOR SAC1->SAC2 vs SAC2->DSGC)             #
###############################################################################
def run_two_SAC_simulation(inhibit_connection=True, params=params, bipolar_data=None):
    """
    Simulate two SAC cells (SAC1 and SAC2) with separate short-term depression:
      - SAC1->SAC2 uses `SAC1toSAC2_U` and `SAC1toSAC2_tau_rec`.
      - SAC2->DSGC uses the original `U` and `tau_rec`.

    The rest of the simulation logic is unchanged, except for the resource
    depletion and recovery for the SAC1->SAC2 connection inside the main loop.
    """
    dt = params["dt"]
    steps = int(params["t_max"] // dt)
    t_arr = np.arange(0, params["t_max"], dt)

    # Precompute bipolar data if not provided
    if bipolar_data is None:
        t_arr, bip1, bip2 = precompute_bipolar_inputs(params)
        bipolar_data = {"SAC1": bip1, "SAC2": bip2}
    else:
        bip1 = bipolar_data["SAC1"]
        bip2 = bipolar_data["SAC2"]

    # Allocate arrays for SAC1 and SAC2 variables
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

    # Short-term depression resource for SAC1->SAC2
    R_syn12 = 1.0
    R_syn12_trace = np.zeros(steps)
    R_syn12_trace[0] = R_syn12

    # Local references to parameters
    V_half = params["threshold_base"]
    k_gate = params["k_gate"]
    tau_m = params["tau_m"]
    tau_Ca = params["tau_Ca"]
    I_Ca_in = params["I_Ca_in"]

    # For the SAC1->SAC2 inhibitory nonlinearity
    soft_thresh = params["soft_inh_threshold"]
    inh_slope = params["inh_slope"]
    I_inh_strength = params["I_inh_strength"]

    # Short-term plasticity for SAC1->SAC2
    U_inh = params["SAC1toSAC2_U"]
    tau_rec_inh = params["SAC1toSAC2_tau_rec"]

    # Main simulation loop
    for i in range(steps - 1):
        # Excitatory input from bipolars
        I_exc1 = bip1[i] * params["I_exc_per_bipolar"]
        I_exc2 = bip2[i] * params["I_exc_per_bipolar"]

        # Update SAC1
        dV1 = (-params["g_leak"] * (V1[i] - params["E_leak"]) + I_exc1) * (dt / params["C_mem"])
        V1[i+1] = V1[i] + dV1

        m_inf_val1 = m_inf(V1[i], V_half, k_gate)
        dm1 = (m_inf_val1 - m1[i]) / tau_m
        m1[i+1] = m1[i] + dm1 * dt

        dCa1 = -Ca1[i]/tau_Ca + I_Ca_in * m1[i]
        Ca1[i+1] = Ca1[i] + dCa1*dt

        # Compute the base inhibitory drive from SAC1->SAC2
        if inhibit_connection:
            I_inh_base = I_inh_strength / (1 + np.exp(-(Ca1[i] - soft_thresh) / inh_slope))
            # Apply short-term depression scale
            I_inh = I_inh_base * R_syn12
        else:
            I_inh_base = 0.0
            I_inh = 0.0

        # Total input to SAC2
        I_total2 = I_exc2 - I_inh

        # Update SAC2
        dV2 = (-params["g_leak"] * (V2[i] - params["E_leak"]) + I_total2) * (dt / params["C_mem"])
        V2[i+1] = V2[i] + dV2

        m_inf_val2 = m_inf(V2[i], V_half, k_gate)
        dm2 = (m_inf_val2 - m2[i]) / tau_m
        m2[i+1] = m2[i] + dm2 * dt

        dCa2 = -Ca2[i]/tau_Ca + I_Ca_in * m2[i]
        Ca2[i+1] = Ca2[i] + dCa2 * dt

        # Update the short-term depression resource R_syn12
        # Recovery: (1 - R_syn12) / tau_rec_inh
        # Depletion: U_inh * R_syn12 * I_inh_base
        dR_syn12 = (1.0 - R_syn12) / tau_rec_inh - U_inh * R_syn12 * I_inh_base
        R_syn12 = R_syn12 + dR_syn12 * dt
        # Optionally clamp to [0,1] if desired
        R_syn12 = max(min(R_syn12, 1.0), 0.0)
        R_syn12_trace[i+1] = R_syn12

    # ------------------------------------------
    # Probabilistic Synaptic Release (SAC2->DSGC)
    # ------------------------------------------
    release_events = np.zeros(steps)
    R_syn = 1.0  # resource for SAC2->DSGC
    R_syn_trace = np.zeros(steps)
    k_release = params["k_release"]

    for i in range(steps):
        p_release = k_release * Ca2[i] * dt
        if np.random.rand() < p_release:
            release_events[i] = params["w_syn0"] * R_syn
            R_syn *= (1 - params["U"])
        # Recover
        R_syn += dt * (1 - R_syn) / params["tau_rec"]
        R_syn = min(R_syn, 1.0)
        R_syn_trace[i] = R_syn

    # Convolve release events => DSGC IPSC
    def dual_exponential_kernel(dt, t_max, tau_rise, tau_decay):
        t_kernel = np.arange(0, t_max, dt)
        kernel = np.exp(-t_kernel/tau_decay) - np.exp(-t_kernel/tau_rise)
        if np.max(kernel) != 0:
            kernel /= np.max(kernel)
        return kernel, t_kernel

    kernel, t_kernel = dual_exponential_kernel(
        params["dt"], params["t_kernel"],
        params["tau_rise"], params["tau_decay"]
    )
    IPSC = np.convolve(release_events, kernel, mode="full")[:steps]

    return t_arr, V1, Ca1, V2, Ca2, IPSC, R_syn_trace, R_syn12_trace

###############################################################################
#                           SPIKE DETECTION (unchanged)                       #
###############################################################################
def detect_spikes(signal, threshold):
    peaks, _ = find_peaks(signal, height=threshold)
    return peaks

###############################################################################
#                               MAIN SCRIPT                                   #
###############################################################################
if __name__ == "__main__":
    # Precompute bipolar inputs once
    t_arr, bipolar_SAC1, bipolar_SAC2 = precompute_bipolar_inputs(params)
    bipolar_data = {"SAC1": bipolar_SAC1, "SAC2": bipolar_SAC2}

    # 1) Intact circuit (with separate STD for SAC1->SAC2 and SAC2->DSGC)
    results_intact = run_two_SAC_simulation(
        inhibit_connection=True,
        params=params,
        bipolar_data=bipolar_data
    )
    (t_arr, V1_intact, Ca1_intact, V2_intact, Ca2_intact,
     IPSC_intact, R_syn_trace_intact, R_syn12_trace_intact) = results_intact

    # 2) Connection cut (no SAC1->SAC2 inhibition)
    results_cut = run_two_SAC_simulation(
        inhibit_connection=False,
        params=params,
        bipolar_data=bipolar_data
    )
    (_, V1_cut, Ca1_cut, V2_cut, Ca2_cut,
     IPSC_cut, R_syn_trace_cut, R_syn12_trace_cut) = results_cut

    # Detect Ca spikes
    spikes_SAC1_intact = detect_spikes(Ca1_intact, params["spike_threshold"])
    spikes_SAC2_intact = detect_spikes(Ca2_intact, params["spike_threshold"])
    spikes_SAC2_cut    = detect_spikes(Ca2_cut,    params["spike_threshold"])

    # ------------------- Plot #1: SAC1, SAC2, DSGC (Intact vs Cut) ---------------
    fig, axs = plt.subplots(3, 2, figsize=(14, 10), sharex=True, sharey='row')

    # Left column: Intact
    axs[0,0].plot(t_arr, V1_intact, 'b-', label="V_SAC1")
    axs[0,0].plot(t_arr, Ca1_intact, 'r-', label="Ca_SAC1")
    axs[0,0].plot(t_arr[spikes_SAC1_intact], Ca1_intact[spikes_SAC1_intact],
                  'ko', markersize=3, label="Ca events")
    axs[0,0].set_title("SAC1 (Intact)")
    axs[0,0].set_ylabel("Voltage (mV) / Ca (a.u.)")
    axs[0,0].legend(fontsize=8)

    axs[1,0].plot(t_arr, V2_intact, 'b-', label="V_SAC2")
    axs[1,0].plot(t_arr, Ca2_intact, 'r-', label="Ca_SAC2")
    axs[1,0].plot(t_arr[spikes_SAC2_intact], Ca2_intact[spikes_SAC2_intact],
                  'ko', markersize=3, label="Ca events")
    axs[1,0].set_title("SAC2 (Intact w/ STD from SAC1)")
    axs[1,0].set_ylabel("Voltage (mV) / Ca (a.u.)")
    axs[1,0].legend(fontsize=8)

    axs[2,0].plot(t_arr, IPSC_intact, 'k-', label="DSGC IPSC")
    axs[2,0].set_title("DSGC IPSC (Intact)")
    axs[2,0].set_xlabel("Time (ms)")
    axs[2,0].set_ylabel("IPSC (a.u.)")
    axs[2,0].legend(fontsize=8)

    # Right column: Connection Cut
    axs[0,1].axis('off')  # no separate plot for SAC1 in the top-right panel
    axs[1,1].plot(t_arr, V2_cut, 'b-', label="V_SAC2")
    axs[1,1].plot(t_arr, Ca2_cut, 'r-', label="Ca_SAC2")
    axs[1,1].plot(t_arr[spikes_SAC2_cut], Ca2_cut[spikes_SAC2_cut],
                  'ko', markersize=3, label="Ca events")
    axs[1,1].set_title("SAC2 (Cut)")
    axs[1,1].set_ylabel("Voltage (mV) / Ca (a.u.)")
    axs[1,1].legend(fontsize=8)

    axs[2,1].plot(t_arr, IPSC_cut, 'k-', label="DSGC IPSC")
    axs[2,1].set_title("DSGC IPSC (Cut)")
    axs[2,1].set_xlabel("Time (ms)")
    axs[2,1].set_ylabel("IPSC (a.u.)")
    axs[2,1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()

    # ------------------- Plot #2: R_syn for SAC2->DSGC  ----------------------------
    fig2, axs2 = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    axs2[0].plot(t_arr, R_syn_trace_intact, 'g-', label="SAC2->DSGC Resource (Intact)")
    axs2[0].set_title("SAC2->DSGC Resource (Intact)")
    axs2[0].set_xlabel("Time (ms)")
    axs2[0].set_ylabel("R_syn")
    axs2[0].legend(fontsize=8)

    axs2[1].plot(t_arr, R_syn_trace_cut, 'g-', label="SAC2->DSGC Resource (Cut)")
    axs2[1].set_title("SAC2->DSGC Resource (Cut)")
    axs2[1].set_xlabel("Time (ms)")
    axs2[1].legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    # ------------------- Plot #3: R_syn12 for SAC1->SAC2 (new STD)  ----------------
    fig3 = plt.figure(figsize=(7,5))
    plt.plot(t_arr, R_syn12_trace_intact, 'm-', label="SAC1->SAC2 Resource (Intact)")
    plt.title("Short-Term Depression of SAC1->SAC2 Connection")
    plt.xlabel("Time (ms)")
    plt.ylabel("R_syn12")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

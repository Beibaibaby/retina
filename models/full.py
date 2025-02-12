import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

###############################################################################
#                             MODEL PARAMETERS                                #
###############################################################################
params = {
    # Simulation timing
    "t_max": 8000.0,          # total simulation time (ms) for ONE run
    "dt": 1.0,                # time step (ms)

    # Bipolar & Noise (kept exactly as in your code)
    "n_bipolars": 10,         # total bipolars
    "active_bipolars": 5,     # how many bipolars are ON each frame if noise is ON
    "refresh_rate": 15,       # Hz => one new frame each ~66.7 ms
    "I_exc_per_bipolar": 0.08,# excitatory current contribution per ON bipolar

    # Membrane Passive
    "C_mem": 1.0,             # membrane capacitance
    "g_leak": 0.05,           # leak conductance
    "E_leak": -70.0,          # leak reversal (mV)

    # HH-like gating for Ca
    "threshold_base": -58.0,  # baseline half-activation voltage (V_half)
    "threshold_shift_LY": -5.0, # shift if +LY => V_half => base + shift
    "k_gate": 0.8,            # slope factor for m_inf(V) = 1/(1+exp(-(V - V_half)/k))
    "tau_m": 10.0,            # gating variable time constant (ms)
    "I_Ca_in": 0.2,           # amplitude of Ca influx if m=1

    # Calcium dynamics
    "tau_Ca": 200.0,          # decay time constant for Ca (ms)

    # Plotting range
    "plot_range": (-80, 50, -1.0, 50.0),

    # Additional parameters for the disinhibitory circuit & synaptic release:
    "I_inh_strength": 0.18,    # maximum inhibitory current applied to SAC2 from SAC1
    "spike_threshold": 4,     # (old threshold, not used in the new probabilistic method)

    # New parameters for the soft threshold on SAC1->SAC2 inhibition:
    "soft_inh_threshold": 30.0,  # soft threshold for SAC1 Ca to start taking effect
    "inh_slope": 0.3,            # slope (steepness) of the inhibitory sigmoid

    # Short-term plasticity (for SAC2→DSGC synapse)
    "tau_rec": 1000.0,         # recovery time constant for synaptic resource (ms)
    "U": 0.04,                # utilization factor (fraction of resources used per release)
    "w_syn0": 1.0,            # baseline synaptic weight scaling

    # Dual-exponential kernel parameters for converting release events into IPSC:
    "tau_rise": 10.0,         # rise time constant (ms)
    "tau_decay": 100.0,       # decay time constant (ms)
    "t_kernel": 100.0,        # kernel duration (ms)
    
    # Scaling factor for probabilistic release (units: 1/(Ca*ms))
    "k_release": 0.005
}

###############################################################################
#              ORIGINAL HH-LIKE GATING FUNCTIONS (unchanged)                #
###############################################################################
def m_inf(V, V_half, k_gate):
    """Sigmoidal steady-state gating: m_inf(V)."""
    return 1.0 / (1.0 + np.exp(-(V - V_half)/k_gate))

###############################################################################
#             BIPOLAR INPUT MODULATION (time & cell dependent)                #
###############################################################################
def get_bipolar_count(cell, t, params):
    """
    Returns the number of active bipolar cells for a given SAC cell at time t.
    
    Four time periods (ms):
      0-2000: both cells get noisy input (sampled from N(5,5^2))
      2000-4000: SAC1 gets high input (all bipolars active), SAC2 remains noisy.
      4000-6000: SAC2 gets high input (all bipolars active), SAC1 remains noisy.
      6000-8000: both return to noisy input.
    """
    if 0 <= t < 2000:
        return int(np.clip(round(np.random.normal(5.0, 5.0)), 0, params["n_bipolars"]))
    elif 2000 <= t < 4000:
        return params["n_bipolars"] if cell == "SAC1" else int(np.clip(round(np.random.normal(5.0, 5.0)), 0, params["n_bipolars"]))
    elif 4000 <= t < 6000:
        return params["n_bipolars"] if cell == "SAC2" else int(np.clip(round(np.random.normal(5.0, 5.0)), 0, params["n_bipolars"]))
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
            bipolar_SAC1[i] = bipolar_SAC1[i-1]
            bipolar_SAC2[i] = bipolar_SAC2[i-1]
    return t_arr, bipolar_SAC1, bipolar_SAC2

###############################################################################
#            TWO-SAC SIMULATION WITH DISINHIBITORY CONNECTION                 #
###############################################################################
def run_two_SAC_simulation(inhibit_connection=True, params=params, bipolar_data=None):
    """
    Simulate two SAC cells (SAC1 and SAC2) concurrently over t_max (8000 ms) using
    the original HH-like dynamics. Bipolar input for each SAC is provided via bipolar_data,
    a dict with keys "SAC1" and "SAC2" (arrays for each time step).
    This ensures identical bipolar input across conditions.
    
    If inhibit_connection is True, SAC1's Ca events subtract an inhibitory current from SAC2's input.
    The inhibitory current is computed using a soft-threshold (sigmoid):
    
         I_inh = I_inh_strength / (1+exp(-(Ca1 - soft_inh_threshold)/inh_slope))
    
    After simulating voltage and Ca for both SACs, synaptic release events are generated from
    SAC2's Ca using a probabilistic method. Each release event is scaled by a synaptic resource
    variable (with short-term depression). The evolution of this resource (R_syn) is recorded,
    and then the discrete release events are convolved with a dual-exponential kernel to yield a smooth
    DSGC IPSC.
    """
    dt = params["dt"]
    steps = int(params["t_max"] // dt)
    t_arr = np.arange(0, params["t_max"], dt)
    
    # Use precomputed bipolar data if provided; otherwise compute it.
    if bipolar_data is None:
        t_arr, bipolar_SAC1, bipolar_SAC2 = precompute_bipolar_inputs(params)
        bipolar_data = {"SAC1": bipolar_SAC1, "SAC2": bipolar_SAC2}
    
    # Allocate arrays for SAC1 and SAC2 variables:
    V1 = np.zeros(steps)
    Ca1 = np.zeros(steps)
    m1 = np.zeros(steps)
    V2 = np.zeros(steps)
    Ca2 = np.zeros(steps)
    m2 = np.zeros(steps)
    
    # Initial conditions:
    V1[0] = params["E_leak"]
    Ca1[0] = 0.0
    m1[0] = 0.0
    V2[0] = params["E_leak"]
    Ca2[0] = 0.0
    m2[0] = 0.0
    
    # SAC dynamics parameters:
    V_half = params["threshold_base"]
    k_gate = params["k_gate"]
    tau_m = params["tau_m"]
    tau_Ca = params["tau_Ca"]
    I_Ca_in = params["I_Ca_in"]
    
    # Simulation loop:
    for i in range(steps - 1):
        I_exc1 = bipolar_data["SAC1"][i] * params["I_exc_per_bipolar"]
        I_exc2 = bipolar_data["SAC2"][i] * params["I_exc_per_bipolar"]
        
        # Soft inhibitory effect from SAC1 on SAC2 using a sigmoid:
        I_inh = 0.0
        if inhibit_connection:
            inh_slope = params.get("inh_slope", 1.0)
            soft_thresh = params.get("soft_inh_threshold", 30.0)
            I_inh = params["I_inh_strength"] / (1 + np.exp(-(Ca1[i] - soft_thresh) / inh_slope))
        I_total2 = I_exc2 - I_inh
        
        # Update SAC1:
        dV1 = (-params["g_leak"] * (V1[i] - params["E_leak"]) + I_exc1) * (dt / params["C_mem"])
        V1[i+1] = V1[i] + dV1
        m_inf_val1 = m_inf(V1[i], V_half, k_gate)
        dm1 = (m_inf_val1 - m1[i]) / tau_m
        m1[i+1] = m1[i] + dm1 * dt
        dCa1 = -Ca1[i] / tau_Ca + I_Ca_in * m1[i]
        Ca1[i+1] = Ca1[i] + dCa1 * dt
        
        # Update SAC2:
        dV2 = (-params["g_leak"] * (V2[i] - params["E_leak"]) + I_total2) * (dt / params["C_mem"])
        V2[i+1] = V2[i] + dV2
        m_inf_val2 = m_inf(V2[i], V_half, k_gate)
        dm2 = (m_inf_val2 - m2[i]) / tau_m
        m2[i+1] = m2[i] + dm2 * dt
        dCa2 = -Ca2[i] / tau_Ca + I_Ca_in * m2[i]
        Ca2[i+1] = Ca2[i] + dCa2 * dt

    # --- Probabilistic Synaptic Release from SAC2 based on its Ca trace ---
    release_events = np.zeros(steps)
    R_syn = 1.0  # initial synaptic resource for SAC2→DSGC
    R_syn_trace = np.zeros(steps)  # record synaptic resource over time
    k_release = params["k_release"]
    for i in range(steps):
        p_release = k_release * Ca2[i] * dt
        if np.random.rand() < p_release:
            release_events[i] = params["w_syn0"] * R_syn
            R_syn *= (1 - params["U"])
        R_syn += dt * (1 - R_syn) / params["tau_rec"]
        R_syn_trace[i] = R_syn

    # Dual-exponential kernel to convert release events to a smooth IPSC:
    def dual_exponential_kernel(dt, t_max, tau_rise, tau_decay):
        t_kernel = np.arange(0, t_max, dt)
        kernel = np.exp(-t_kernel/tau_decay) - np.exp(-t_kernel/tau_rise)
        if np.max(kernel) != 0:
            kernel = kernel / np.max(kernel)
        return kernel, t_kernel
    
    kernel, t_kernel = dual_exponential_kernel(params["dt"], params["t_kernel"],
                                               params["tau_rise"], params["tau_decay"])
    IPSC = np.convolve(release_events, kernel, mode="full")[:steps]
    
    return t_arr, V1, Ca1, V2, Ca2, IPSC, R_syn_trace

###############################################################################
#                           SPIKE DETECTION (for plotting)                    #
###############################################################################
def detect_spikes(signal, threshold):
    """Return indices where the signal crosses above the threshold."""
    peaks, _ = find_peaks(signal, height=threshold)
    return peaks

###############################################################################
#                               PLOTTING                                    #
###############################################################################
if __name__ == "__main__":
    # Precompute bipolar inputs once for both conditions.
    t_arr, bipolar_SAC1, bipolar_SAC2 = precompute_bipolar_inputs(params)
    bipolar_data = {"SAC1": bipolar_SAC1, "SAC2": bipolar_SAC2}
    
    # Run simulation for two conditions (using the same bipolar_data):
    # 1) Intact disinhibitory circuit: SAC1→SAC2 connection active.
    t_arr, V1_intact, Ca1_intact, V2_intact, Ca2_intact, IPSC_intact, R_syn_trace_intact = run_two_SAC_simulation(inhibit_connection=True, params=params, bipolar_data=bipolar_data)
    
    # 2) Circuit with the SAC1→SAC2 connection cut.
    t_arr, V1_cut, Ca1_cut, V2_cut, Ca2_cut, IPSC_cut, R_syn_trace_cut = run_two_SAC_simulation(inhibit_connection=False, params=params, bipolar_data=bipolar_data)
    
    # Detect Ca "spikes" for plotting (using the spike_threshold)
    spikes_SAC1_intact = detect_spikes(Ca1_intact, params["spike_threshold"])
    spikes_SAC2_intact = detect_spikes(Ca2_intact, params["spike_threshold"])
    spikes_SAC2_cut    = detect_spikes(Ca2_cut, params["spike_threshold"])
    
    # ---------------- Combined Plot Layout ----------------
    # Create a 3-row x 2-column grid.
    # Left column: intact circuit (SAC1, SAC2, DSGC IPSC).
    # Right column: top cell empty; row2: SAC2 (connection cut); row3: DSGC IPSC (connection cut).
    fig, axs = plt.subplots(3, 2, figsize=(14, 10), sharex=True, sharey='row')
    
    # Left column: Intact Circuit
    axs[0,0].plot(t_arr, V1_intact, 'b-', label="V_SAC1")
    axs[0,0].plot(t_arr, Ca1_intact, 'r-', label="Ca_SAC1")
    axs[0,0].plot(t_arr[spikes_SAC1_intact], Ca1_intact[spikes_SAC1_intact], 'ko', markersize=3, label="Ca events")
    axs[0,0].set_title("SAC1 (Intact)")
    axs[0,0].set_ylabel("Voltage (mV) / Ca (a.u.)")
    axs[0,0].legend(fontsize=8)
    
    axs[1,0].plot(t_arr, V2_intact, 'b-', label="V_SAC2")
    axs[1,0].plot(t_arr, Ca2_intact, 'r-', label="Ca_SAC2")
    axs[1,0].plot(t_arr[spikes_SAC2_intact], Ca2_intact[spikes_SAC2_intact], 'ko', markersize=3, label="Ca events")
    axs[1,0].set_title("SAC2 (Intact; inhibited by SAC1)")
    axs[1,0].set_ylabel("Voltage (mV) / Ca (a.u.)")
    axs[1,0].legend(fontsize=8)
    
    axs[2,0].plot(t_arr, IPSC_intact, 'k-', label="DSGC IPSC")
    axs[2,0].set_title("DSGC IPSC (Intact)")
    axs[2,0].set_xlabel("Time (ms)")
    axs[2,0].set_ylabel("IPSC (a.u.)")
    axs[2,0].legend(fontsize=8)
    
    # Right column: Connection Cut
    axs[0,1].axis('off')
    
    axs[1,1].plot(t_arr, V2_cut, 'b-', label="V_SAC2")
    axs[1,1].plot(t_arr, Ca2_cut, 'r-', label="Ca_SAC2")
    axs[1,1].plot(t_arr[spikes_SAC2_cut], Ca2_cut[spikes_SAC2_cut], 'ko', markersize=3, label="Ca events")
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
    
    # ---------------- Additional Plot: Synaptic Resource (R_syn) ----------------
    fig2, axs2 = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    axs2[0].plot(t_arr, R_syn_trace_intact, 'g-', label="R_syn (Intact)")
    axs2[0].set_title("Synaptic Resource (Intact)")
    axs2[0].set_xlabel("Time (ms)")
    axs2[0].set_ylabel("R_syn")
    axs2[0].legend(fontsize=8)
    
    axs2[1].plot(t_arr, R_syn_trace_cut, 'g-', label="R_syn (Cut)")
    axs2[1].set_title("Synaptic Resource (Cut)")
    axs2[1].set_xlabel("Time (ms)")
    axs2[1].legend(fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # ---------------- Third Plot: Visualizing the Inhibitory Sigmoid Function -----------
    # This plot shows the soft-threshold sigmoid mapping from SAC1 Ca to inhibition,
    # and overlays sample SAC1 Ca values (from the intact simulation) on the curve.
    
    # Use the soft threshold parameters:
    soft_thresh = params.get("soft_inh_threshold", 30.0)
    inh_slope = params.get("inh_slope", 0.2)
    
    # Generate a range of Ca values:
    x_vals = np.linspace(0, 50, 200)
    # Compute the sigmoid function:
    sigmoid_vals = params["I_inh_strength"] / (1 + np.exp(-(x_vals - soft_thresh) / inh_slope))
    
    # Sample a subset of SAC1 Ca values from the intact simulation:
    sample_indices = np.arange(0, len(Ca1_intact), 10)
    Ca_sample = Ca1_intact[sample_indices]
    I_inh_sample = params["I_inh_strength"] / (1 + np.exp(-(Ca_sample - soft_thresh) / inh_slope))
    
    # Plot the sigmoid function and overlay the sample data:
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    ax1.plot(x_vals, sigmoid_vals, 'b-', label="Sigmoid Inhibition")
    ax1.set_title("Sigmoid Function for Inhibition")
    ax1.set_xlabel("SAC1 Ca Level")
    ax1.set_ylabel("Inhibitory Current")
    ax1.legend(fontsize=8)
    
    ax2.scatter(Ca_sample, I_inh_sample, color='r', label="Sampled SAC1 Ca")
    ax2.set_title("SAC1 Ca vs. Inhibition")
    ax2.set_xlabel("SAC1 Ca Level")
    ax2.legend(fontsize=8)
    
    plt.tight_layout()
    plt.show()

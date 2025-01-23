import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import gaussian_kde

###############################################################################
#                             MODEL PARAMETERS                                #
###############################################################################

params = {
    # Simulation timing
    "t_max": 8000.0,          # total simulation time (ms) for ONE run
    "dt": 1.0,                # time step (ms)

    # Bipolar & Noise
    "n_bipolars": 10,         # total bipolars
    "active_bipolars": 5,     # how many bipolars are ON each frame if noise is ON
    "refresh_rate": 15,       # Hz => one new frame each ~66.7 ms
    "I_exc_per_bipolar": 0.08,# excitatory current contribution per ON bipolar

    # Membrane Passive
    "C_mem": 1.0,             # membrane capacitance
    "g_leak": 0.05,           # leak conductance
    "E_leak": -70.0,          # leak reversal (mV)

    # HH-like gating for Ca
    # We'll interpret threshold_base as the baseline V_half
    "threshold_base": -56.0,  # baseline half-activation voltage (V_half)
    "threshold_shift_LY": -5.0, # shift if +LY => V_half => base + shift
    "k_gate": 2.0,            # slope factor for m_inf(V) = 1/(1+exp(-(V - V_half)/k))
    "tau_m": 10.0,            # gating variable time constant (ms)
    "I_Ca_in": 0.2,           # amplitude of Ca influx if m=1

    # Calcium dynamics
    "tau_Ca": 200.0,          # decay time constant for Ca (ms)

    # Plotting range
    "plot_range": (-80, 50, -1.0, 50.0),
}

###############################################################################
#                           HH-LIKE GATING FUNCTIONS                          #
###############################################################################

def pick_bipolars_active(condition, params):
    """
    If 'Noise' in condition => sample from N(5,5^2), else 0 bipolars active.
    """
    if condition in ["Noise_NoLY", "Noise_+LY"]:
        val = np.random.normal(5.0, 5.0)
        val = int(np.clip(round(val), 0, params["n_bipolars"]))
        return val
    else:
        return 0

def m_inf(V, V_half, k_gate):
    """
    Sigmoidal steady-state gating: m_inf(V).
    """
    return 1.0 / (1.0 + np.exp(-(V - V_half)/k_gate))

def run_simulation_HHlike(condition, params):
    """
    Single-compartment:
      dV/dt = (-g_leak*(V - E_leak) + I_exc)/C_mem
      dm/dt = [m_inf(V) - m]/tau_m
      dCa/dt= -Ca/tau_Ca + I_Ca_in*m
    The 'threshold_base' is used as V_half; if condition has +LY, we shift it.
    """
    dt = params["dt"]
    steps = int(params["t_max"] // dt)
    time_arr = np.arange(0, params["t_max"], dt)

    # Calculate the modded V_half if +LY
    V_half_base = params["threshold_base"]
    shift = params["threshold_shift_LY"] if "+LY" in condition else 0.0
    V_half_mod = V_half_base + shift

    # Initialize states
    V = params["E_leak"]
    m = 0.0
    Ca = 0.0

    # Gating constants
    k_gate = params["k_gate"]
    tau_m_val = params["tau_m"]
    tau_ca = params["tau_Ca"]
    I_ca_in = params["I_Ca_in"]

    frame_period = 1000.0 / params["refresh_rate"]
    frame_step_count = int(frame_period // dt)
    bipolars_on = 0
    step_in_frame = 0

    V_history = np.zeros(steps)
    Ca_history = np.zeros(steps)

    for i in range(steps):
        # Possibly pick bipolars at refresh
        if step_in_frame == 0:
            bipolars_on = pick_bipolars_active(condition, params)
        step_in_frame = (step_in_frame + 1) % frame_step_count

        # excitatory input
        I_exc = bipolars_on * params["I_exc_per_bipolar"]

        # 1) Membrane eq
        dV = (-params["g_leak"]*(V - params["E_leak"]) + I_exc) * (dt/params["C_mem"])
        V += dV

        # 2) Gating update
        m_inf_val = m_inf(V, V_half_mod, k_gate)
        dm = (m_inf_val - m)/tau_m_val
        m += dm*dt

        # 3) Ca update
        dCa = -Ca/tau_ca + I_ca_in*m
        Ca += dCa*dt

        V_history[i] = V
        Ca_history[i] = Ca

    return time_arr, V_history, Ca_history

###############################################################################
#                      MULTI-RUN EVENT DETECTION & PLOTTING                   #
###############################################################################

def compute_event_rate(t, Ca, height=None, prominence=None):
    """
    find_peaks on Ca array => # of peaks / total_time_s
    """
    peaks, props = find_peaks(Ca, height=height, prominence=prominence)
    total_time_s = (t[-1] - t[0])/1000.0
    rate = len(peaks)/total_time_s if total_time_s>0 else 0
    return rate, peaks

def run_condition_multiple_times(condition, params, n_repeats=5):
    """
    We'll run the HHlike sim multiple times for each condition,
    so we can average over random noise states if condition has noise.
    """
    results = []
    for _ in range(n_repeats):
        t, V, Ca = run_simulation_HHlike(condition, params)
        results.append((t, V, Ca))
    return results

def run_all_conditions_multi(params, n_repeats=5, height=None, prominence=None):
    """
    1) For each condition: run n_repeats times
    2) We display one example run's V & Ca (with peaks on Ca)
    3) Compute average event rates across runs => bar chart
    """
    conditions = ["NoNoise_NoLY", "NoNoise_+LY", "Noise_NoLY", "Noise_+LY"]
    mean_rates = []
    std_rates  = []

    fig, axs = plt.subplots(2, 2, figsize=(12,6), sharex=True)
    fig.suptitle("HH-like gating: One Example Trace per Condition", fontsize=14)

    for idx, cond in enumerate(conditions):
        row = idx//2
        col = idx%2
        ax = axs[row][col]

        # run multiple
        multi_runs = run_condition_multiple_times(cond, params, n_repeats)
        # pick first as example
        t_ex, V_ex, Ca_ex = multi_runs[0]

        # detect Ca peaks in example
        ex_rate, ex_peaks = compute_event_rate(t_ex, Ca_ex, height=height, prominence=prominence)

        # Plot voltage + Ca on a twinx
        ax.plot(t_ex, V_ex, 'b-', label="Voltage")
        ax.set_xlim(0, params["t_max"])
        (Vm_min, Vm_max, Ca_min, Ca_max) = params["plot_range"]
        ax.set_ylim(Vm_min, Vm_max)
        ax.set_title(f"{cond}: example => {ex_rate:.2f} ev/s")

        ax2 = ax.twinx()
        ax2.plot(t_ex, Ca_ex, 'r-', label="Ca")
        ax2.plot(t_ex[ex_peaks], Ca_ex[ex_peaks], 'ko', label="Peaks")
        ax2.set_ylim(Ca_min, Ca_max)

        if row==1:
            ax.set_xlabel("Time (ms)")
        if col==0:
            ax.set_ylabel("Voltage (mV)")
        ax2.set_ylabel("Calcium (a.u.)")

        # gather event rates from all runs
        run_rates = []
        for (t_i, V_i, Ca_i) in multi_runs:
            r_i, p_i = compute_event_rate(t_i, Ca_i, height=height, prominence=prominence)
            run_rates.append(r_i)

        m_r = np.mean(run_rates)
        s_r = np.std(run_rates)
        mean_rates.append(m_r)
        std_rates.append(s_r)

        # combined legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax2.legend(h1+h2, l1+l2, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()

    # Bar chart
    plt.figure(figsize=(6,4))
    xvals = np.arange(len(conditions))
    plt.bar(xvals, mean_rates, yerr=std_rates, capsize=4,
            color='skyblue', edgecolor='black')
    plt.xticks(xvals, conditions, rotation=45, ha='right')
    plt.ylabel("Mean Calcium Event Rate (events/s)")
    plt.title(f"HH-like gating: n_repeats={n_repeats}")
    plt.tight_layout()
    plt.show()

    for cond, mr, sr in zip(conditions, mean_rates, std_rates):
        print(f"{cond}: {mr:.2f} ± {sr:.2f} events/sec (n={n_repeats})")

###############################################################################
#              LEFT-PROMINENCE ANALYSIS (OPTIONAL, SAME STRUCTURE)           #
###############################################################################

def compute_left_prominences(trace, peaks):
    prominences, left_bases, _ = peak_prominences(trace, peaks)
    left_proms = trace[peaks] - trace[left_bases]
    return left_proms

def collect_all_left_prominences_multi(params, n_repeats=5):
    conditions = ["NoNoise_NoLY", "NoNoise_+LY", "Noise_NoLY", "Noise_+LY"]
    all_left_proms = []
    cond_left_prom_dict = {}
    for cond in conditions:
        runs = run_condition_multiple_times(cond, params, n_repeats)
        cond_proms = []
        for (t_i, V_i, Ca_i) in runs:
            peaks_i, _ = find_peaks(Ca_i, height=0.0, prominence=0.0)
            L_i = compute_left_prominences(Ca_i, peaks_i)
            cond_proms.extend(L_i)
            all_left_proms.extend(L_i)
        cond_left_prom_dict[cond] = np.array(cond_proms)
    return cond_left_prom_dict, np.array(all_left_proms)

def define_prominence_threshold(all_left_proms, method="kde", k=3):
    if method=="kde":
        kde = gaussian_kde(all_left_proms)
        x_grid = np.linspace(all_left_proms.min(), all_left_proms.max(), 1000)
        pdf = kde.evaluate(x_grid)
        from scipy.signal import find_peaks
        p_peaks, _ = find_peaks(pdf)
        if len(p_peaks)==0:
            threshold = np.percentile(all_left_proms, 80)
        else:
            noise_peak_idx = p_peaks[0]
            noise_peak_x = x_grid[noise_peak_idx]
            threshold = noise_peak_x + (k*0.1)
        return threshold
    else:
        threshold = np.percentile(all_left_proms, 80)
        return threshold

def detect_events_with_leftprom_multi(params, n_repeats=5, leftprom_threshold=0.5, do_plot=True):
    conditions = ["NoNoise_NoLY", "NoNoise_+LY", "Noise_NoLY", "Noise_+LY"]
    mean_rates = []
    std_rates = []

    if do_plot:
        fig, axs = plt.subplots(2, 2, figsize=(12,6))
        fig.suptitle(f"Left-Prom Filter >= {leftprom_threshold:.2f} (HH-like gating)", fontsize=14)

    for idx, cond in enumerate(conditions):
        runs = run_condition_multiple_times(cond, params, n_repeats)
        all_rates = []

        # pick an example
        t_ex, V_ex, Ca_ex = runs[0]
        ex_peaks, _ = find_peaks(Ca_ex, height=0.0, prominence=0.0)
        ex_left_proms = compute_left_prominences(Ca_ex, ex_peaks)
        keep_mask = (ex_left_proms >= leftprom_threshold)
        ex_kept_peaks = ex_peaks[keep_mask]

        # compute rates across runs
        for (t_i, V_i, Ca_i) in runs:
            p_i, _ = find_peaks(Ca_i, height=0.0, prominence=0.0)
            L_i = compute_left_prominences(Ca_i, p_i)
            keep_i = (L_i >= leftprom_threshold)
            total_s = (t_i[-1]-t_i[0])/1000.0
            r_i = np.sum(keep_i)/total_s if total_s>0 else 0
            all_rates.append(r_i)

        mean_r = np.mean(all_rates)
        std_r = np.std(all_rates)
        mean_rates.append(mean_r)
        std_rates.append(std_r)

        if do_plot:
            row = idx//2
            col = idx%2
            ax = axs[row][col]

            # We'll do a twin axis for Voltage vs Ca
            ax.plot(t_ex, V_ex, 'b-', label="Voltage")
            ax.set_xlim(0, params["t_max"])
            (Vm_min, Vm_max, Ca_min, Ca_max) = params["plot_range"]
            ax.set_ylim(Vm_min, Vm_max)
            ax.set_title(f"{cond}: ex-run {len(ex_kept_peaks)/(params['t_max']/1000.0):.2f} ev/s, avg={mean_r:.2f}")

            ax2 = ax.twinx()
            ax2.plot(t_ex, Ca_ex, 'r-', label="Ca")
            ax2.plot(t_ex[ex_kept_peaks], Ca_ex[ex_kept_peaks], 'ko', label="Filtered events")
            ax2.set_ylim(Ca_min, Ca_max)

            if row==1:
                ax.set_xlabel("Time (ms)")
            if col==0:
                ax.set_ylabel("Voltage (mV)")
            ax2.set_ylabel("Calcium (a.u.)")

            # combined legend
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax2.legend(h1+h2, l1+l2, loc="upper right", fontsize=8)

    if do_plot:
        plt.tight_layout()
        plt.show()

    # bar chart
    plt.figure(figsize=(6,4))
    xvals = np.arange(len(conditions))
    plt.bar(xvals, mean_rates, yerr=std_rates, capsize=4, color='lightgreen', edgecolor='black')
    plt.xticks(xvals, conditions, rotation=45, ha='right')
    plt.ylabel("Mean Event Rate (ev/s)")
    plt.title(f"Left-Prom >= {leftprom_threshold:.2f}, HH-like gating")
    plt.tight_layout()
    plt.show()

    for cond, m_r, s_r in zip(conditions, mean_rates, std_rates):
        print(f"{cond} => {m_r:.2f} ± {s_r:.2f} events/sec (n={n_repeats})")

###############################################################################
#                                 MAIN                                        #
###############################################################################

if __name__ == "__main__":
    # 1) Single-run demonstration: show V & Ca side by side
    conditions_one_run = ["NoNoise_NoLY", "NoNoise_+LY", "Noise_NoLY", "Noise_+LY"]
    fig1, axs1 = plt.subplots(2,2, figsize=(12,6))
    fig1.suptitle("HH-like gating: Single-run demonstration (Voltage & Ca)")

    for idx, cond in enumerate(conditions_one_run):
        row= idx//2
        col= idx%2
        ax= axs1[row][col]

        t, V, Ca = run_simulation_HHlike(cond, params)

        # Plot Voltage on primary axis
        ax.plot(t, V, 'b-', label="Voltage")
        ax.set_xlim(0, params["t_max"])
        (Vm_min, Vm_max, Ca_min, Ca_max) = params["plot_range"]
        ax.set_ylim(Vm_min, Vm_max)
        ax.set_title(cond)

        # Plot Ca on secondary axis
        ax2 = ax.twinx()
        ax2.plot(t, Ca, 'r-', label="Calcium")
        ax2.set_ylim(Ca_min, Ca_max)

        if row==1:
            ax.set_xlabel("Time (ms)")
        if col==0:
            ax.set_ylabel("Voltage (mV)")
        ax2.set_ylabel("Calcium (a.u.)")

        # Combined legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax2.legend(h1+h2, l1+l2, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()

    # 2) Multi-run approach with direct find_peaks on Ca
    print("\n=== Multi-run approach with direct find_peaks (HH-like gating) ===")
    run_all_conditions_multi(params, n_repeats=5, prominence=0.05)

    # 3) Left-prom approach:
    print("\n=== Left-prominences approach with multiple runs (HH-like gating) ===")
    cond_lp_dict, all_lp = collect_all_left_prominences_multi(params, n_repeats=5)
    leftprom_threshold = define_prominence_threshold(all_lp, method="kde", k=3)
    print(f"Chosen left-prom threshold: {leftprom_threshold:.2f}")
    detect_events_with_leftprom_multi(params, n_repeats=5, leftprom_threshold=leftprom_threshold, do_plot=True)

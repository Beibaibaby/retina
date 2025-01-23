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
    "refresh_rate": 15,       # Hz => one new "frame" every ~66.7 ms
    "I_exc_per_bipolar": 0.08,# excitatory current contribution per ON bipolar

    # Membrane & Threshold
    "C_mem": 1.0,             # membrane capacitance
    "g_leak": 0.05,           # leak conductance
    "E_leak": -70.0,          # leak reversal (mV)
    "threshold_base": -56.0,  # base threshold for Ca event
    "threshold_shift_LY": -3.0, # how much we shift threshold if +LY

    # Calcium dynamics
    "tau_Ca": 200.0,          # decay time constant for Ca (ms)
    "I_Ca_in": 0.2,           # Ca influx once above threshold

    # Plotting ranges
    "plot_range": (-100, 0, -1.0, 50.0),
}

###############################################################################
#                           CORE MODEL FUNCTIONS                              #
###############################################################################

def get_threshold(condition, params):
    base = params["threshold_base"]
    if "+LY" in condition:
        return base + params["threshold_shift_LY"]
    else:
        return base

def pick_bipolars_active(condition, params):
    # If 'Noise' in condition, sample from N(5,5^2). Otherwise 0.
    if condition in ["Noise_NoLY", "Noise_+LY"]:
        val = np.random.normal(5.0, 5.0)
        val = int(np.clip(round(val), 0, params["n_bipolars"]))
        return val
    else:
        return 0

def run_simulation(condition, params):
    dt = params["dt"]
    steps = int(params["t_max"] // dt)
    time_arr = np.arange(0, params["t_max"], dt)

    frame_period = 1000.0 / params["refresh_rate"]
    frame_step_count = int(frame_period // dt)

    V = params["E_leak"]
    Ca = 0.0

    V_history = np.zeros(steps)
    Ca_history = np.zeros(steps)

    bipolars_on = 0
    step_in_frame = 0
    threshold = get_threshold(condition, params)

    for i in range(steps):
        if step_in_frame == 0:
            bipolars_on = pick_bipolars_active(condition, params)
        step_in_frame = (step_in_frame + 1) % frame_step_count

        I_exc = bipolars_on * params["I_exc_per_bipolar"]
        dV = (-params["g_leak"]*(V - params["E_leak"]) + I_exc)*(dt/params["C_mem"])
        V += dV

        if V >= threshold:
            dCa = -Ca / params["tau_Ca"] + params["I_Ca_in"]
        else:
            dCa = -Ca / params["tau_Ca"]
        Ca += dCa * dt

        V_history[i] = V
        Ca_history[i] = Ca

    return time_arr, V_history, Ca_history

###############################################################################
#                MULTIPLE INSTANCES PER CONDITION: AVERAGE RATE              #
###############################################################################

def run_condition_multiple_times(condition, params, n_repeats=5):
    """
    Runs the given condition n_repeats times, each for t_max ms.
    Returns a list of (time, voltage_array, calcium_array) for each run.
    """
    results = []
    for _ in range(n_repeats):
        t, V, Ca = run_simulation(condition, params)
        results.append((t, V, Ca))
    return results

def compute_event_rate(t, Ca, height=None, prominence=None):
    """
    Using a direct find_peaks approach: find # of peaks, then compute (#peaks)/(t_max in sec).
    """
    peaks, props = find_peaks(Ca, height=height, prominence=prominence)
    total_time_s = (t[-1] - t[0]) / 1000.0
    rate = len(peaks) / total_time_s if total_time_s>0 else 0
    return rate, peaks

def run_all_conditions_multi(params, n_repeats=5, height=None, prominence=None):
    """
    1) For each condition, we run n_repeats times.
    2) We compute the event rate (# events / total_time) for each run, then average.
    3) We also display a single example trace for plotting.
    4) Finally, we show a bar chart of average rates across conditions.
    """
    conditions = ["NoNoise_NoLY", "NoNoise_+LY", "Noise_NoLY", "Noise_+LY"]
    mean_rates = []
    std_rates = []

    # We'll store one example trace for each condition
    fig, axs = plt.subplots(2, 2, figsize=(12,6), sharex=True)
    fig.suptitle("One Example Trace per Condition + Detected Events", fontsize=14)

    for idx, cond in enumerate(conditions):
        row = idx // 2
        col = idx % 2
        ax = axs[row][col]

        # run n_repeats
        all_rates = []
        multi_results = run_condition_multiple_times(cond, params, n_repeats=n_repeats)

        # We'll pick the first run as the "example" trace
        t_ex, V_ex, Ca_ex = multi_results[0]
        # Detect peaks in the example
        ex_rate, ex_peaks = compute_event_rate(t_ex, Ca_ex, height=height, prominence=prominence)

        # Plot example Ca trace + peaks
        ax.plot(t_ex, Ca_ex, 'b-', label="Ca trace")
        ax.plot(t_ex[ex_peaks], Ca_ex[ex_peaks], 'ro', label="Detected peaks")
        ax.set_xlim(0, params["t_max"])
        (Vm_min, Vm_max, Ca_min, Ca_max) = params["plot_range"]
        ax.set_ylim(Ca_min, Ca_max)
        ax.set_title(f"{cond} (example run: {ex_rate:.2f} ev/s)")

        # compute rates for all runs
        for (t_i, V_i, Ca_i) in multi_results:
            r_i, p_i = compute_event_rate(t_i, Ca_i, height=height, prominence=prominence)
            all_rates.append(r_i)

        mean_r = np.mean(all_rates)
        std_r = np.std(all_rates)
        mean_rates.append(mean_r)
        std_rates.append(std_r)

        if row == 1:
            ax.set_xlabel("Time (ms)")
        if col == 0:
            ax.set_ylabel("Calcium (a.u.)")

        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

    # Now show bar chart of average rates
    plt.figure(figsize=(6,4))
    xvals = np.arange(len(conditions))
    plt.bar(xvals, mean_rates, yerr=std_rates, capsize=4, color='skyblue', edgecolor='black')
    plt.xticks(xvals, conditions, rotation=45, ha='right')
    plt.ylabel("Mean Calcium Event Rate (events/sec)")
    plt.title(f"Comparison of Ca Event Rates (n_repeats={n_repeats})")
    plt.tight_layout()
    plt.show()

    # Print numeric results
    for cond, m_r, s_r in zip(conditions, mean_rates, std_rates):
        print(f"{cond}: {m_r:.2f} ± {s_r:.2f} (events/sec) over {n_repeats} runs")

###############################################################################
#                  LEFT-PROMINENCE METHOD WITH MULTIPLE RUNS                 #
###############################################################################

def compute_left_prominences(trace, peaks):
    prominences, left_bases, _ = peak_prominences(trace, peaks)
    left_proms = trace[peaks] - trace[left_bases]
    return left_proms

def collect_all_left_prominences_multi(params, n_repeats=20):
    """
    For each condition, run n_repeats. For each run, detect all peaks.
    Gather left_prominences in a big global list. 
    Return them plus a dict condition->(list_of_left_proms).
    """
    conditions = ["NoNoise_NoLY", "NoNoise_+LY", "Noise_NoLY", "Noise_+LY"]
    all_left_proms = []
    cond_left_prom_dict = {}

    for cond in conditions:
        cond_proms = []
        multi_results = run_condition_multiple_times(cond, params, n_repeats=n_repeats)
        for (t_i, V_i, Ca_i) in multi_results:
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
        # find local maxima in pdf
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

def detect_events_with_leftprom_multi(params, n_repeats=20, leftprom_threshold=0.5, do_plot=True):
    """
    For each condition, run n_repeats. For each run:
     - detect all peaks
     - filter peaks by left_prom >= leftprom_threshold
    Then average the # of events over total time. 
    We'll plot one example run per condition with the filtered events shown.
    """
    conditions = ["NoNoise_NoLY", "NoNoise_+LY", "Noise_NoLY", "Noise_+LY"]
    mean_rates = []
    std_rates = []

    if do_plot:
        fig, axs = plt.subplots(2, 2, figsize=(12,6))
        fig.suptitle(f"Left-Prom Filter >= {leftprom_threshold:.2f} (Multi-run avg)", fontsize=14)

    for idx, cond in enumerate(conditions):
        multi_results = run_condition_multiple_times(cond, params, n_repeats=n_repeats)
        all_rates = []

        # pick 1 example
        t_ex, V_ex, Ca_ex = multi_results[0]
        example_peaks, _ = find_peaks(Ca_ex, height=0.0, prominence=0.0)
        ex_left_proms = compute_left_prominences(Ca_ex, example_peaks)
        keep_mask = (ex_left_proms >= leftprom_threshold)
        ex_kept_peaks = example_peaks[keep_mask]

        # compute rates across all runs
        for (t_i, V_i, Ca_i) in multi_results:
            all_peaks_i, _ = find_peaks(Ca_i, height=0.0, prominence=0.0)
            L_i = compute_left_prominences(Ca_i, all_peaks_i)
            keep_i = (L_i >= leftprom_threshold)
            # final # events is sum(keep_i)
            total_time_s = (t_i[-1]-t_i[0])/1000.0
            rate_i = np.sum(keep_i)/ total_time_s if total_time_s>0 else 0.0
            all_rates.append(rate_i)

        mean_r = np.mean(all_rates)
        std_r = np.std(all_rates)
        mean_rates.append(mean_r)
        std_rates.append(std_r)

        if do_plot:
            row = idx//2
            col = idx%2
            ax = axs[row][col]
            ax.plot(t_ex, Ca_ex, 'b-')
            ax.plot(t_ex[ex_kept_peaks], Ca_ex[ex_kept_peaks], 'ro', label=f"Kept (>= {leftprom_threshold:.2f})")
            ax.set_title(f"{cond} ex-run rate={len(ex_kept_peaks)/(params['t_max']/1000.0):.2f}, avg={mean_r:.2f}")
            ax.set_xlim(0, params["t_max"])
            (Vm_min, Vm_max, Ca_min, Ca_max) = params["plot_range"]
            ax.set_ylim(Ca_min, Ca_max)
            if row==1:
                ax.set_xlabel("Time (ms)")
            if col==0:
                ax.set_ylabel("Calcium (a.u.)")
            ax.legend(fontsize=8)

    if do_plot:
        plt.tight_layout()
        plt.show()

    # bar chart
    plt.figure(figsize=(6,4))
    xvals = np.arange(len(conditions))
    plt.bar(xvals, mean_rates, yerr=std_rates, capsize=4, color='lightgreen', edgecolor='black')
    plt.xticks(xvals, conditions, rotation=45, ha='right')
    plt.ylabel("Mean Event Rate (ev/s)")
    plt.title(f"Left-Prom >= {leftprom_threshold:.2f}, n={n_repeats} repeats")
    plt.tight_layout()
    plt.show()

    for cond, m_r, s_r in zip(conditions, mean_rates, std_rates):
        print(f"{cond} => {m_r:.2f} ± {s_r:.2f} events/sec (n={n_repeats})")

###############################################################################
#                              MAIN                                          #
###############################################################################

if __name__ == "__main__":

    # 1) Show one-run-per-condition (original approach)
    print("=== Single-run example for each condition ===")
    conditions_one_run = ["NoNoise_NoLY", "NoNoise_+LY", "Noise_NoLY", "Noise_+LY"]
    fig1, axs1 = plt.subplots(2,2, figsize=(12,6))
    fig1.suptitle("Single-run demonstration (Voltage & Ca)")

    for idx, cond in enumerate(conditions_one_run):
        row= idx//2
        col= idx%2
        ax= axs1[row][col]
        t, V, Ca = run_simulation(cond, params)
        ax.plot(t, V, 'b-', label="Voltage")
        ax2= ax.twinx()
        ax2.plot(t, Ca, 'r-', label="Calcium")

        ax.set_title(cond)
        ax.set_xlim(0, params["t_max"])
        (Vm_min, Vm_max, Ca_min, Ca_max) = params["plot_range"]
        ax.set_ylim(Vm_min, Vm_max)
        ax2.set_ylim(Ca_min, Ca_max)

        if row==1:
            ax.set_xlabel("Time (ms)")
        if col==0:
            ax.set_ylabel("Voltage (mV)")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    # 2) Multiple runs approach with direct find_peaks
    print("\n=== Multiple runs, direct find_peaks approach ===")
    run_all_conditions_multi(params, n_repeats=20, prominence=0.05)

    # 3) Left-prom approach with multiple runs
    print("\n=== Left-prominences approach with multiple runs ===")
    # Collect all left_proms across conditions & runs
    cond_lp_dict, all_lp = collect_all_left_prominences_multi(params, n_repeats=5)
    # define threshold
    threshold = define_prominence_threshold(all_lp, method="kde", k=3)
    print(f"Chosen left-prom threshold: {threshold:.2f}")

    # use that threshold to filter events
    detect_events_with_leftprom_multi(params, n_repeats=20, leftprom_threshold=threshold, do_plot=True)

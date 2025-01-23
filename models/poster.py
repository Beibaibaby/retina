import numpy as np
import matplotlib.pyplot as plt

###############################################################################
#                             MODEL PARAMETERS                                #
###############################################################################

params = {
    # Simulation timing
    "t_max": 5000.0,          # total simulation time (ms)
    "dt": 1.0,                # time step (ms)

    # Bipolar & Noise
    "n_bipolars": 10,         # total bipolars
    "active_bipolars": 5,     # how many bipolars are ON each frame if noise is ON
    "refresh_rate": 15,       # Hz => one new "frame" every ~66.7 ms
    "I_exc_per_bipolar": 0.08, # excitatory current contribution per ON bipolar

    # Membrane & Threshold
    "C_mem": 1.0,             # membrane capacitance
    "g_leak": 0.05,           # leak conductance
    "E_leak": -70.0,          # leak reversal (mV)
    "threshold_base": -55.0,  # base threshold for Ca event
    "threshold_shift_LY": -4.0,  # how much we shift threshold if +LY (sign depends on effect)

    # Calcium dynamics
    "tau_Ca": 200.0,          # decay time constant for Ca (ms)
    "I_Ca_in": 0.2,           # Ca influx once above threshold

    # For plotting
    "plot_range": (-100, 0, -1.0, 50.0),  # (Vm_min, Vm_max, Ca_min, Ca_max) for y-limits
}

###############################################################################
#                      FOUR CONDITIONS & HELPER FUNCTIONS                     #
###############################################################################


def get_threshold(condition, params):
    """
    If condition has '+LY', shift the threshold by threshold_shift_LY.
    Otherwise use threshold_base.
    """
    base = params["threshold_base"]
    if "+LY" in condition:
        # We'll assume + shift means a *higher* threshold 
        # (so it's "harder" to cross). Or you can do - shift if it’s easier.
        return base + params["threshold_shift_LY"]
    else:
        return base
import numpy as np




def pick_bipolars_active(condition, params):
    """
    If 'Noise' in condition, sample from a random distribution 
    such that the mean is ~5, but with large variance.
    Otherwise, return 0.
    """
    if condition in ["Noise_NoLY", "Noise_+LY"]:
        # Example: a normal(5, 2.5^2) => mean=5, stdev=2.5
        val = np.random.normal(5.0, 5.0)  # adjust stdev to get desired variance
        # Clamp and round to integer within [0, n_bipolars]
        val = int(np.clip(round(val), 0, params["n_bipolars"]))
        return val
    else:
        return 0

def run_simulation(condition, params):
    dt = params["dt"]
    steps = int(params["t_max"] // dt)
    time_arr = np.arange(0, params["t_max"], dt)

    frame_period = 1000.0 / params["refresh_rate"]  # ms per frame
    frame_step_count = int(frame_period // dt)

    V = params["E_leak"]
    Ca = 0.0

    V_history = np.zeros(steps)
    Ca_history = np.zeros(steps)

    # We'll keep track of how many bipolars are ON for the current frame
    bipolars_on = 0
    step_in_frame = 0

    # Precompute threshold for this condition (see earlier code)
    threshold = get_threshold(condition, params)

    for i in range(steps):
        # Possibly pick a new # of bipolars if we hit a new refresh
        if step_in_frame == 0:
            bipolars_on = pick_bipolars_active(condition, params)
        step_in_frame = (step_in_frame + 1) % frame_step_count

        # Excitatory input from bipolars
        I_exc = bipolars_on * params["I_exc_per_bipolar"]

        # Membrane voltage update (no spike reset)
        dV = (-params["g_leak"] * (V - params["E_leak"]) + I_exc) * (dt / params["C_mem"])
        V += dV

        # Calcium update
        if V >= threshold:
            dCa = -Ca / params["tau_Ca"] + params["I_Ca_in"]
        else:
            dCa = -Ca / params["tau_Ca"]
        Ca += dCa * dt

        V_history[i] = V
        Ca_history[i] = Ca

    return time_arr, V_history, Ca_history

def run_all_conditions(params):
    """
    We define 4 conditions:
      1) "NoNoise_NoLY"
      2) "NoNoise_+LY"
      3) "Noise_NoLY"
      4) "Noise_+LY"
    Then we run the simulation for each and plot results.
    """
    conditions = ["NoNoise_NoLY", "NoNoise_+LY", "Noise_NoLY", "Noise_+LY"]

    fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    fig.suptitle("Threshold-based Ca Events in SAC (No Reset)", fontsize=14)

    for idx, cond in enumerate(conditions):
        row = idx // 2
        col = idx % 2
        ax = axs[row][col]

        t, V, Ca = run_simulation(cond, params)

        # Plot membrane potential
        ax.plot(t, V, 'b-', label="Voltage (mV)")

        # Plot Ca on a secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(t, Ca, 'r-', label="[Ca] (arb units)")

        ax.set_title(cond)
        ax.set_xlim(0, params["t_max"])

        (Vm_min, Vm_max, Ca_min, Ca_max) = params["plot_range"]
        ax.set_ylim(Vm_min, Vm_max)
        ax2.set_ylim(Ca_min, Ca_max)

        if row == 1:
            ax.set_xlabel("Time (ms)")
        if col == 0:
            ax.set_ylabel("V (mV)")
        if col == 1:
            ax2.set_ylabel("[Ca] (a.u.)")

        # Make a single combined legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax2.legend(h1+h2, l1+l2, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def analyze_calcium_events(t, Ca, condition, height=None, prominence=None):
    """
    Detects peaks (Ca events) in a single trace (t, Ca) using scipy.signal.find_peaks.
    
    Parameters:
    -----------
    t : array-like
        Time array (ms).
    Ca : array-like
        Calcium trace (arbitrary units).
    condition : str
        Name of the condition (for labeling).
    height : float or None
        If specified, minimum height to accept a peak.
    prominence : float or None
        If specified, minimum prominence to accept a peak.
    
    Returns:
    --------
    peak_times : np.array
        Times (ms) at which Ca peaks occur.
    peak_values : np.array
        Calcium values at the peak points.
    peak_rate : float
        Number of peaks divided by total simulation time in seconds (peaks/sec).
    """

    # Use find_peaks to detect peaks with given thresholds
    # Adjust 'height' or 'prominence' if you want to filter out small events
    peaks, properties = find_peaks(Ca, height=height, prominence=prominence)
    
    peak_times = t[peaks]
    peak_values = Ca[peaks]
    
    # Compute rate in peaks/second
    total_time_s = (t[-1] - t[0]) / 1000.0  # convert ms -> sec
    peak_rate = len(peaks) / total_time_s if total_time_s > 0 else 0.0
    
    return peak_times, peak_values, peak_rate

def run_all_conditions_with_events(params, height=None, prominence=None):
    """
    1) Runs the four conditions from your model script.
    2) Detects Ca events using analyze_calcium_events.
    3) Stores and plots the event rates in a bar chart.
    
    The 'height' and 'prominence' arguments are optional parameters
    for scipy.signal.find_peaks. Adjust them to filter out small bumps.
    """
    conditions = ["NoNoise_NoLY", "NoNoise_+LY", "Noise_NoLY", "Noise_+LY"]
    event_rates = []
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    fig.suptitle("Calcium Traces + Detected Events", fontsize=14)
    
    for idx, cond in enumerate(conditions):
        row = idx // 2
        col = idx % 2
        ax = axs[row][col]
        
        # Run your existing simulation
        t, V, Ca = run_simulation(cond, params)
        
        # Plot the Ca trace
        ax.plot(t, Ca, 'b-', label="Ca trace")
        
        # Detect events
        peak_times, peak_values, peak_rate = analyze_calcium_events(
            t, Ca, cond,
            height=height, prominence=prominence
        )
        event_rates.append(peak_rate)
        
        # Mark detected peaks on the plot
        ax.plot(peak_times, peak_values, 'ro', label="Detected peaks")
        
        ax.set_title(f"{cond} (Rate={peak_rate:.2f} ev/s)")
        ax.set_xlim(0, params["t_max"])
        
        (Vm_min, Vm_max, Ca_min, Ca_max) = params["plot_range"]
        ax.set_ylim(Ca_min, Ca_max)  # reusing Ca-min & Ca-max from your code's range

        if row == 1:
            ax.set_xlabel("Time (ms)")
        if col == 0:
            ax.set_ylabel("[Ca] (a.u.)")

        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Now plot a bar chart of event rates
    plt.figure(figsize=(6, 4))
    plt.bar(conditions, event_rates, color='skyblue', edgecolor='black')
    plt.ylabel("Calcium Event Rate (events/sec)")
    plt.title("Comparison of Ca Event Rates Across Conditions")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Print the numeric results
    for cond, rate in zip(conditions, event_rates):
        print(f"{cond}: {rate:.2f} events/sec")




import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import gaussian_kde

###############################################################################
#                 STAGE 1: RUN ALL SIMS, COLLECT LEFT PROMINENCES            #
###############################################################################

def compute_left_prominences(trace, peaks):
    """
    For each peak, the 'left prominence' is peak_height - left_base_height,
    as returned by scipy.signal.peak_prominences. We'll compute them here.
    """
    # peak_prominences returns (prominences, left_bases, right_bases)
    prominences, left_bases, _ = peak_prominences(trace, peaks, wlen=None)
    left_proms = trace[peaks] - trace[left_bases]
    return left_proms

def collect_all_left_prominences(params):
    """
    1) Runs the four conditions (NoNoise_NoLY, NoNoise_+LY, Noise_NoLY, Noise_+LY).
    2) For each condition:
       - We'll find ALL Ca peaks (with minimal constraints).
       - We'll compute left_prominences for each peak.
    3) We return a dictionary with arrays of left_prominences for each condition,
       plus a combined list of all left_prominences across conditions.
    """
    conditions = ["NoNoise_NoLY", "NoNoise_+LY", "Noise_NoLY", "Noise_+LY"]
    all_left_proms = []
    cond_left_prom_dict = {}
    
    for cond in conditions:
        # Run your simulation for one condition
        t, V, Ca = run_simulation(cond, params)
        
        # Detect ALL peaks with minimal constraints, e.g. threshold=0.0
        peaks, _ = find_peaks(Ca, height=0.0, prominence=0.0)
        left_proms = compute_left_prominences(Ca, peaks)
        
        # Store in dictionary
        cond_left_prom_dict[cond] = left_proms
        # Collect for the global distribution
        all_left_proms.extend(left_proms)
    
    return cond_left_prom_dict, np.array(all_left_proms)

###############################################################################
#         STAGE 2: FIT NOISE OR DEFINE A THRESHOLD ON LEFT PROMINENCES       #
###############################################################################

def define_prominence_threshold(all_left_proms, method="kde", k=3):
    """
    Takes all left_prominences from all conditions, sets a threshold
    based on a 'noise distribution' approach or simply a percentile.

    method="kde" => use gaussian_kde, find a local maximum, then define threshold
    method="percentile" => do something simpler, e.g. 90th percentile
    k => how many std or how many "some measure" above the noise peak we go

    For demonstration, let's do:
      1) Fit a KDE
      2) Identify the 'noise peak' as the first local maximum
      3) Then define threshold = noise_peak_value + k * some small offset
    This is simplified, you can expand it or do the advanced method from your snippet.
    """
    if method == "kde":
        # Basic kernel density estimate
        kde = gaussian_kde(all_left_proms)
        
        # Evaluate the KDE on a grid
        x_grid = np.linspace(all_left_proms.min(), all_left_proms.max(), 1000)
        pdf = kde.evaluate(x_grid)
        
        # Find the highest local maximum near the lower end (the 'noise' peak)
        from scipy.signal import find_peaks
        peaks_indices, _ = find_peaks(pdf)
        # If none found, fallback
        if len(peaks_indices) == 0:
            # e.g. fallback: threshold = median + ...
            threshold = np.median(all_left_proms) + 0.5
        else:
            # pick the first peak as "noise" peak
            noise_peak_idx = peaks_indices[0]
            noise_peak_x   = x_grid[noise_peak_idx]
            threshold = noise_peak_x + (k * 0.1)  # 'k' is arbitrary scale, or we can do stdev in that region

        return threshold

    elif method == "percentile":
        # e.g. 80th percentile
        threshold = np.percentile(all_left_proms, 80)
        return threshold

    else:
        # fallback
        return np.percentile(all_left_proms, 80)

###############################################################################
# STAGE 3: RE-RUN DETECTION, FILTER BY LEFT PROM THRESH => COUNT EVENT RATES
###############################################################################

def detect_events_with_leftprom_filter(params, leftprom_threshold, do_plot=True):
    """
    We'll re-run each condition, find all peaks, compute left_prominences,
    then only keep those peaks whose left_prom >= leftprom_threshold.

    We'll compute an event rate (peaks above threshold) per second
    and optionally plot the filtered peaks on the Ca trace.
    """
    conditions = ["NoNoise_NoLY", "NoNoise_+LY", "Noise_NoLY", "Noise_+LY"]
    event_rates = []

    # optional figure
    if do_plot:
        fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
        fig.suptitle(f"Ca Traces + Filtered Events (left_prom >= {leftprom_threshold:.2f})", fontsize=14)

    for idx, cond in enumerate(conditions):
        # Run your simulation again
        t, V, Ca = run_simulation(cond, params)
        
        # find all peaks
        peaks, _ = find_peaks(Ca, height=0.0, prominence=0.0)
        left_proms = compute_left_prominences(Ca, peaks)
        
        # filter peaks by left_prom >= threshold
        keep_mask = (left_proms >= leftprom_threshold)
        kept_peaks = peaks[keep_mask]
        
        # compute event rate
        total_time_s = (t[-1] - t[0]) / 1000.0
        event_rate = len(kept_peaks) / total_time_s if total_time_s > 0 else 0.0
        event_rates.append(event_rate)

        # Plot if desired
        if do_plot:
            row = idx // 2
            col = idx % 2
            ax = axs[row][col]

            ax.plot(t, Ca, 'b-', label="Ca trace")
            ax.set_title(f"{cond}: rate={event_rate:.2f} ev/s")
            ax.set_xlim(0, params["t_max"])
            (Vm_min, Vm_max, Ca_min, Ca_max) = params["plot_range"]
            ax.set_ylim(Ca_min, Ca_max)

            # Mark kept peaks in red
            ax.plot(t[kept_peaks], Ca[kept_peaks], 'ro', label="Filtered events")
            if row == 1:
                ax.set_xlabel("Time (ms)")
            if col == 0:
                ax.set_ylabel("[Ca] (a.u.)")
            ax.legend(fontsize=8)

    if do_plot:
        plt.tight_layout()
        plt.show()

    return conditions, event_rates

###############################################################################
#                              MAIN EXAMPLE USAGE                             #
###############################################################################

def run_all_conditions_with_leftprom(params):
    """
    1) Gather all left_prominences from the 4 conditions.
    2) Fit a threshold using a 'kde' or 'percentile' approach.
    3) Re-run detection in each condition, keep only peaks with left_prom >= threshold.
    4) Plot event rates in a bar chart.
    """

    # --- STAGE 1: Collect left prominences ---
    cond_left_prom_dict, all_left_proms = collect_all_left_prominences(params)

    # --- STAGE 2: Define threshold ---
    leftprom_threshold = define_prominence_threshold(all_left_proms, method="kde", k=3.0)
    print(f"Chosen Left-Prom Threshold = {leftprom_threshold:.2f}")

    # --- STAGE 3: Re-run detection with final threshold, compute event rates ---
    conditions, event_rates = detect_events_with_leftprom_filter(params, leftprom_threshold, do_plot=True)

    # Show bar chart comparing final event rates
    plt.figure(figsize=(6,4))
    plt.bar(conditions, event_rates, color='lightgreen', edgecolor='black')
    plt.xlabel("Condition")
    plt.ylabel("Event Rate (events/sec)")
    plt.title("Final Ca Event Rates (Filtered by Left Prominence)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Print results
    for cond, rate in zip(conditions, event_rates):
        print(f"{cond} => {rate:.2f} events/sec")

if __name__ == "__main__":
    run_all_conditions(params)
    run_all_conditions_with_events(params, prominence=0.05)
    run_all_conditions_with_leftprom(params)

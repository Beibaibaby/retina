import numpy as np
import matplotlib.pyplot as plt

###############################################################################
#                          MODEL PARAMETERS                                   #
###############################################################################

params = {
    # Simulation
    "t_max": 2000.0,     # total simulation time (ms)
    "dt": 1.0,           # time step (ms)

    # Stimulus & Visual Input
    "refresh_rate": 15,      # Hz, refresh of noise / motion
    "pixel_prob_on": 0.3,    # probability pixel is ON in random flicker
    "grid_size": (5, 5),     # small grid for demonstration
    "bar_width_pixels": 2,   # width of bar in pixels
    "motion_speeds": [2, 8, 15],  # example speeds in pixel-steps per refresh
    "n_directions": 8,       # number of directions to simulate (0 to 360)
    "gray_background": 0.0,  # 0 => all OFF for "noise-free"

    # Bipolar -> SAC
    "alpha_exc": 2.0,
    # Bipolar -> DSGC
    "beta_exc": 0.2,

    # SAC Membrane
    "C_SAC": 1.0,
    "g_leak_SAC": 0.02,
    "E_leak_SAC": -60.0,
    "Vth_base_SAC": -50.0,   # baseline threshold for Ca event
    "E_inh": -70.0,          # inhibitory reversal potential

    # Lateral Inhibition among SACs
    "g_SAC_SAC": 0.001,
    "tau_rec_SAC_SAC": 200.0,  # short-term recovery (ms)
    "U_SAC_SAC": 0.1,          # release prob factor

    # SAC -> DSGC Inhibition
    "g_SAC_DSGC": 0.05,
    "tau_rec_SAC_DSGC": 200.0,
    "U_SAC_DSGC": 0.1,

    # DSGC Membrane (Leaky Integrate-and-Fire)
    "C_DSGC": 1.0,
    "g_leak_DSGC": 0.02,
    "E_leak_DSGC": -60.0,
    "V_th_DSGC": -50.0,
    "V_reset_DSGC": -65.0,

    # mGluR2 threshold shift
    "delta_th_mGluR2": -15.0,  # shift in mV that lowers threshold
}

###############################################################################
#                      STIMULUS GENERATION FUNCTIONS                          #
###############################################################################

def generate_noise_frame(grid_size, p_on=0.3):
    return (np.random.rand(*grid_size) < p_on).astype(float)

def generate_gray_frame(grid_size, intensity=0.0):
    return np.ones(grid_size) * intensity

def generate_bar_frame(grid_size, bar_width, bar_pos, direction, background_frame):
    """
    Very simplified motion approach for demonstration:
      direction in [0, 180) => horizontal bar 
      direction in [90, 270) => vertical bar, etc.
    For fully general angles, you'd do real geometry or a line raster approach.
    """
    frame = background_frame.copy()
    rows, cols = grid_size

    # We do a crude direction approach: 0 = left->right, 90= top->down, 180= right->left, 270= bottom->up
    dir_deg = direction % 360

    if 0 <= dir_deg < 90:
        # "right"
        col_start = int(bar_pos)
        col_end = int(bar_pos + bar_width)
        col_end = min(col_end, cols)
        if col_start < cols:
            frame[:, col_start:col_end] = 1.0

    elif 90 <= dir_deg < 180:
        # "down"
        row_start = int(bar_pos)
        row_end = int(bar_pos + bar_width)
        row_end = min(row_end, rows)
        if row_start < rows:
            frame[row_start:row_end, :] = 1.0

    elif 180 <= dir_deg < 270:
        # "left"
        col_end = int(cols - bar_pos)
        col_start = int(cols - bar_pos - bar_width)
        col_start = max(col_start, 0)
        if col_start < col_end:
            frame[:, col_start:col_end] = 1.0

    else:
        # "up"
        row_end = int(rows - bar_pos)
        row_start = int(rows - bar_pos - bar_width)
        row_start = max(row_start, 0)
        if row_start < row_end:
            frame[row_start:row_end, :] = 1.0

    return frame


###############################################################################
#                   SHORT-TERM DEPRESSION HELPER FUNCTION                     #
###############################################################################

def short_term_depression_update(W, U, tau_rec, dt, pre_spike):
    """
    Simple short-term depression model:
    dW/dt = (1 - W)/tau_rec - U * W * delta_spike
    """
    W0 = 1.0
    dW = (W0 - W) / tau_rec
    if pre_spike:
        dW -= U * W
    return W + dW * (dt / 1000.0)


###############################################################################
#                     CORE SIMULATION OF SAC-SAC-DSGC                         #
###############################################################################

def simulate_SAC_SAC_DSGC(params,
                          condition="normal",
                          background="noise-free",
                          direction=0,
                          speed=15,
                          visualize=False):
    dt = params["dt"]
    t_max = params["t_max"]
    steps = int(t_max // dt)
    time_arr = np.arange(0, t_max, dt)

    refresh_period = 1000.0 / params["refresh_rate"]  # ms per frame

    # 2 SACs for demonstration
    nSAC = 2
    adjacency = np.ones((nSAC, nSAC)) - np.eye(nSAC)  # fully connected except self
    if condition == "cKO":
        adjacency = np.zeros((nSAC, nSAC))

    if condition == "mGluR2_block":
        delta_th = 0.0
    else:
        delta_th = params["delta_th_mGluR2"]

    V_SAC = np.ones(nSAC) * params["E_leak_SAC"]
    W_SAC_SAC = np.ones((nSAC, nSAC))
    W_SAC_DSGC = np.ones(nSAC)

    V_DSGC = params["E_leak_DSGC"]
    spike_times = []

    V_SAC_history = np.zeros((steps, nSAC))
    V_DSGC_history = np.zeros(steps)

    frame_history = []

    bar_pos = -params["bar_width_pixels"]  # start off-screen

    # Noise or not
    if background == "noise-free":
        p_flicker = 0.0
    else:
        p_flicker = params["pixel_prob_on"]

    for s in range(steps):
        t = s * dt

        if s % int(refresh_period//dt) == 0:
            if background == "noise-free":
                bg_frame = generate_gray_frame(params["grid_size"],
                                               intensity=params["gray_background"])
            else:
                bg_frame = generate_noise_frame(params["grid_size"],
                                                p_flicker)
            # Add bar
            frame = generate_bar_frame(params["grid_size"],
                                       params["bar_width_pixels"],
                                       bar_pos,
                                       direction,
                                       bg_frame)
            frame_history.append(frame)
            bar_pos += speed  # move bar each refresh

        current_frame = frame_history[-1]

        # 1) Bipolar -> SAC
        n_ON = np.sum(current_frame)
        I_exc_SAC = params["alpha_exc"] * n_ON

        # 2) SAC-SAC Inhibition
        I_inh_SAC = np.zeros(nSAC)
        for i in range(nSAC):
            for j in range(nSAC):
                if adjacency[i,j] > 0:
                    syn_w = W_SAC_SAC[i,j]
                    I_inh_SAC[i] += (params["g_SAC_SAC"] * syn_w *
                                     (V_SAC[i] - params["E_inh"]))

        # 3) Update SAC membrane
        dV_SAC = (-params["g_leak_SAC"] * (V_SAC - params["E_leak_SAC"])
                  + I_exc_SAC - I_inh_SAC) * (dt / params["C_SAC"])
        V_SAC += dV_SAC

        # 4) Threshold crossing
        spike_SAC = (V_SAC >= (params["Vth_base_SAC"] + delta_th))

        # 5) Synaptic depression update
        for i in range(nSAC):
            if spike_SAC[i]:
                # i -> j
                for j in range(nSAC):
                    if adjacency[i,j] > 0:
                        W_SAC_SAC[i,j] = short_term_depression_update(
                            W_SAC_SAC[i,j],
                            params["U_SAC_SAC"],
                            params["tau_rec_SAC_SAC"],
                            dt,
                            pre_spike=True
                        )
                # i -> DSGC
                W_SAC_DSGC[i] = short_term_depression_update(
                    W_SAC_DSGC[i],
                    params["U_SAC_DSGC"],
                    params["tau_rec_SAC_DSGC"],
                    dt,
                    pre_spike=True
                )
            else:
                # recover
                for j in range(nSAC):
                    if adjacency[i,j] > 0:
                        W_SAC_SAC[i,j] = short_term_depression_update(
                            W_SAC_SAC[i,j],
                            params["U_SAC_SAC"],
                            params["tau_rec_SAC_SAC"],
                            dt,
                            pre_spike=False
                        )
                W_SAC_DSGC[i] = short_term_depression_update(
                    W_SAC_DSGC[i],
                    params["U_SAC_DSGC"],
                    params["tau_rec_SAC_DSGC"],
                    dt,
                    pre_spike=False
                )

        # 6) Bipolar -> DSGC
        I_exc_DSGC = params["beta_exc"] * n_ON

        # 7) Inhibitory from SAC to DSGC
        I_inh_DSGC = 0.0
        for i in range(nSAC):
            if spike_SAC[i]:
                I_inh_DSGC += (params["g_SAC_DSGC"] * W_SAC_DSGC[i] *
                               (V_DSGC - params["E_inh"]))

        # 8) DSGC update
        dV_DSGC = (-params["g_leak_DSGC"] * (V_DSGC - params["E_leak_DSGC"])
                   + I_exc_DSGC - I_inh_DSGC) * (dt / params["C_DSGC"])
        V_DSGC += dV_DSGC

        # 9) Check spike
        if V_DSGC >= params["V_th_DSGC"]:
            spike_times.append(t)
            V_DSGC = params["V_reset_DSGC"]

        V_SAC_history[s, :] = V_SAC
        V_DSGC_history[s] = V_DSGC

    results = {
        "time": time_arr,
        "V_SAC": V_SAC_history,
        "V_DSGC": V_DSGC_history,
        "DSGC_spike_times": np.array(spike_times),
        "spike_count": len(spike_times),
    }

    # Optional plot: waveforms for one run
    if visualize:
        fig, axs = plt.subplots(3, 1, figsize=(8,9), sharex=True)
        axs[0].plot(time_arr, V_SAC_history[:,0], label="SAC #0")
        axs[0].plot(time_arr, V_SAC_history[:,1], label="SAC #1")
        axs[0].set_ylabel("SAC Vm (mV)")
        axs[0].legend()
        axs[0].set_title(f"{condition}, {background}, dir={direction}, speed={speed}")

        axs[1].plot(time_arr, V_DSGC_history, 'g', label="DSGC Vm")
        axs[1].set_ylabel("DSGC Vm (mV)")
        axs[1].legend()

        spike_raster_y = np.zeros_like(results["DSGC_spike_times"])
        axs[2].scatter(results["DSGC_spike_times"], spike_raster_y,
                       marker='|', color='r')
        axs[2].set_ylabel("Spike Raster")
        axs[2].set_xlabel("Time (ms)")

        plt.tight_layout()
        plt.show()

    return results

###############################################################################
#                      DIRECTION SELECTIVITY INDEX                            #
###############################################################################

def direction_selectivity_index(spike_counts, directions_deg):
    dirs_rad = np.deg2rad(directions_deg)
    x = np.sum(spike_counts * np.cos(dirs_rad))
    y = np.sum(spike_counts * np.sin(dirs_rad))
    vector_mag = np.sqrt(x**2 + y**2)
    total_spikes = np.sum(spike_counts)
    if total_spikes > 0:
        return vector_mag / total_spikes
    else:
        return 0.0

###############################################################################
#                   SYSTEMATIC TESTING + ENHANCED PLOTTING                    #
###############################################################################

import numpy as np
import matplotlib.pyplot as plt

def generate_mock_epsc(condition, n_points=2000):
    """
    Generate a mock EPSC trace for demonstration.
    In a real script, you'd load your actual EPSC data.
    """
    time = np.linspace(0, 2, n_points)  # 2 seconds for example
    # Create a baseline
    epsc = -5 + np.random.normal(0, 0.5, n_points)

    # Add a stimulus-evoked "dip" or "peak" around 0.5 - 1.0 s for demonstration
    stim_start = int(n_points * 0.3)
    stim_end = int(n_points * 0.6)

    # A quick hack: each condition can have a slightly different shape
    if condition == "Bar":
        epsc[stim_start:stim_end] -= 5
    elif condition == "Bar + LY":
        epsc[stim_start:stim_end] -= 4
    elif condition == "Noise":
        epsc[stim_start:stim_end] -= 6
    elif condition == "Noise + LY":
        epsc[stim_start:stim_end] -= 5.5

    return time, epsc

def generate_mock_df_f0(condition, n_points=1000, n_traces=10):
    """
    Generate a set of 'dF/F0' traces for demonstration in either the
    preferred or null direction. We'll just randomize them around 
    some typical shape.
    """
    time = np.linspace(0, 2, n_points)  # 2 seconds
    # We'll store multiple "repeats" or "cells" as separate traces
    data_traces = []

    for _ in range(n_traces):
        # Baseline near 0
        trace = np.random.normal(0, 0.02, n_points)
        
        # A "peak" around 0.3 - 0.6 s
        stim_start = int(n_points * 0.15)
        stim_end   = int(n_points * 0.35)

        if "Noise" in condition:
            # Noisy or smaller/longer peak
            trace[stim_start:stim_end] += np.random.uniform(0.1, 0.3)
        else:
            # Bar conditions => a bigger peak
            trace[stim_start:stim_end] += np.random.uniform(0.4, 1.0)
        
        # If it's a LY condition, let's reduce the amplitude slightly
        if "+ LY" in condition:
            trace[stim_start:stim_end] *= 0.8
        
        data_traces.append(trace)

    return time, np.array(data_traces)

def plot_poster_figure():
    """
    Create a figure with:
      - 4 rows (one for each condition)
      - 3 columns: 
         1) EPSC trace
         2) normalized dF/F0 in preferred direction
         3) normalized dF/F0 in null direction
    """
    conditions = ["Bar", "Bar + LY", "Noise", "Noise + LY"]

    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(10, 10), sharex=False)
    fig.suptitle("The role of activity-dependent mGluR2 signaling in SAC direction selectivity",
                 fontsize=14, y=0.98)

    for row, cond in enumerate(conditions):
        # 1) EPSC
        ax_epsc = axs[row, 0]
        t_epsc, epsc_trace = generate_mock_epsc(cond)
        ax_epsc.plot(t_epsc, epsc_trace, color='black', linewidth=1.0)
        ax_epsc.set_ylabel("EPSC (pA)")
        # Label the row on the left
        ax_epsc.set_title(cond if row == 0 else "", fontsize=11)
        ax_epsc.set_xlim(0, t_epsc[-1])
        ax_epsc.set_ylim(epsc_trace.min() - 5, epsc_trace.max() + 5)

        # Example axis annotation
        if row == 3:  # bottom row
            ax_epsc.set_xlabel("Time (s)")

        # 2) dF/F0 in preferred direction
        ax_pref = axs[row, 1]
        t_df, df_data_pref = generate_mock_df_f0(cond, n_points=1000, n_traces=10)
        for trace in df_data_pref:
            ax_pref.plot(t_df, trace, alpha=0.8)
        ax_pref.set_xlim(0, t_df[-1])
        ax_pref.set_ylim(-0.2, 1.2)  # based on the random data
        if row == 3:
            ax_pref.set_xlabel("Time (s)")
        if row == 0:
            ax_pref.set_title("Normalized dF/F0\n(Preferred Direction)")

        # 3) dF/F0 in null direction
        ax_null = axs[row, 2]
        # Let's just reuse the same code (but "null direction" might have a smaller peak).
        # We'll reduce it slightly to illustrate a difference.
        t_df_null, df_data_null = generate_mock_df_f0(cond, n_points=1000, n_traces=10)
        # Scale them smaller for "null" direction
        df_data_null *= 0.5  
        for trace in df_data_null:
            ax_null.plot(t_df_null, trace, alpha=0.8)
        ax_null.set_xlim(0, t_df_null[-1])
        ax_null.set_ylim(-0.2, 1.2)
        if row == 3:
            ax_null.set_xlabel("Time (s)")
        if row == 0:
            ax_null.set_title("Normalized dF/F0\n(Null Direction)")

    plt.tight_layout()
    plt.show()



def run_systematic_tests(params):
    """
    1) We systematically simulate:
        conditions = [normal, cKO, mGluR2_block]
        backgrounds = [noise-free, noisy]
        speeds = [2, 8, 15]  (or your preference)
        directions = 8 directions from 0..315 deg
    2) For each combination, we store the spike counts across directions.
    3) We do:
       - Print numeric results to console
       - Create polar plot of direction tuning
       - Create bar plot of DSIs vs speed
       - Plot waveforms for a single example direction in all conditions
    """
    conditions = ["normal", "cKO", "mGluR2_block"]
    backgrounds = ["noise-free", "noisy"]
    speeds = params["motion_speeds"]
    directions = np.linspace(0, 360, params["n_directions"], endpoint=False)

    # We'll store results in nested dict
    # results_dict[cond][bg][speed] = {
    #   "spike_counts": array of shape [n_directions],
    #   "DSI": scalar
    # }
    results_dict = {}
    for cond in conditions:
        results_dict[cond] = {}
        for bg in backgrounds:
            results_dict[cond][bg] = {}
            for spd in speeds:
                spike_counts = []
                for d in directions:
                    out = simulate_SAC_SAC_DSGC(
                        params, condition=cond, background=bg,
                        direction=d, speed=spd, visualize=False
                    )
                    spike_counts.append(out["spike_count"])
                spike_counts = np.array(spike_counts)
                DSI = direction_selectivity_index(spike_counts, directions)
                results_dict[cond][bg][spd] = {
                    "spike_counts": spike_counts,
                    "DSI": DSI
                }

    #---------------------------------------------------------------
    # 1) Print the numeric results
    print("\n=== SYSTEMATIC TEST RESULTS ===")
    for cond in conditions:
        for bg in backgrounds:
            for spd in speeds:
                spike_counts = results_dict[cond][bg][spd]["spike_counts"]
                DSI = results_dict[cond][bg][spd]["DSI"]
                print(f"Condition={cond}, BG={bg}, speed={spd} =>")
                print(f"   Directions={directions}")
                print(f"   SpikeCounts={spike_counts}")
                print(f"   DSI={DSI:.3f}")

    #---------------------------------------------------------------
    # 2) Detailed waveforms for one "example" direction across all conditions
    #    so we can see how each condition changes the SAC & DSGC traces
    #    We'll do direction=0 as an example
    example_direction = 0
    fig_ex, axs_ex = plt.subplots(len(conditions), len(backgrounds),
                                  figsize=(12, 8), sharex=True, sharey=True)
    fig_ex.suptitle(f"Waveforms at direction={example_direction} for all Conditions/Backgrounds")

    for i, cond in enumerate(conditions):
        for j, bg in enumerate(backgrounds):
            ax = axs_ex[i][j]
            # We'll simulate again with "visualize=False" but then plot inside this multi-panel figure
            out = simulate_SAC_SAC_DSGC(params, cond, bg,
                                        direction=example_direction,
                                        speed=speeds[1],  # pick the middle speed
                                        visualize=False)
            t = out["time"]
            V_SAC = out["V_SAC"]
            V_DSGC = out["V_DSGC"]
            spikes = out["DSGC_spike_times"]

            ax.plot(t, V_SAC[:,0], label="SAC#0")
            ax.plot(t, V_SAC[:,1], label="SAC#1")
            ax.plot(t, V_DSGC, label="DSGC")
            for st in spikes:
                ax.axvline(st, color='r', linestyle='--', alpha=0.5)
            ax.set_title(f"{cond}, {bg}")
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

    #---------------------------------------------------------------
    # 3) Polar (tuning) plots of directions for each condition + background + speed
    #    We'll make separate figures for each speed
    for spd in speeds:
        fig_pol, axs_pol = plt.subplots(len(conditions), len(backgrounds),
                                        figsize=(12, 10),
                                        subplot_kw={"projection": "polar"})
        fig_pol.suptitle(f"Polar Tuning Curves at speed={spd}")
        for i, cond in enumerate(conditions):
            for j, bg in enumerate(backgrounds):
                axp = axs_pol[i][j]
                spike_counts = results_dict[cond][bg][spd]["spike_counts"]
                # close the curve
                dirs_rad = np.deg2rad(directions)
                dirs_rad_cl = np.concatenate([dirs_rad, [dirs_rad[0]]])
                sc_cl = np.concatenate([spike_counts, [spike_counts[0]]])
                axp.plot(dirs_rad_cl, sc_cl, marker='o')
                axp.set_title(f"{cond}, {bg}", va='bottom')
                axp.set_theta_zero_location("E")
                axp.set_theta_direction(-1)

        plt.tight_layout()
        plt.show()

    #---------------------------------------------------------------
    # 4) Bar plots for DSIs across speeds
    fig_dsi, axs_dsi = plt.subplots(1, len(backgrounds), figsize=(12,5), sharey=True)
    fig_dsi.suptitle("Direction Selectivity Indices vs Speed")
    xvals = np.arange(len(speeds))
    width = 0.25

    for k, bg in enumerate(backgrounds):
        axd = axs_dsi[k]
        dsis_allconds = []
        for cond in conditions:
            ds = []
            for spd in speeds:
                ds.append(results_dict[cond][bg][spd]["DSI"])
            dsis_allconds.append(ds)

        # dsis_allconds is shape [len(conditions), len(speeds)]
        for i, cond in enumerate(conditions):
            offset = (i - 1) * width
            axd.bar(xvals + offset, dsis_allconds[i],
                    width=width, label=cond)
        axd.set_xticks(xvals)
        axd.set_xticklabels(speeds)
        axd.set_title(bg)
        axd.set_xlabel("Speed (pixels/refresh)")
        if k == 0:
            axd.set_ylabel("DSI")
        axd.legend()

    plt.tight_layout()
    plt.show()

###############################################################################
#                                  MAIN                                       #
###############################################################################

if __name__ == "__main__":
    plot_poster_figure()
    # Example single run & plot
    #_ = simulate_SAC_SAC_DSGC(params, condition="normal", background="noise-free", direction=0, speed=8, visualize=True)

    # Now run the full systematic testing with printing + multi-figure output
    #run_systematic_tests(params)

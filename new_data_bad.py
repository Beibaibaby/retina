from scipy.signal import find_peaks, peak_prominences
from sklearn.mixture import GaussianMixture
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# Function to compute left prominence
def compute_left_prominences(trace, peaks):
    prominences, left_bases, _ = peak_prominences(trace, peaks, wlen=300)
    left_prominences = trace[peaks] - trace[left_bases]  # Peak height - left base height
    return left_prominences

# Load the dataset
file_path = 'ca_imaging_traces.mat'  # Update this path to where your file is located
data = sio.loadmat(file_path)

# Define datasets for threshold computation and full processing
threshold_datasets = [
    'plot_all_rs_norm_null1_traces_final',
    'plot_all_rs_norm_pref1_traces_final'
]
all_datasets = [
    'plot_all_rs_norm_null1_traces_final',
    'plot_all_rs_norm_null2_traces_final',
    'plot_all_rs_norm_null4_traces_final',
    'plot_all_rs_norm_pref1_traces_final',
    'plot_all_rs_norm_pref2_traces_final',
    'plot_all_rs_norm_pref4_traces_final'
]

# Collect left prominences after 2 seconds for threshold computation
all_left_prominences_for_threshold = []
time_threshold = 2.0  # Time threshold in seconds

for dataset_name in threshold_datasets:
    trace_data = data[dataset_name]
    time_axis = data['rs_time_axis'].squeeze()

    for i in range(trace_data.shape[1]):
        trace = trace_data[:, i]
        peaks, _ = find_peaks(
            trace,
            height=None,
            threshold=0.0001,
            prominence=0.01
        )
        # Compute left prominences
        left_prominences = compute_left_prominences(trace, peaks)
        peaks_after_2s = peaks[time_axis[peaks] > time_threshold]
        left_prominences_after_2s = left_prominences[time_axis[peaks] > time_threshold]
        all_left_prominences_for_threshold.extend(left_prominences_after_2s)

# Fit GMM to collected left prominences for threshold computation
left_prominences_array = np.array(all_left_prominences_for_threshold).reshape(-1, 1)
gmm = GaussianMixture(n_components=1, random_state=42).fit(left_prominences_array)

# Extract GMM parameters
noise_mean = np.min(gmm.means_)
noise_std = np.sqrt(np.min(gmm.covariances_))
signal_mean = np.max(gmm.means_)
signal_std = np.sqrt(np.max(gmm.covariances_))
prominence_threshold = signal_mean + signal_std  # Global threshold for signal

# Visualize GMM fit on prominence histogram
plt.figure(figsize=(10, 6))
plt.hist(all_left_prominences_for_threshold, bins=100, color='skyblue', edgecolor='black', alpha=0.7, density=True)
x = np.linspace(min(all_left_prominences_for_threshold), max(all_left_prominences_for_threshold), 1000).reshape(-1, 1)
gmm_density = np.exp(gmm.score_samples(x))
plt.plot(x, gmm_density, 'k-', label='GMM Fit')
plt.axvline(prominence_threshold, color='purple', linestyle='--', label='Prominence Threshold')
plt.xlabel('Left Prominence')
plt.ylabel('Density')
plt.title('Left Prominence Distribution for Threshold Computation (GMM Fit)')
plt.legend()
plt.savefig("threshold_left_prominence_distribution_gmm.png")
plt.show()

print(f"Noise Mean: {noise_mean:.4f}, Noise Std: {noise_std:.4f}")
print(f"Signal Mean: {signal_mean:.4f}, Signal Std: {signal_std:.4f}")
print(f"Prominence Threshold: {prominence_threshold:.4f}")

# Store mean and std of spike rates for comparison later
spike_rate_means = []
spike_rate_stds = []

# Analyze each dataset using the global threshold
for dataset_name in all_datasets:
    trace_data = data[dataset_name]
    time_axis = data['rs_time_axis'].squeeze()

    # Plotting six traces in a 2x3 grid of subplots with left prominence visualization
    plt.figure(figsize=(15, 8))
    for i in range(6):  # Plot the first six traces
        trace = trace_data[:, i]
        peaks, _ = find_peaks(trace, prominence=prominence_threshold)
        peaks_after_2s = peaks[time_axis[peaks] > time_threshold]
        left_prominences = compute_left_prominences(trace, peaks)
        left_prominences_after_2s = left_prominences[time_axis[peaks] > time_threshold]

        # Adding subplot for each trace
        plt.subplot(2, 3, i + 1)
        plt.plot(time_axis, trace, label=f'Trial {i+1}')
        plt.plot(time_axis[peaks_after_2s], trace[peaks_after_2s], 'ro', label='Spikes (Time > 2 s)', markersize=4)
        
        # Annotate left prominences
        for peak, prom in zip(peaks_after_2s, left_prominences_after_2s):
            plt.vlines(x=time_axis[peak], ymin=trace[peak] - prom, ymax=trace[peak], color='blue', linestyle='--', linewidth=1)
            plt.text(time_axis[peak], trace[peak], f'{prom:.2f}', fontsize=6, color='blue', ha='center')
        
        plt.xlabel('Time (ms)')
        plt.ylabel('Trace Intensity')
        plt.title(f'{dataset_name} - Trial {i+1} (Left Prominence)')
        plt.legend()
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_traces_with_left_prominence.png")
    plt.close()

    # Initialize lists to collect stats
    all_isis = []  # For inter-spike intervals
    spike_rates = []  # For mean spike rate per trace

    # Loop through each trace and compute statistics
    for i in range(trace_data.shape[1]):
        trace = trace_data[:, i]
        peaks, _ = find_peaks(trace, prominence=prominence_threshold)
        peaks_after_2s = peaks[time_axis[peaks] > time_threshold]
        
        # Compute ISIs in milliseconds
        isi = np.diff(time_axis[peaks_after_2s])  # Difference between successive peak times
        all_isis.extend(isi)  # Append ISIs from each trace
        spike_rate = len(peaks_after_2s) / (time_axis[-1] - time_axis[0]) * 1000  # Spikes per second
        spike_rates.append(spike_rate)  # Collect spike rate for each trace

    # Calculate mean and standard deviation of spike rates for the dataset
    mean_spike_rate = np.mean(spike_rates)
    std_spike_rate = np.std(spike_rates)
    spike_rate_means.append(mean_spike_rate)
    spike_rate_stds.append(std_spike_rate)

    # Plot and save the histogram of ISIs for the current dataset
    plt.figure(figsize=(10, 6))
    plt.hist(all_isis, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('Inter-Spike Interval (ms)')
    plt.ylabel('Frequency')
    plt.title(f'Inter-Spike Interval Histogram for {dataset_name}')
    plt.savefig(f"{dataset_name}_isi_histogram.png")
    plt.close()

    # Print mean and standard deviation of spike rates for current dataset
    print(f"{dataset_name} - Mean Spike Rate (spikes per second): {mean_spike_rate:.2f}")
    print(f"{dataset_name} - Standard Deviation of Spike Rate: {std_spike_rate:.2f}")

# Visualization of mean spike rates with standard deviation for comparison across datasets
plt.figure(figsize=(10, 6))
plt.bar(all_datasets, spike_rate_means, yerr=spike_rate_stds, capsize=5, color='lightgreen', edgecolor='black')
plt.ylabel('Mean Spike Rate (spikes per second)')
plt.xlabel('Dataset')
plt.title('Mean Spike Rate with Standard Deviation across Datasets')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("spike_rate_comparison.png")
plt.show()
# Import necessary libraries
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
import numpy as np
from sklearn.mixture import GaussianMixture

# Load the dataset
file_path = 'ca_imaging_traces.mat'  # Update this path to where your file is located
data = sio.loadmat(file_path)

# List of datasets to process
datasets = [
    'plot_all_rs_norm_null1_traces_final',
    'plot_all_rs_norm_null2_traces_final',
    'plot_all_rs_norm_null4_traces_final',
    'plot_all_rs_norm_pref1_traces_final',
    'plot_all_rs_norm_pref2_traces_final',
    'plot_all_rs_norm_pref4_traces_final'
]

# Collect all prominences after 2 seconds from all datasets
all_prominences_after_2s = []
time_threshold = 2.0  # Time threshold in seconds

for dataset_name in datasets:
    trace_data = data[dataset_name]
    time_axis = data['rs_time_axis'].squeeze()

    for i in range(trace_data.shape[1]):
        trace = trace_data[:, i]
        peaks, _ = find_peaks(trace)  # Initial peak finding
        prominences = peak_prominences(trace, peaks)[0]  # Extract prominences
        peaks_after_2s = peaks[time_axis[peaks] > time_threshold]
        prominences_after_2s = prominences[time_axis[peaks] > time_threshold]
        all_prominences_after_2s.extend(prominences_after_2s)

# Fit GMM to all collected prominences
prominences_array = np.array(all_prominences_after_2s).reshape(-1, 1)
gmm = GaussianMixture(n_components=2, random_state=42).fit(prominences_array)

# Extract GMM parameters
noise_mean = np.min(gmm.means_)
noise_std = np.sqrt(np.min(gmm.covariances_))
big_mean = np.max(gmm.means_)
big_std = np.sqrt(np.max(gmm.covariances_))
prominence_threshold = noise_mean + 1 * noise_std  # Global threshold

print(f"Global Prominence Threshold (GMM): {prominence_threshold:.4f}")

# Visualize the fitted GMM on the prominence histogram
plt.figure(figsize=(10, 6))
counts, bins, _ = plt.hist(all_prominences_after_2s, bins=200, color='skyblue', edgecolor='black', alpha=0.7, density=True)
plt.xlabel('Prominence')
plt.ylabel('Density')
plt.title('Prominence Distribution Across All Datasets')

# Add GMM components to the histogram plot
x = np.linspace(bins[0], bins[-1], 1000).reshape(-1, 1)
gmm_density = np.exp(gmm.score_samples(x))
plt.plot(x, gmm_density, 'k--', label='GMM Combined Fit')
plt.plot(x, gmm.weights_[0] * (1 / (np.sqrt(2 * np.pi) * noise_std)) * 
         np.exp(-0.5 * ((x - noise_mean) / noise_std) ** 2), 
         'r-', label='Noise Cluster Fit')
plt.plot(x, gmm.weights_[1] * (1 / (np.sqrt(2 * np.pi) * big_std)) * 
         np.exp(-0.5 * ((x - big_mean) / big_std) ** 2), 
         'g-', label='Signal Cluster Fit')
plt.axvline(prominence_threshold, color='purple', linestyle='--', label='Global Threshold')
plt.legend()
plt.savefig("global_prominence_distribution.png")
plt.show()

# Store mean and std of spike rates for comparison later
spike_rate_means = []
spike_rate_stds = []

# Analyze each dataset using the global threshold
for dataset_name in datasets:
    trace_data = data[dataset_name]
    time_axis = data['rs_time_axis'].squeeze()

    # Plotting six traces in a 2x3 grid of subplots
    plt.figure(figsize=(15, 8))
    for i in range(6):  # Plot the first six traces
        trace = trace_data[:, i]
        peaks, _ = find_peaks(trace, prominence=prominence_threshold)
        peaks_after_2s = peaks[time_axis[peaks] > time_threshold]
        
        # Adding subplot for each trace
        plt.subplot(2, 3, i + 1)
        plt.plot(time_axis, trace, label=f'Trial {i+1}')
        plt.plot(time_axis[peaks_after_2s], trace[peaks_after_2s], 'ro', label='Spikes (Time > 2 s)', markersize=2)
        plt.xlabel('Time (ms)')
        plt.ylabel('Trace Intensity')
        plt.title(f'{dataset_name} - Trial {i+1}')
        plt.legend()
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_traces.png")
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
plt.bar(datasets, spike_rate_means, yerr=spike_rate_stds, capsize=5, color='lightgreen', edgecolor='black')
plt.ylabel('Mean Spike Rate (spikes per second)')
plt.xlabel('Dataset')
plt.title('Mean Spike Rate with Standard Deviation across Datasets')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("spike_rate_comparison.png")
plt.show()
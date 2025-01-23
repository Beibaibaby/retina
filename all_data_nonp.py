# Import necessary libraries
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
import numpy as np
from sklearn.neighbors import KernelDensity

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
        # Find peaks with increased sensitivity
        peaks, _ = find_peaks(
            trace, 
            height=None,          # Include all peaks regardless of amplitude
            threshold=0.0001,       # Allow very small vertical drop
            prominence=0.01       # Include low-prominence peaks
        )
        prominences = peak_prominences(trace, peaks)[0]  # Extract prominences
        peaks_after_2s = peaks[time_axis[peaks] > time_threshold]
        prominences_after_2s = prominences[time_axis[peaks] > time_threshold]
        all_prominences_after_2s.extend(prominences_after_2s)

# Fit KDE to all collected prominences
prominences_array = np.array(all_prominences_after_2s).reshape(-1, 1)
kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(prominences_array)

# Define range for KDE evaluation
x = np.linspace(min(all_prominences_after_2s), max(all_prominences_after_2s), 1000).reshape(-1, 1)
log_density = kde.score_samples(x)  # Log-density values
density = np.exp(log_density)

# Determine threshold as the 80th percentile of prominence
threshold = np.percentile(all_prominences_after_2s, 1)

# Visualize the KDE fit on prominence histogram
plt.figure(figsize=(10, 6))
plt.hist(all_prominences_after_2s, bins=100, color='skyblue', edgecolor='black', alpha=0.7, density=True)
plt.plot(x, density, 'k-', label='KDE Fit')
plt.axvline(threshold, color='purple', linestyle='--', label='80th Percentile Threshold')
plt.xlabel('Prominence')
plt.ylabel('Density')
plt.title('Prominence Distribution Across All Datasets (KDE Fit)')
plt.legend()
plt.savefig("global_prominence_distribution_kde.png")
plt.show()

print(f"Global Prominence Threshold (KDE 80th Percentile): {threshold:.4f}")

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
        peaks, _ = find_peaks(trace, prominence=threshold)
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
        peaks, _ = find_peaks(trace, prominence=threshold)
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
    

# Analyze each dataset using the global threshold
for dataset_name in datasets:
    trace_data = data[dataset_name]
    time_axis = data['rs_time_axis'].squeeze()

    # Plotting six traces in a 2x3 grid of subplots with prominence visualization
    plt.figure(figsize=(15, 8))
    for i in range(6):  # Plot the first six traces
        trace = trace_data[:, i]
        peaks, properties = find_peaks(trace, prominence=threshold)
        peaks_after_2s = peaks[time_axis[peaks] > time_threshold]
        prominences = peak_prominences(trace, peaks,wlen=300)[0]
        prominences_after_2s = prominences[time_axis[peaks] > time_threshold]

        # Adding subplot for each trace
        plt.subplot(2, 3, i + 1)
        plt.plot(time_axis, trace, label=f'Trial {i+1}')
        plt.plot(time_axis[peaks_after_2s], trace[peaks_after_2s], 'ro', label='Peaks', markersize=4)
        
        # Annotate prominences
        for peak, prom in zip(peaks_after_2s, prominences_after_2s):
            # Draw vertical lines to indicate prominence
            plt.vlines(x=time_axis[peak], ymin=trace[peak] - prom, ymax=trace[peak], color='blue', linestyle='--', linewidth=1)
            # Add prominence value as text
            plt.text(time_axis[peak], trace[peak], f'{prom:.2f}', fontsize=6, color='blue', ha='center')
        
        plt.xlabel('Time (ms)')
        plt.ylabel('Trace Intensity')
        plt.title(f'{dataset_name} - Trial {i+1}')
        plt.legend()
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_traces_kde.png")
    plt.close()
    
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
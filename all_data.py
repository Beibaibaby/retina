# Import necessary libraries
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np

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

# Store mean and std of spike rates for comparison later
spike_rate_means = []
spike_rate_stds = []

# Iterate over each dataset to analyze and plot results separately
for dataset_name in datasets:
    # Extract the data variable and the time axis
    trace_data = data[dataset_name]
    time_axis = data['rs_time_axis'].squeeze()
    
    # Plotting six traces in a 2x3 grid of subplots
    plt.figure(figsize=(15, 8))
    for i in range(6):  # Plot the first six traces
        trace = trace_data[:, i]
        peaks, _ = find_peaks(trace,prominence=0.07)
        
        # Adding subplot for each trace
        plt.subplot(2, 3, i + 1)
        plt.plot(time_axis, trace, label=f'Trial {i+1}')
        plt.plot(time_axis[peaks], trace[peaks], 'ro', label='Local Maxima', markersize=2)
        plt.xlabel('Time (ms)')
        plt.ylabel('Trace Intensity')
        plt.title(f'{dataset_name} - Trial {i+1}')
        plt.legend()
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_traces.png")
    #plt.show()

    # Initialize lists to collect stats
    all_isis = []  # For inter-spike intervals
    spike_rates = []  # For mean spike rate per trace

    # Loop through each trace and compute statistics
    for i in range(trace_data.shape[1]):
        trace = trace_data[:, i]
        peaks, _ = find_peaks(trace,prominence=0.07)
        
        # Compute ISIs in milliseconds
        isi = np.diff(time_axis[peaks])  # Difference between successive peak times
        all_isis.extend(isi)  # Append ISIs from each trace
        spike_rate = len(peaks) / (time_axis[-1] - time_axis[0]) * 1000  # Spikes per second
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
    #plt.show()

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
#plt.ylim(5000,7000)
plt.tight_layout()
plt.savefig("spike_rate_comparison.png")
plt.show()

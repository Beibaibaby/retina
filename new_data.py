from scipy.signal import find_peaks, peak_prominences
from scipy.stats import gaussian_kde, norm
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
time_threshold = 2.5  # Time threshold in seconds

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

# Convert to numpy array
left_prominences_array = np.array(all_left_prominences_for_threshold)

# Perform KDE on the left prominences
kde = gaussian_kde(left_prominences_array)
x_grid = np.linspace(left_prominences_array.min(), left_prominences_array.max(), 1000)
kde_pdf = kde.evaluate(x_grid)

# Find local maxima in the KDE estimate
from scipy.signal import find_peaks

peaks_indices, _ = find_peaks(kde_pdf)
peak_prominences_kde = compute_left_prominences(kde_pdf, peaks_indices)

# Assume the first peak corresponds to the noise distribution
if len(peaks_indices) > 0:
    noise_peak_index = peaks_indices[0]
    noise_peak_value = x_grid[noise_peak_index]

    # Extract data around the noise peak
    bandwidth = 0.05  # Adjust as needed
    noise_data_indices = (x_grid >= (noise_peak_value - bandwidth)) & (x_grid <= (noise_peak_value + bandwidth))
    noise_data_x = x_grid[noise_data_indices]
    noise_data_pdf = kde_pdf[noise_data_indices]

    # Fit a Gaussian to the noise peak
    # Since the KDE is a density estimate, we need to reconstruct data points
    # We can approximate the noise data by sampling from the noise peak region
    noise_sample_size = 1000  # Number of samples to generate
    noise_cdf = np.cumsum(noise_data_pdf)
    noise_cdf = noise_cdf / noise_cdf[-1]  # Normalize to make it a proper CDF
    random_values = np.random.rand(noise_sample_size)
    noise_samples = np.interp(random_values, noise_cdf, noise_data_x)

    # Fit Gaussian to the noise samples
    noise_mean, noise_std = norm.fit(noise_samples)

    # Set the threshold
    k = 4  # Number of standard deviations above the mean; adjust as needed
    prominence_threshold = noise_mean + k * noise_std
else:
    # Fallback if no peaks are found
    prominence_threshold = np.percentile(left_prominences_array, 90)
    noise_mean = np.mean(left_prominences_array)
    noise_std = np.std(left_prominences_array)

# Visualize the histogram and KDE with the threshold
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(left_prominences_array, bins=100, color='skyblue', edgecolor='black', alpha=0.7, density=True, label='Histogram')
plt.plot(x_grid, kde_pdf, 'r-', label='KDE')
plt.axvline(prominence_threshold, color='purple', linestyle='--', label='Prominence Threshold')
plt.axvline(noise_mean, color='green', linestyle='--', label='Noise Mean')
plt.xlabel('Left Prominence')
plt.ylabel('Density')
plt.title('Left Prominence Distribution with KDE and Noise Gaussian Fit')
plt.legend()
plt.savefig("threshold_left_prominence_distribution_kde_gaussian.png")
plt.show()

print(f"Noise Mean: {noise_mean:.4f}, Noise Std: {noise_std:.4f}")
print(f"Prominence Threshold (Noise Mean + {k} * Noise Std): {prominence_threshold:.4f}")

# Store mean and std of spike rates for comparison later
spike_rate_means = []
spike_rate_stds = []

# Analyze each dataset using the new threshold
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
        
        plt.xlabel('Time (s)')
        plt.ylabel('Trace Intensity')
        plt.title(f'{dataset_name} - Trial {i+1} (Left Prominence)')
        plt.legend()
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_traces_new.png")
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
        spike_rate = len(peaks_after_2s) / (time_axis[-1] - time_axis[0]) # # Spikes per second
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
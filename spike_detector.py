import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
import numpy as np
from sklearn.mixture import GaussianMixture

# Load the dataset
file_path = 'ca_imaging_traces.mat'  # Update this path to where your file is located
data = sio.loadmat(file_path)

# Extract the data variable and the time axis
trace_data = data['plot_all_rs_norm_pref1_traces_final']
time_axis = data['rs_time_axis'].squeeze()

# Initialize a list to collect prominences after 2 seconds
all_prominences_after_2s = []

# Time threshold in seconds
time_threshold = 2.0

# Compute prominences for all traces and collect those after 2 seconds
for i in range(trace_data.shape[1]):
    trace = trace_data[:, i]
    peaks, _ = find_peaks(trace)  # Initial peak finding
    prominences = peak_prominences(trace, peaks)[0]  # Extract prominences
    
    # Filter peaks based on time > 2s
    peaks_after_2s = peaks[time_axis[peaks] > time_threshold]
    prominences_after_2s = prominences[time_axis[peaks] > time_threshold]
    
    # Collect prominences for analysis
    all_prominences_after_2s.extend(prominences_after_2s)

# Plot histogram of prominences after 2 seconds
plt.figure(figsize=(10, 6))
counts, bins, _ = plt.hist(all_prominences_after_2s, bins=50, color='skyblue', edgecolor='black', alpha=0.7, density=True)
plt.xlabel('Prominence')
plt.ylabel('Density')
plt.title('Prominence Distribution (Time > 2 s)')

# Fit GMM to prominences
use_gmm = True  # Set to False to use mean + 1 * std for thresholding
if use_gmm:
    prominences = np.array(all_prominences_after_2s).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42).fit(prominences)
    noise_mean = np.min(gmm.means_)
    noise_std = np.sqrt(np.min(gmm.covariances_))
    big_mean = np.max(gmm.means_)
    big_std = np.sqrt(np.max(gmm.covariances_))
    threshold = np.min(gmm.means_) + 1 * np.sqrt(np.min(gmm.covariances_))  # High-prominence cluster
    prominence_threshold = threshold
    print(f"Big Mean Prominence (GMM): {big_mean:.4f}")
    print(f"Big Standard Deviation of Prominence (GMM): {big_std:.4f}")
    print(f"Prominence Threshold (GMM): {prominence_threshold:.4f}")
    print(f"Noise Mean: {noise_mean:.4f}")
    print(f"Noise Standard Deviation: {noise_std:.4f}")

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
    plt.axvline(prominence_threshold, color='purple', linestyle='--', label='Threshold')
    plt.legend()

else:
    # Set the threshold as mean + 2 * std (or 1 * std, based on your preference)
    mean_prominence = np.mean(all_prominences_after_2s)
    std_prominence = np.std(all_prominences_after_2s)
    prominence_threshold = mean_prominence + 1 * std_prominence
    print(f"Mean Prominence: {mean_prominence:.4f}")
    print(f"Standard Deviation of Prominence: {std_prominence:.4f}")
    print(f"Prominence Threshold: {prominence_threshold:.4f}")

plt.show()

# Updated peak detection
plt.figure(figsize=(15, 8))
for i in range(6):  # Plotting the first six traces
    trace = trace_data[:, i]
    peaks, _ = find_peaks(trace, prominence=prominence_threshold)
    
    # Filter peaks to include only those after 2 seconds
    peaks_after_2s = peaks[time_axis[peaks] > time_threshold]
    
    # Adding subplot for each trace
    plt.subplot(2, 3, i + 1)
    plt.plot(time_axis, trace, label=f'Trial {i+1}')
    plt.plot(time_axis[peaks_after_2s], trace[peaks_after_2s], 'ro', label='Spikes (Time > 2 s)', markersize=2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Trace Intensity')
    plt.title(f'Trial {i+1}')
    plt.legend()

# Adjust layout for readability and show plot
plt.tight_layout()
plt.show()
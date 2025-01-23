# Import necessary libraries
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
import numpy as np
from scipy.optimize import minimize
from scipy.stats import expon, norm

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

# Prepare prominence array
prominences_array = np.array(all_prominences_after_2s)

# Define the Exponential-Gaussian Mixture Log-Likelihood function
def egmm_log_likelihood(params, data):
    w, lambda_exp, mu_gauss, sigma_gauss = params
    # Ensure constraints (weights between 0 and 1, sigma > 0)
    if not (0 <= w <= 1 and sigma_gauss > 0 and lambda_exp > 0):
        return np.inf  # Invalid parameter set

    # Exponential and Gaussian components
    exp_component = w * expon.pdf(data, scale=1/lambda_exp)
    gauss_component = (1 - w) * norm.pdf(data, loc=mu_gauss, scale=sigma_gauss)
    
    # Total mixture log likelihood
    mixture = exp_component + gauss_component
    return -np.sum(np.log(mixture + 1e-9))  # Add small value to avoid log(0)

# Initial guess for parameters
w_init = 0.5  # Weight for the exponential component
lambda_exp_init = 1 / np.mean(prominences_array)  # Exponential rate (1 / mean)
mu_gauss_init = np.mean(prominences_array)  # Gaussian mean
sigma_gauss_init = np.std(prominences_array)  # Gaussian standard deviation
initial_params = [w_init, lambda_exp_init, mu_gauss_init, sigma_gauss_init]

# Optimize the parameters
result = minimize(
    egmm_log_likelihood,
    initial_params,
    args=(prominences_array,),
    bounds=[(0, 1), (1e-6, None), (0, None), (1e-6, None)],  # Constraints
    method='L-BFGS-B'
)

# Extract optimized parameters
w_opt, lambda_exp_opt, mu_gauss_opt, sigma_gauss_opt = result.x
print(f"Optimized Parameters:")
print(f"  Exponential Weight (w): {w_opt:.4f}")
print(f"  Exponential Rate (lambda): {lambda_exp_opt:.4f}")
print(f"  Gaussian Mean (mu): {mu_gauss_opt:.4f}")
print(f"  Gaussian Std Dev (sigma): {sigma_gauss_opt:.4f}")

# Determine global threshold as mean + k * std of the Gaussian component
k = 2  # You can adjust this multiplier
threshold = mu_gauss_opt + k * sigma_gauss_opt
print(f"Global Prominence Threshold (Exponential-Gaussian Mixture): {threshold:.4f}")

# Visualize the prominence distribution and the EGMM fit
x = np.linspace(0, max(prominences_array), 1000)
exp_pdf = w_opt * expon.pdf(x, scale=1/lambda_exp_opt)
gauss_pdf = (1 - w_opt) * norm.pdf(x, loc=mu_gauss_opt, scale=sigma_gauss_opt)
mixture_pdf = exp_pdf + gauss_pdf

plt.figure(figsize=(10, 6))
plt.hist(prominences_array, bins=100, color='skyblue', edgecolor='black', alpha=0.7, density=True, label='Prominence Histogram')
plt.plot(x, exp_pdf, 'r-', label='Exponential Component')
plt.plot(x, gauss_pdf, 'g-', label='Gaussian Component')
plt.plot(x, mixture_pdf, 'k--', label='Mixture Model')
plt.axvline(threshold, color='purple', linestyle='--', label='Global Threshold')
plt.xlabel('Prominence')
plt.ylabel('Density')
plt.title('Prominence Distribution with Exponential-Gaussian Mixture Fit')
plt.legend()
plt.savefig("global_prominence_distribution_egmm.png")
plt.show()
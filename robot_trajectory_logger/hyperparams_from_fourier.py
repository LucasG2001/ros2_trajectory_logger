import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from filters import compute_length_scale_from_fft

"""
This script demonstrates the application of Gaussian Process Regression (GPR) for 
modeling noisy sinusoidal signals. It uses GPR to estimate and predict signals based on 
noisy observations while leveraging different hyperparameter tuning approaches, such as optimizing the 
length scale and fixing it. The code is structured in a way that allows for testing different 
frequency components of a signal and comparing the effects of varying the length scale on the predictions.
"""

def generate_noisy_sinusoid(x, freqs, noise_level=0.1):
    """Generates a noisy sinusoidal signal from a list of frequencies."""
    y_clean = np.sum([np.sin(2 * np.pi * f * x) for f in freqs], axis=0)
    y_noisy = y_clean + np.random.normal(0, noise_level, size=x.shape)
    return y_clean, y_noisy

def fit_gpr(x_train, y_train, length_scale=None, fixed=False):
    """Fits a Gaussian Process model with an optional fixed length scale."""
    l_bounds = (1e-2, 1e3) if not fixed else "fixed"
    kernel = C(1.0) * RBF(length_scale=length_scale if length_scale else 1.0, length_scale_bounds=l_bounds)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-1, normalize_y=True)
    gpr.fit(x_train[:, np.newaxis], y_train)
    return gpr

# Define sampling frequency
sampling_freq = 1000  # Hz (Increased for higher frequency resolution)
signal_duration = 1.0  # seconds
noise_level = 0.2
num_samples = int(sampling_freq * signal_duration)
cutoff_frequency = 50  # Hz, cutoff frequency for the fourier-spectrum based estimation process of gaussian hyperparameters

# Generate training data
x = np.linspace(0, signal_duration, num_samples, endpoint=False)
signal_cases = [[5], [1, 3], [1, 3, 5, 20], [5, 10, 20, 50], [10, 20, 50, 100]]  # More complex cases
signals = [generate_noisy_sinusoid(x, freqs, noise_level) for freqs in signal_cases]

# Fit GPR models and plot results
fig, axes = plt.subplots(len(signal_cases), 2, figsize=(12, 12))
x_pred = np.linspace(0, signal_duration, num_samples * 5)

for i, (freqs, (y_clean, y_noisy)) in enumerate(zip(signal_cases, signals)):
    x_train, y_train = x, y_noisy
    
    # Fit with standard hyperparameter tuning
    gpr_default = fit_gpr(x_train, y_train, length_scale=None, fixed=False)
    y_pred_default, sigma_default = gpr_default.predict(x_pred[:, np.newaxis], return_std=True)
    length_scale_default = gpr_default.kernel_.k2.length_scale
    
    # Fit with precomputed length scale from FFT
    length_scale = compute_length_scale_from_fft(y_train, sampling_freq, sc=cutoff_frequency, plot=False) # cut off at sc
    gpr_fixed = fit_gpr(x_train, y_train, length_scale=length_scale, fixed=True)
    y_pred_fixed, sigma_fixed = gpr_fixed.predict(x_pred[:, np.newaxis], return_std=True)
    
    # Print length scales
    print(f"Signal frequencies: {freqs}")
    print(f"Optimized length scale: {length_scale_default:.4f}")
    print(f"Fixed length scale: {length_scale:.4f}\n")

    # Plot standard hyperparameter tuning
    ax = axes[i, 0]
    ax.plot(x, y_noisy, 'k.', label='Noisy observations')
    ax.plot(x, y_clean, 'g-', label='True signal')
    ax.plot(x_pred, y_pred_default, 'b-', label=f'GPR (optimized, LS={length_scale_default:.4f})')
    ax.fill_between(x_pred, y_pred_default - sigma_default, y_pred_default + sigma_default, alpha=0.2, color='blue')
    ax.set_title(f'GPR with Optimized Hyperparams (Freqs: {freqs})')
    
    # Plot fixed length scale from FFT
    ax = axes[i, 1]
    ax.plot(x, y_noisy, 'k.', label='Noisy observations')
    ax.plot(x, y_clean, 'g-', label='True signal')
    ax.plot(x_pred, y_pred_fixed, 'r-', label=f'GPR (fixed LS={length_scale:.4f})')
    ax.fill_between(x_pred, y_pred_fixed - sigma_fixed, y_pred_fixed + sigma_fixed, alpha=0.2, color='red')
    ax.set_title(f'GPR with Fixed Length Scale (Freqs: {freqs})')
    
for ax in axes.flat:
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')

plt.tight_layout()
plt.show()

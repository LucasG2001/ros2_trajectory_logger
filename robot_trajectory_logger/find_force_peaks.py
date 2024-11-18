import json
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import find_peaks
from plot import load_log_file, extract_data, apply_ema
import pandas as pd

# TODO 
# analyze the impact of cutting out the non-drilling part (change in signal mean)

def compute_scale(max_frequency, sampling_frequency):
    dt = 1/sampling_frequency
    lower = 1
    upper = 0.75/dt * (1/max_frequency)
    print("scales are ", lower, " ", upper)
    return lower, upper

def wavelet_analysis(timestamps, quantity, max_frequency, sampling_frequency):
    # Perform continuous wavelet transform (CWT) with the Mexican hat wavelet
    wavelet = 'mexh'  # Mexican hat wavelet is good for detecting spikes
    lower, upper = compute_scale(max_frequency, sampling_frequency)
    scales = np.arange(lower, upper)  # Adjust scale range based on data characteristics
    coefficients, frequencies = pywt.cwt(quantity, scales, wavelet)
    # Calculate the wavelet power spectrum (magnitude squared of coefficients)
    power = np.abs(coefficients) ** 2

    # Detect spikes by looking for high-energy events at fine scales (higher frequency)
    # Here we focus on the first few scales, where sharp transitions are prominent
    fine_scale_power = np.mean(power[:20], axis=0)  # Average power over fine scales
    spike_indices, _ = find_peaks(fine_scale_power, height=np.mean(fine_scale_power) * 3)

    # Plot the original force data
    plt.figure(figsize=(14, 8))
    plt.subplot(3, 1, 1)
    plt.plot(timestamps, quantity, label="Force")
    plt.plot(timestamps[spike_indices], quantity[spike_indices], "ro", label="Detected Spikes")
    plt.title("Original Force Data with Detected Spikes")
    plt.xlabel("Time")
    plt.ylabel("Force")
    plt.legend()

    # Plot the wavelet power spectrum as a heatmap
    plt.subplot(3, 1, 2)
    plt.imshow(
        power, extent=[timestamps.min(), timestamps.max(), scales.max(), scales.min()],
        aspect="auto", cmap="jet"
    )
    plt.colorbar(label="Power")
    plt.title("Wavelet Power Spectrum (Time vs. Scale)")
    plt.xlabel("Time")
    plt.ylabel("Scale")

    # Plot the average power at fine scales to illustrate spike detection
    plt.subplot(3, 1, 3)
    plt.plot(timestamps, fine_scale_power, label="Fine Scale Power")
    plt.plot(timestamps[spike_indices], fine_scale_power[spike_indices], "ro", label="Detected Spikes")
    plt.axhline(np.mean(fine_scale_power) * 3, color="red", linestyle="--", label="Detection Threshold")
    plt.title("Fine Scale Power for Spike Detection")
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # Load data from JSON file
    logfile = 'robot_state_log_2024_11_14_1317_trigger_6000.json'
    # Load and process the log file
    data = load_log_file(logfile)
    timestamps, forces, torques, reference_positions, euler_angles, ee_positions, ee_orientations = extract_data(data)
    dF_ext = np.array([item['dt_Fext_z'] for item in data if 'dt_Fext_z' in item])
    # forces = apply_ema(forces, alpha=0.1)
    # Convert lists to numpy arrays for further processing
    forces = np.array(forces['z'])
    timestamps = np.array(timestamps)/500
    # print(np.abs(dF_ext - forces))
    # cutoff
    cutoff_freq = 10 # Hz
    f_sampling = 500 # Hz

    wavelet_analysis(timestamps, dF_ext, cutoff_freq, f_sampling)
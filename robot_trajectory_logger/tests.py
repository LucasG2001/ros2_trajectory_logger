from robot_trajectory_logger.RTFilters import RealTimeBandpassFilter
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

"""
Example and TEst functions for filters in the robot_trajectory_logger package.
"""
def bandpass_filter_example():
    fs = 500  # Sampling frequency (Hz)
    lowcut = 5  # Low cutoff (Hz)
    highcut = 15  # High cutoff (Hz)

    filter_obj = RealTimeBandpassFilter(lowcut, highcut, fs)

    # Simulated real-time signal: 10 Hz sine wave + noise
    t = np.linspace(0, 1, fs)  # 1-second time vector
    raw_signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(len(t))

    filtered_signal = []
    for sample in raw_signal:
        y = filter_obj.filter_sample(sample)
        filtered_signal.append(y)

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(t, raw_signal, label="Raw Signal", alpha=0.5)
    plt.plot(t, filtered_signal, label="Filtered Signal", linewidth=2)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Real-Time Butterworth Bandpass Filter (5-15 Hz)")
    plt.show()

def stft_spectogram_example():

    # Generate synthetic signal (spike in a 10 Hz sinusoid)
    fs = 500  # Sampling frequency (Hz)
    t = np.linspace(0, 1, fs, endpoint=False)
    signal_clean = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
    spike = np.zeros_like(t)
    spike[250] = 10  # A sharp spike at the midpoint
    signal_with_spike = signal_clean + spike

    # Compute Short-Time Fourier Transform (STFT)
    f, t_stft, Zxx = signal.stft(signal_with_spike, fs, nperseg=100)

    # Plot STFT Spectrogram
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t_stft, f, np.abs(Zxx), shading='gouraud')
    plt.colorbar(label="Magnitude")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.title("STFT Spectrogram of Signal with Spike")
    plt.show()


import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from filters import ema_filter, moving_average_filter, normalize_array, compute_and_plot_stft, real_time_outlier_detection, plot_real_time_outliers, plot_spectral_intensity
from RTBandPassFilter import RealTimeBandpassFilter
from linear_regression import extract_data
from scipy.signal import spectrogram, stft
from spike_detection import z_score_spike_detection

"""
implements spike detector class to save and operate on dtill data
"""

class SpikeDetector:
    def __init__(self, logfile, fs=500, time_window=8, passband=(3, 8)):
        T = 1/fs # sampling period
        self.fs = fs # sampling frequency
        self.passband = passband # filtering passband
        self.data_file = logfile
        self.timestamps, forces, _ , _ , ee_positions, orientations, dt_Fext_z = extract_data(logfile, fs, time_window) # dt_-fext is already filtered
        self.timestamps = np.array(self.timestamps)
        self.orientations = np.array(orientations)
        # get drilling forces
        self.drilling_force = np.array(forces['z'])
        self.dt_Fext_z = np.array(dt_Fext_z)
        self.dt_Fext_z_raw = np.concatenate(([0], np.diff(self.dt_Fext_z) / T))
        self.dt_Fext_z_filtered = self.simulate_rt_bandpass_filter(self.dt_Fext_z_raw, passband) # Bandpass filter the derivative of the force
        # get displacements
        self.displacement = np.array(ee_positions['z'])
        self.displacement = self.displacement - self.displacement[0]  # Set initial displacement to zero
        # get velocities
        position_differences = np.diff(self.displacement)  # Differences between consecutive positions
        self.velocities = np.concatenate(([0], position_differences / T)) * 100 # Norms divided by sampling time 


    def simulate_rt_bandpass_filter(self, signal_data, passband):
        # Apply real-time bandpass filter to the force data
        low, high = passband
        self.bandpass_filter = RealTimeBandpassFilter(low, high, self.fs)
        filtered_signal = []
        for sample in signal_data:
            y = self.bandpass_filter.filter_sample(sample)
            filtered_signal.append(y)

        return np.array(filtered_signal)

    def plot_K_v_forces(self):
          K = normalize_array((self.drilling_force / (self.velocities + 0.0001)))
          plt.figure()
          plt.plot(normalize_array(moving_average_filter(self.velocities)))
          plt.plot(moving_average_filter(K, window_size=10))
          plt.plot(normalize_array(self.dt_Fext_z))
          plt.show()

    def compute_and_plot_mad(self):
        # Run real-time outlier detection
        medians = []
        mads = []
        outlier_indices = []
        threshold = 3.5
        for index, (point, is_outlier, median, mad) in enumerate(real_time_outlier_detection(self.spectral_intensity, 10, threshold)):
            medians.append(median)
            mads.append(mad)
            if is_outlier:
                outlier_indices.append(index)

        medians = np.array(medians)
        mads = np.array(mads)

        plt.plot(self.spectral_intensity, color = "blue")
        plt.fill_between(np.arange(len(medians)), medians - threshold * mads, medians + threshold * mads, color="orange", alpha=0.3, label="3 x MAD Confidence Interval")
        plt.scatter(outlier_indices, self.spectral_intensity[outlier_indices], color="red", label="Outliers", s=10, zorder=3)

    def plot_metrics(self):
        # add simulated bandpass filter data
       
        # Plot the displacement (Z-axis)
        # Plot linear regression results alongside position or force data
        fig, axs = plt.subplots(6, 1, figsize=(18, 18), sharex=True)

        axs[0].plot(self.timestamps, self.displacement, label="Displacement (Z-axis)", color='blue')
        axs[0].set_xlabel("Timestamps")
        axs[0].set_ylabel("Displacement (Z)")
        axs[0].legend()
        axs[0].grid(True)

        # Plot the filtered force (Z-axis)
        axs[1].plot(self.timestamps, self.drilling_force, label="Force (Z-axis)", color='green')
        axs[1].set_xlabel("Timestamps")
        axs[1].set_ylabel("Force (Z)")
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(self.timestamps, self.velocities, label="Velocities_z", color='dodgerblue')
        axs[2].set_xlabel("Timestamps")
        axs[2].set_ylabel("Derivative_F (Z)")
        axs[2].legend()
        axs[2].grid(True)
        # placeholder on no. 3
        axs[3].plot(self.timestamps, self.simulate_rt_bandpass_filter(self.dt_Fext_z_raw, self.passband), label="Bandpass filtered F", color='purple')
        axs[3].set_xlabel("Timestamps")
        axs[3].set_ylabel("Filtered DF (Z)")
        axs[3].legend()
        axs[3].grid(True)

        axs[4].plot(self.timestamps, self.dt_Fext_z_raw, label="F_ext_dt (Z-axis)", color='black')
        axs[4].set_xlabel("Timestamps")
        axs[4].set_ylabel("F_ext_dt")
        axs[4].legend()
        axs[4].grid(True)

        # Plot spectral intensity of signal additionally
        nperseg = 256  # Window size
        noverlap = nperseg - 1  # Maximum overlap for per-sample estimates
        f_min, f_max = (0, self.fs/2)  # Define passband
        # Zero-pad signal to ensure full coverage
        padded_signal = np.concatenate((self.dt_Fext_z_raw, np.ones(nperseg - 1) * self.dt_Fext_z_raw[0]))
        # Compute STFT
        f, t, Zxx = stft(padded_signal, fs=self.fs, nperseg=nperseg, noverlap=noverlap, window='hann')
        # Compute spectral density (power) |Zxx|^2
        Sxx = np.abs(Zxx) ** 2
        # Select only frequencies in the passband
        mask = (f >= f_min) & (f <= f_max)
        spectral_density = np.sum(Sxx[mask, :], axis=0)  # Sum power over selected frequencies
        # Map STFT time bins back to original signal time indices
        original_time = np.linspace(0, len(self.dt_Fext_z_raw) / self.fs, len(self.dt_Fext_z_raw))
        # Interpolate to get per-sample spectral density estimates
        spectral_density_interp = np.interp(original_time, t, spectral_density)
        self.spectral_intensity = spectral_density_interp
        # Plot the total spectral intensity over time
        axs[5].plot(np.arange(len(self.dt_Fext_z_raw)), np.abs(spectral_density_interp), label=f'Total Spectral Intensity ({f_min}-{f_max} Hz)', color='b')
        axs[5].set_xlabel('Time [s]')
        axs[5].set_ylabel('Spectral Intensity')
        axs[5].legend()

        plt.tight_layout()
        plt.show()
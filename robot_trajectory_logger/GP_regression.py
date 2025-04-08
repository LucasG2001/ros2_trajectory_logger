import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from RTFilters import RealTimeBandpassFilter
from filters import moving_average_filter, compute_length_scale_from_fft, ema_filter, real_time_autocorrelation, plot_frequency_bands_over_time
from SpikeDetector import SpikeDetector

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from time import time
import GPy
from gaussian_on_slices import fit_gaussian_gp, autocorrellation_cpd
from scipy.stats import ttest_ind


def perform_gp_self_correlation(displacement, velocities, f_magnitude, passband=(10, 50), sampling_frequency=1000):
    """
    Perform Gaussian Process Regression with self-correlation on the force magnitude data using GPy library. 
    Simulates a Datastream of the Force and uses the previous values to predict the next value(s).
    """
    # Parameters
    feature_size = 10  # feature size (number of previous states to use for prediction of next state)
    padding_dim = feature_size + 1 # zero pasdding when looking back in time
    refit_interval = 200  # Refit every "refit_interval" steps
    no_samples_per_interval = 10
    subsampling_factor = refit_interval//no_samples_per_interval # subsampling factor for training data
    y_data = np.hstack([np.zeros(padding_dim), f_magnitude])  # prepend zeros
    n_points = len(y_data)
    displacement = np.hstack([np.zeros(padding_dim), displacement])  # prepend zeros
    velocities = np.hstack([np.zeros(padding_dim), velocities])  # prepend zeros
    has_triggered = False # keep track of change points
    
    X_train = np.zeros([refit_interval, feature_size])  # n_samples x n_features
    y_train = np.zeros([refit_interval, 1])  # n_samples x n_targets

    print("Shape of y_data:", y_data.shape)
    
    # Storage for predictions
    means = np.zeros(n_points)
    sigmas = np.ones(n_points)
    anomalies = []
    times = []

    # Define GP kernel (Sum of RBF and a Constant term)
    kernel = GPy.kern.RBF(input_dim=feature_size,variance=1.0, lengthscale=np.ones(feature_size) * 1, ARD=True) #+ GPy.kern.Bias(input_dim=window_size, variance=0.1)
    # Set constraints for the kernel hyperparameters
    kernel.lengthscale.constrain_bounded(0.1 * np.ones(1), 5.0 * np.ones(1))  # Lengthscale between 0.1 and 5
    kernel.variance.constrain_bounded(0.5, 100.0)    # Variance between 0.5 and 10
    # Initialize GP model
    gp = GPy.models.GPRegression(np.zeros((1, feature_size)), np.zeros((1, feature_size)), kernel, noise_var=0.1)
    # Constrain noise variance
    gp.Gaussian_noise.variance.constrain_bounded(0.1, 0.5)  # Noise variance between 1e-4 and 0.5
    print("Initial hyperparameters:\n", gp)

    # Streaming GP Processing
    for i in range(feature_size + 1, n_points):
        start_time = time()
        
        # Feature vector: last `feature size` points
        feature_vector = y_data[i - feature_size - 1 : i - 1].reshape(1, -1)
        
        # Predict next sequence
        y_pred, y_std = gp.predict(feature_vector) # predict disturbance
        means[i], sigmas[i] = y_pred.flatten()[-1], np.sqrt(y_std.flatten()[-1])  # Extract only the last prediction
        # check for anomaly (change point)
        lower_bound, upper_bound = means[i] - 1.96 * sigmas[i], means[i] + 1.96 * sigmas[i]
        if has_triggered == True and velocities[i] > 0.0:
            has_triggered = False
        if y_data[i] < lower_bound or y_data[i] > upper_bound:
            # avoid false positives by only counting anomalies when inside the bone
            # also avoid detecting anomalies when the process has not been fitted yet
            if displacement[i] < 0.001 and i > 400 and y_data[i] < 0.0: 
                if has_triggered == False:
                    anomalies.append(i)
                    has_triggered = True

        # Store training data in buffer, refill for every interval
        if has_triggered == False:
            X_train[i % refit_interval, :] = feature_vector
            y_train[i % refit_interval, :] = y_data[i].reshape(1, -1)

        # Refit GP every `refit_interval` steps
        if i % refit_interval == 0 and has_triggered == False:
            # Update kernel hyperparameters
            # Set new data and refit (same hyperparameters) but only on a subset of data
            gp.set_XY(X_train[::subsampling_factor], y_train[::subsampling_factor])
            if i < 1000:
                gp.optimize()
            print("Updated hyperparameters:\n", gp)

            end_time = time()
            times.append(end_time - start_time)
    
 
    #benchmark
    print("Average time per fitting iteration:", np.mean(times))

    # Plot results
    fig, axs = plt.subplots(3, 1, figsize=(12, 16))
    axs[0].plot(y_data, label="Original Data", color="blue", alpha=0.6, linewidth=1.0)
    axs[0].plot(means, label="GP Mean", color="red", linewidth=2.0)
    axs[0].fill_between(np.arange(len(means)), means - 1.96 * sigmas, means + 1.96 * sigmas, 
                     color="orange", alpha=0.3, label="95% Confidence Interval")
    axs[0].scatter(anomalies, y_data[anomalies], color="black", label="Anomalies", zorder=3, s=3)
    axs[0].set_xlabel("Time Steps")
    axs[0].set_ylabel("Force Magnitude")
     # veolcity plot
    axs[1].plot(velocities, label="Velocity", color="purple", linewidth=1.5)
    axs[1].set_xlabel("Time Steps")
    axs[1].set_ylabel("Velocity")
    axs[1].set_title("Velocity Over Time")
    axs[1].legend()
    # Displacement plot
    axs[2].plot(displacement, label="Displacement", color="purple", linewidth=1.5)
    axs[2].set_xlabel("Time Steps")
    axs[2].set_ylabel("Displacement")
    axs[2].set_title("Displacement Over Time")
    axs[2].legend()

    # Add vertical dashed lines in all subplots
    for i in range(1, len(axs)):
        ax = axs[i]
        for pos in anomalies:
            ax.axvline(pos, linestyle="dashed", color="black", alpha=0.7)

    plt.title("GP Regression with Self-Correlation")
    fig.legend()
    plt.show()


    return means, sigmas

def perform_gp_velocity(displacement, velocities, f_magnitude, passband=(10, 50), sampling_frequency=1000):
    """
    Perform Gaussian Process Regression with self-correlation on the force magnitude data using GPy library. 
    Simulates a Datastream of the Force and uses the previous values to predict the next value(s).
    """
    # Parameters
    feature_size = 10  # feature size (number of previous states to use for prediction of next state)
    padding_dim = feature_size + 1 # zero pasdding when looking back in time
    refit_interval = 200  # Refit every "refit_interval" steps
    no_samples_per_interval = 20
    subsampling_factor = refit_interval//no_samples_per_interval # subsampling factor for training data
    y_data = np.hstack([np.zeros(padding_dim), velocities])  # prepend zeros
    n_points = len(y_data)
    displacement = np.hstack([np.zeros(padding_dim), displacement])  # prepend zeros
    forces = np.hstack([np.zeros(padding_dim), f_magnitude])  # prepend zeros
    has_triggered = False # keep track of change points
    
    X_train = np.zeros([refit_interval, feature_size])  # n_samples x n_features
    y_train = np.zeros([refit_interval, 1])  # n_samples x n_targets

    print("Shape of y_data:", y_data.shape)
    
    # Storage for predictions
    means = np.zeros(n_points)
    sigmas = np.ones(n_points)
    anomalies = []
    times = []

    # Define GP kernel (Sum of RBF and a Constant term)
    kernel = GPy.kern.RBF(input_dim=feature_size,variance=1.0, lengthscale=np.ones(feature_size) * 1, ARD=True) #+ GPy.kern.Bias(input_dim=window_size, variance=0.1)
    # Set constraints for the kernel hyperparameters
    # kernel.lengthscale.constrain_bounded(1e-1 * np.ones(1), 1 * np.ones(1))  # Lengthscale between 0.1 and 5
    kernel.lengthscale.fix()
    kernel.variance.constrain_bounded(1e-2, 0.2)    # Variance between 0.5 and 10
    # Initialize GP model
    gp = GPy.models.GPRegression(np.zeros((1, feature_size)), np.zeros((1, feature_size)), kernel, noise_var=1e-3)
    # Constrain noise variance
    #gp.Gaussian_noise.variance.constrain_bounded(6*1e-4, 1e-3)  # Noise variance between 1e-4 and 0.5
    gp.Gaussian_noise.variance.fix()  # Noise variance between 1e-4 and 0.5
    print("Initial hyperparameters:\n", gp)

    # Streaming GP Processing
    for i in range(feature_size + 1, n_points):
        start_time = time()
        
        # Feature vector: last `feature size` points
        feature_vector = y_data[i - feature_size - 1 : i - 1].reshape(1, -1)
        
        # Predict next sequence
        y_pred, y_std = gp.predict(feature_vector) # predict disturbance
        means[i], sigmas[i] = y_pred.flatten()[-1], np.sqrt(y_std.flatten()[-1])  # Extract only the last prediction
        # check for anomaly (change point)
        lower_bound, upper_bound = means[i] - 1.99 * sigmas[i], means[i] + 1.99 * sigmas[i]
        if has_triggered == True and y_data[i] > 0.0:
            has_triggered = False
        if y_data[i] < lower_bound or y_data[i] > upper_bound:
            # avoid false positives by only counting anomalies when inside the bone
            # also avoid detecting anomalies when the process has not been fitted yet
            if displacement[i] < -0.002 and i > 400 and forces[i] < 0.0 and velocities[i] < 0.0: 
                if has_triggered == False:
                    anomalies.append(i)
                    has_triggered = True

        # Store training data in buffer, refill for every interval
        if displacement[i] < -0.001: # has_triggered == False:
            X_train[i % refit_interval, :] = feature_vector
            y_train[i % refit_interval, :] = y_data[i].reshape(1, -1)

        # Refit GP every `refit_interval` steps
        if i % refit_interval == 0 and has_triggered == False:
            # Update kernel hyperparameters
            # Set new data and refit (same hyperparameters) but only on a subset of data
            gp.set_XY(X_train[::subsampling_factor], y_train[::subsampling_factor])
            gp.optimize()
            print("Updated hyperparameters:\n", gp)

            end_time = time()
            times.append(end_time - start_time)
    
 
    #benchmark
    print("Average time per fitting iteration:", np.mean(times))

    # Plot results
    fig, axs = plt.subplots(3, 1, figsize=(12, 16))
    axs[0].plot(y_data, label="Original Velocity  Data", color="dodgerblue", alpha=0.6, linewidth=1.0)
    axs[0].plot(means, label="GP Mean", color="red", linewidth=2.0)
    axs[0].fill_between(np.arange(len(means)), means - 1.96 * sigmas, means + 1.96 * sigmas, 
                     color="orange", alpha=0.3, label="95% Confidence Interval")
    axs[0].scatter(anomalies, y_data[anomalies], color="black", label="Anomalies", zorder=3, s=3)
    axs[0].set_xlabel("Time Steps")
    axs[0].set_ylabel("Velocity")
     # veolcity plot
    axs[1].plot(forces, label="Forces", color="purple", linewidth=1.5)
    axs[1].set_xlabel("Time Steps")
    axs[1].set_ylabel("Forces")
    axs[1].set_title("Force Over Time")
    axs[1].legend()
    # Displacement plot
    axs[2].plot(displacement, label="Displacement", color="green", linewidth=1.5)
    axs[2].set_xlabel("Time Steps")
    axs[2].set_ylabel("Displacement")
    axs[2].set_title("Displacement Over Time")
    axs[2].legend()

    # Add vertical dashed lines in all subplots
    for i in range(1, len(axs)):
        ax = axs[i]
        for pos in anomalies:
            ax.axvline(pos, linestyle="dashed", color="black", alpha=0.7)

    plt.title("GP Regression with Self-Correlation")
    fig.legend()
    plt.show()


    return means, sigmas

# Main Execution
if __name__ == "__main__": 
    folder_path = "Logs"
    sampling_frequency = 1000    
    cutoff_time = 10.5
    passband = (5, 50)
   
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):  # Ensure it's a file
            print(f"Processing file: {file_path}")
        # Load data
        spike_detector = SpikeDetector(file_path, fs=sampling_frequency, time_window=cutoff_time, passband=passband) # read out metrics
        # spike_detector.plot_metrics()
        # spike_detector.causal_lowpass_filter(spike_detector.drilling_force, cutoff=3.0, order=4)
        # spike_detector.plot_frequency_bands_over_time(spike_detector.drilling_force, window_size=0.1, overlap=0.95)
        # real_time_autocorrelation(spike_detector.drilling_force, fs=spike_detector.fs, window_size=10)

        # compute_and_plot_stft(spike_detector.velocities, spike_detector.fs)

        # plt.scatter(indices, spike_detector.spectral_intensity[indices], color="red", label="Outliers", s=10, zorder=3)
        f_magnitude = spike_detector.drilling_force
        positions = spike_detector.displacement
        velocities = spike_detector.velocities
        # autocorrellation_cpd(f_magnitude, positions, noise_level=0.1)
        # Perform GP regression
        #plot_moving_averages(f_magnitude)
        means, sigmas = perform_gp_velocity(positions, velocities, f_magnitude, passband, sampling_frequency)
        # real_time_autocorrelation(f_magnitude, fs=sampling_frequency, window_size=50, time_shift=1)
        #slice = 20
          
        # Plot results
        #plot_results_sequential(spike_detector.dt_Fext_z_filtered, spike_detector.velocities, means, sigmas, indices, outliers, type="GP")

        # plot_velocity_force(f_magnitude, velocities)
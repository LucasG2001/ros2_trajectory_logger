import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from RTBandPassFilter import RealTimeBandpassFilter
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

def perform_gp_self_correlation_sklearn(f_magnitude, passband=(10, 50), sampling_frequency=500):
    """
    Perform Gaussian Process Regression with self-correlation on the force magnitude data USING SCIKIT-LEARN library. 
    Simulates a Datastream of the Force and uses the previous values to predict the next value(s).

    """
    # Parameters
    window_size = 100  # Sliding window size
    refit_interval = 50  # Refit every 10 iterations
    y_data = np.hstack([np.zeros(window_size + 1), f_magnitude]) # prepend y_data for fitting
    n_points = len(y_data)
    X_train = np.zeros([1, window_size]) # n_samples x n_features
    y_train = np.zeros([1, window_size]) # n_samples x n_targets
    print("shape of y data is ", y_data.shape)
    
    # Storage for predictions
    means = np.zeros(n_points)
    sigmas = np.ones(n_points) * 0.5
    anomalies = []
    times = []

    # Define GP kernel
    kernel = (
        C(1.0, constant_value_bounds="fixed") * RBF(length_scale=np.ones(window_size)* 1, length_scale_bounds="fixed"))
    
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.01, n_restarts_optimizer=5, normalize_y=False)
    
    print("Initial hyperparameters:\n", gp.get_params())

       # Filtering and GP Processing1
    for i in range(window_size + 1, n_points):
        print(i)
        start_time = time()
        # Construct feature vector: last `window_size` points
        feature_vector = y_data[i -window_size-1 : i-1].reshape(1, -1)
        l1 = compute_length_scale_from_fft(y_data[:i], sampling_frequency, sc = 5, plot = False)
        l = l1
        # Predict next sequence
        y_pred, y_std = gp.predict(feature_vector, return_std=True)
        means[i], sigmas[i] = y_pred.flatten()[-1], y_std.flatten()[-1]  # Extract only the prediction for `i+1`

        # Detect anomalies outside confidence interval
        lower_bound, upper_bound = means[i] - 1.96 * sigmas[i], means[i] + 1.96 * sigmas[i]
        if y_data[i] < lower_bound or y_data[i] > upper_bound:
            anomalies.append(i)

        if i % refit_interval == 0:
            X_train = np.vstack([X_train, feature_vector]) # Shape: (n_samples, window_size)
            y_train = np.vstack([y_train, y_data[i - window_size: i]])  # Shape: (n_samples, window_size)
            # Convert buffer to NumPy arrays (stacked feature vectors)
            # Compute the prior mean before fitting
            #prior_mean, _ = gp.predict(X_train, return_std=True)
            # print("Prior Mean (before training):\n", prior_mean)
            # Train GP on stacked data
            # gp.set_params(kernel__k1__k2__length_scale=np.ones(window_size)*l)
            # gp.set_params(kernel__k1__k1__constant_value=np.std(y_train))
            gp.set_params(kernel__k2__length_scale=np.ones(window_size)*l)
            gp.set_params(kernel__k1__constant_value=1)
            # Print updated hyperparameters
            #fit with subsampled points
            # print("\nUpdated hyperparameters:\n", gp.get_params())
            gp.fit(X_train[::1], y_train[::1]) # without normalize_y = True this will always assume a zero-mean prior
            # Compute the posterior mean after fitting
            #posterior_mean, _ = gp.predict(X_train, return_std=True)
            #print("Posterior Mean (after training):\n", posterior_mean)


        end_time = time()
        times.append(end_time - start_time)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(y_data, label="Original Data", color="blue", alpha=0.6, linewidth=1.0)
    plt.plot(means, label="GP Mean", color="red", linewidth=2.0)
    plt.fill_between(np.arange(len(means)), means - 1.96 * sigmas, means + 1.96 * sigmas, 
                     color="orange", alpha=0.3, label="95% Confidence Interval")
    plt.scatter(anomalies, y_data[anomalies], color="black", label="Anomalies", zorder=5, s=15)
    plt.xlabel("Time Steps")
    plt.ylabel("Force Magnitude")
    plt.title("GP Regression with Self-Correlation")
    plt.legend()
    plt.show()

    print("Average time per iteration: ", np.mean(times))

    return means, sigmas


def perform_gp_self_correlation(f_magnitude, passband=(10, 50), sampling_frequency=500):
    """
    Perform Gaussian Process Regression with self-correlation on the force magnitude data using GPy library. 
    Simulates a Datastream of the Force and uses the previous values to predict the next value(s).
    """
    # Parameters
    window_size = 25  # Sliding window size for datapoints
    refit_interval = 10  # Refit every "refit_interval" steps
    y_data = np.hstack([np.zeros(window_size + 1), f_magnitude])  # prepend zeros
    n_points = len(y_data)

    X_train = np.zeros([refit_interval, window_size])  # n_samples x n_features
    y_train = np.zeros([refit_interval, window_size])  # n_samples x n_targets

    print("Shape of y_data:", y_data.shape)
    
    # Storage for predictions
    means = np.zeros(n_points)
    sigmas = np.ones(n_points)
    anomalies = []
    times = []

    # Define GP kernel (Sum of RBF and a Constant term)
    kernel = GPy.kern.RBF(input_dim=window_size, lengthscale=np.ones(window_size) * 1.0, ARD=True) #+ GPy.kern.Bias(input_dim=window_size, variance=0.1)
    
    # Initialize GP model
    gp = GPy.models.GPRegression(np.zeros((1, window_size)), np.zeros((1, window_size)), kernel, noise_var=0.01)

    print("Initial hyperparameters:\n", gp)

    # Streaming GP Processing
    for i in range(window_size + 1, n_points):
        print(i)
        start_time = time()
        
        # Feature vector: last `window_size` points
        feature_vector = y_data[i - window_size - 1 : i - 1].reshape(1, -1)
        
        # Compute new length scale from FFT
        l1 = compute_length_scale_from_fft(y_data[:i-1], sampling_frequency, sc=5, plot=False)
        l = l1  # Use computed length scale
        
        # Predict next sequence
        y_pred, y_std = gp.predict(feature_vector)
        means[i], sigmas[i] = y_pred.flatten()[-1], y_std.flatten()[-1]  # Extract only the last prediction
        
        # Detect anomalies based on confidence interval
        lower_bound, upper_bound = means[i] - 1.96 * sigmas[i], means[i] + 1.96 * sigmas[i]
        if y_data[i] < lower_bound or y_data[i] > upper_bound:
            anomalies.append(i)

        # Store training data in buffer
        buffer_X = y_data[i - window_size - 1 : i - 1].reshape(1, -1)  # Feature vector
        buffer_y = y_data[i - window_size : i].reshape(1, -1)  # Target values
        X_train[i % refit_interval, :] = buffer_X
        y_train[i % refit_interval, :] = buffer_y

        # Refit GP every `refit_interval` steps
        if i % refit_interval == 0:
            # Update kernel hyperparameters
            gp.kern.lengthscale = np.ones(window_size) * l
            # gp.kern.bias.variance = np.var(y_train)

            # Set new data and refit
            gp.set_XY(X_train, y_train)
            # gp.optimize()

        end_time = time()
        times.append(end_time - start_time)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(y_data, label="Original Data", color="blue", alpha=0.6, linewidth=1.0)
    plt.plot(means, label="GP Mean", color="red", linewidth=2.0)
    plt.fill_between(np.arange(len(means)), means - 1.96 * sigmas, means + 1.96 * sigmas, 
                     color="orange", alpha=0.3, label="95% Confidence Interval")
    plt.scatter(anomalies, y_data[anomalies], color="black", label="Anomalies", zorder=5, s=15)
    plt.xlabel("Time Steps")
    plt.ylabel("Force Magnitude")
    plt.title("GP Regression with Self-Correlation")
    plt.legend()
    plt.show()

    print("Average time per iteration:", np.mean(times))

    return means, sigmas



# Step 3: Plot results
def plot_results_sequential(f_magnitude, velocities, means, sigmas, indices, outliers, type="GP"):
    """
    Plot the original force magnitude data and the learned predictions.
    type: title for plot (GP or BLR)
    """
    x = np.arange(len(f_magnitude))/500

    plt.figure(figsize=(12, 8))
    
    # Plot the original force magnitude data
    plt.plot(x, f_magnitude, label="Force Magnitude (|F_ext|)", color="blue", alpha=0.6)
    # plt.plot(x, ema_filter(f_magnitude), label="Force Magnitude filtered (|F_ext|)", color="green", alpha=0.9)
    
    # Plot the Regression Predictions
    plt.plot(x, means, color="red", label="GP Mean", linewidth=1.5, alpha = 0.8)
    # plt.scatter(np.array(indices)/500, outliers ,color="yellow",edgecolor="red",s=20,label="Outliers")
    plt.fill_between(x, means - 1.96 * sigmas, means + 1.96 * sigmas, color="orange", alpha=0.3, label="95% Confidence Interval")
    
    # Annotations
    plt.xlabel("Time (seconds)")
    plt.ylabel("Force Magnitude |F_ext|")
    plt.title("Sequential " + type + " Regression on Force Magnitude")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.scatter(velocities, means)
    plt.title("GPR Prediction Velocity vs Force Magnitude")
    plt.show()

def plot_moving_averages(f_magnitude):
    # Plot results
    fig = plt.figure(figsize=(12, 6))
    plt.plot(f_magnitude, label="Original Data", color="blue")
    plt.plot(moving_average_filter(f_magnitude, 10), label="MA 10", color="green", linestyle = "--")
    plt.plot(moving_average_filter(f_magnitude, 50), label="MA 100", color="orange", linestyle = "--")
    plt.plot(moving_average_filter(f_magnitude, 100), label="MA 250", color="red", linestyle = "--")
    plt.plot(moving_average_filter(f_magnitude, 2000), label="MA 1000", color="black", linestyle = "-")
    plt.title("Moving average analysis for GP")
    plt.legend()
    plt.show()
    
# Main Execution
if __name__ == "__main__": 
    folder_path = "med_drill_data"
    sampling_frequency = 1000    
    cutoff_time = 4.5
    passband = (10, 50)
   
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):  # Ensure it's a file
            print(f"Processing file: {file_path}")
        # Load data
        spike_detector = SpikeDetector(file_path, fs=sampling_frequency, time_window=cutoff_time, passband=passband) # read out metrics
        # spike_detector.plot_metrics()
        # spike_detector.causal_lowpass_filter(spike_detector.drilling_force, cutoff=3.0, order=4)
        # spike_detector.plot_frequency_bands_over_time(spike_detector.drilling_force, window_size=0.1, overlap=0.95)
        # spike_detector.plot_frequency_bands_over_time(spike_detector.drilling_force, window_size=0.1, overlap=0.95)
        # real_time_autocorrelation(spike_detector.drilling_force, fs=spike_detector.fs, window_size=10)

        # compute_and_plot_stft(spike_detector.velocities, spike_detector.fs)

        # plt.plot(spike_detector.spectral_intensity, color = "blue"3)5
        # plt.scatter(indices, spike_detector.spectral_intensity[indices], color="red", label="Outliers", s=10, zorder=3)
        f_magnitude = spike_detector.drilling_force
        positions = spike_detector.displacement
        velocities = spike_detector.velocities  
        autocorrellation_cpd(f_magnitude, noise_level=0.1)
        # Perform GP regression
        #plot_moving_averages(f_magnitude)
        # means, sigmas = perform_gp_self_correlation_sklearn(f_magnitude, passband, sampling_frequency)
        # TODO: EXTRACT SHIFTED AUTOCORRELATION AT EACH STEP AND FIT GAUSSIAN WITH RBF + WHITE NOISE KERNEL
        # TODO: THEN PREDICT AUTOCORRELATION FOR FUTURE STEPS
        # THIS MIGHT BE EXACTLY WHAT A GAUSSIAN PROCESS OF F WOULD DO
        # WE CAN INCREMENTALLY UPDATE THE VARIANCE BY SETTING SIGMA(K+1) = PREDICTED_STD(k)
        # BUT HOW DO WE SET THE LENGHT SCALE? THE MEAN IS ASSUMED 0 ANYWAYS. 
        # real_time_autocorrelation(f_magnitude, fs=sampling_frequency, window_size=50, time_shift=1)
        # real_time_autocorrelation(f_magnitude, fs=sampling_frequency, window_size=50, time_shift=10)
        # real_time_autocorrelation(f_magnitude, fs=sampling_frequency, window_size=50, time_shift=45)
        #slice = 20
        #"""
        #apply a GP regression with GPY once based on optimization based method and once based on the fourier transform
        #"""
        #for i in range(slice):
        #    n = len(f_magnitude)
        #    print(np.shape(f_magnitude))
        #    min_index = np.min([0, i * n//slice])
        #    max_index = np.min([(i + 1) * n//slice, n-1])
        #    # TODO: Why does it not fit when length scale is fixed?
        #    # THE PROBLEM IS COMPUTING VARIANCE/SUM TERM BUT IT IS NOT CLEAR WHY OR IF WE CAN JUST USE 1 INSTEAD
        #    # plot_frequency_bands_over_time(f_magnitude[min_index:max_index], sampling_frequency)
        #    _ = fit_gaussian_gp(f_magnitude[min_index:max_index], n_window=100, optimize=True) # optimized model
        #    _ = fit_gaussian_gp(f_magnitude[min_index:max_index], n_window=100, optimize=False) # fixed model

        #l = compute_length_scale_from_fft(f_magnitude, sampling_frequency)
        #print("l is ", l)
        # print("size of indices is ", len(indices))
        # Perform GP regression
        
        # Plot results
        #plot_results_sequential(spike_detector.dt_Fext_z_filtered, spike_detector.velocities, means, sigmas, indices, outliers, type="GP")
        # plot_results_sequential(f_magnitude, velocities, mu_blr, sigma_blr, type="BLR")

        # plot_velocity_force(f_magnitude, velocities)
import numpy as np
import GPy
import matplotlib.pyplot as plt
from filters import compute_length_scale_from_fft, compute_length_scale_from_fft_windowed, compute_autocorrelation, RealTimeBandpassFilter
import time

def fit_gaussian_gp(data: np.ndarray, n_window: int = 25, optimize: bool = True) -> GPy.models.GPRegression:
    """
    Fits a standard Gaussian Process (GP) regression model to predict a signal using its shifted version.
    
    Parameters:
        data (np.ndarray): 1D array of the signal data.
        n_window (int): Number of time steps to shift the signal. Defaults to 25.
        optimize (bool): Whether to optimize the GP hyperparameters. Defaults to True. If False the hyperparams are set via FFT analysis. 
        
    Returns:
        model (GPy.models.GPRegression): Trained GP model.
    """
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array.")
    
    N = len(data)
    print("length of data is ", N)
    # Pad the signal with zeros at the beginning
    padded_data = np.concatenate((np.zeros(n_window), data))
    # Sample every 100th point in the data
    sampled_indices = np.arange(n_window + 1, len(padded_data), 100)
    sampled_data = padded_data[sampled_indices]
    # Create input-output pairs for training
    X_test = np.array([padded_data[i - n_window - 1: i -1] for i in range(n_window + 1, len(padded_data))]) 
    X = np.array([padded_data[i - n_window - 1: i -1] for i in sampled_indices])
    print("sampled {} points".format(len(X)))
    Y = sampled_data.reshape(-1, 1)  # Target values
    print("length of Y is ", len(Y))
    
    # Reshape X to match expected input format for GPy
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    
    # Define and fit the GP model
    if optimize:
         kernel = GPy.kern.RBF(input_dim=n_window)
         model = GPy.models.GPRegression(X, Y, kernel)
    else:
        # Compute length scale from FFT
        l = compute_length_scale_from_fft_windowed(data, sc=50, fs=500, plot=True)
        l = np.sqrt(l)
        kernel = GPy.kern.RBF(input_dim=n_window, variance=1.0, lengthscale=l)
        model = GPy.models.GPRegression(X, Y, kernel, noise_var=0.03)
        # Constrain lengthscale between 0.5*l and 1.5*l
        model.kern.lengthscale.constrain_bounded(0.5 * l, 1.5 * l)

    #optimize the model
    model.optimize()

    # Print optimized model parameters
    print("Optimized model parameters:")
    print(model)
    # Plot the fitted model
    Y_pred, Y_var = model.predict(X_test)
    Y_std = np.sqrt(Y_var)
    
    plt.figure(figsize=(10, 5))
    plt.plot(data, '--b', label='True Signal')
    plt.plot(Y_pred, 'r', label='Predicted Signal')
    plt.fill_between(range(N-1), Y_pred.flatten() - 2 * Y_std.flatten(),
                     Y_pred.flatten() + 2 * Y_std.flatten(), color='r', alpha=0.2)
    
    # Plot the sampled points
    plt.scatter(sampled_indices - n_window, sampled_data, marker='o', color='k', label='Sampled Points')
    
    plt.legend()
    plt.title("Gaussian Process Regression with Sampled Points")
    plt.show()
    
    return model


def autocorrellation_cpd(f_magnitude, displacement, noise_level=0.1):
    detected_breakthrough = False
    # Parameters
    window_size = 40  # Sliding window size for datapoints
    time_shift = 20  # Time shift for the autocorrelation signal
    padding_dim = window_size + time_shift + 1
    y_data = np.hstack([np.zeros(padding_dim), f_magnitude])  # prepend zeros
    displacement = np.hstack([np.zeros(padding_dim), displacement])  # prepend zeros
    n_points = len(y_data)

    print("Shape of y_data:", y_data.shape)

    # Storage for predictions
    means = np.zeros(n_points)
    sigmas = np.ones(n_points) * noise_level
    alpha = 0.1
    autocorrelations = np.zeros(n_points)
    dominant_frequencies = np.ones(n_points) * 0.1
    dominant_magnitudes = np.zeros(n_points)
    filtered_signal = np.zeros(n_points)
    anomalies = []
    times = []
    bandpass_filter = RealTimeBandpassFilter(5, 25, 1000)

    # Streaming GP Processing
    for i in range(padding_dim, n_points):
        start_time = time.time()

        # Feature vector: last `window_size` points
        filtered_signal[i] = bandpass_filter.filter_sample(y_data[i])
        # Dominant frequency extraction
        windowed_signal = y_data[i - np.min([800, i-1]) : i]  # Time window of 400 observations
        print(len(windowed_signal))
        freqs = np.fft.rfftfreq(len(windowed_signal), d=1/1000)
        fft_spectrum = np.fft.rfft(windowed_signal)
        # fft_spectrum[0:int(5 * len(windowed_signal) / 1000)] = 0  # Zero out frequencies below 5 Hz
        dominant_frequency = np.max([freqs[np.argmax(np.abs(fft_spectrum))], 0.0001]) # set freq = 5 if 0 is dominant
        dominant_frequencies[i] = dominant_frequency
        print("Dominant frequency:", dominant_frequencies[i])
        dominant_index = np.argmax(np.abs(fft_spectrum))
        dominant_magnitudes[i] = np.abs(fft_spectrum[dominant_index])
        # use dominnt frequency to shift signal
        time_shift = np.min([int(1/dominant_frequency * 1000), int((i - 1)/3)]) # cap shift size
        time_shift = np.min([time_shift, 30])
        print("Time Shift:", time_shift)
        window_size = 2 * time_shift - 1
        print("Window Size:", window_size)
        shifted_signal = y_data[i - window_size - time_shift : i - time_shift].reshape(1, -1).flatten()
        feature_vector = y_data[i - window_size : i].reshape(1, -1).flatten()
        print("size of shifted signal is:", shifted_signal.shape)
        print("size of feature vector is:", feature_vector.shape)

        r_xx = np.correlate(feature_vector - np.mean(feature_vector), shifted_signal - np.mean(shifted_signal), mode='valid')/np.sqrt(window_size)
        autocorrelations[i] = np.abs(r_xx[0])

        means[i] = np.mean(autocorrelations[i - window_size:i])
        sigmas[i] = noise_level + alpha * np.std(autocorrelations[i - window_size:i]) + (1 - alpha) * sigmas[i - 1]

        # Detect anomalies based on confidence interval
        lower_bound, upper_bound = means[i] - 1.96 * sigmas[i], means[i] + 1.96 * sigmas[i]
        if (autocorrelations[i] < lower_bound or autocorrelations[i] > upper_bound) and displacement[i] < -0.001:
            if detected_breakthrough == False:
                anomalies.append(i)
                detected_breakthrough = True
            if detected_breakthrough == True and autocorrelations[i] < noise_level:
                detected_breakthrough = False

        end_time = time.time()
        times.append(end_time - start_time)

    # Create a figure with shared x-axis
    fig, axes = plt.subplots(5, 1, figsize=(10, 8), sharex=True)

    # Autocorrelation plot
    axes[0].plot(autocorrelations, label="Autocorrelation Measured", color="blue", alpha=0.6, linewidth=1.0)
    axes[0].plot(means, label="Expected Mean", color="red", linewidth=0.8)
    axes[0].scatter(anomalies, autocorrelations[anomalies], color="black", label="Anomalies", zorder=5, s=10)
    axes[0].fill_between(np.arange(len(means)), means - 1.96 * sigmas, means + 1.96 * sigmas, color="orange", alpha=0.3, label="95% Confidence Interval")
    axes[0].set_ylabel("Autocorrelation")
    axes[0].set_title("GP Regression with Self-Correlation")
    axes[0].legend()

    # Force magnitudes plot
    axes[1].plot(y_data, label="Force Magnitudes", color="green", linewidth=1.5)
    axes[1].plot(filtered_signal, label="filtered forces", color="purple", linewidth=1.5)
    axes[1].set_ylabel("F Magnitudes")
    axes[1].set_title("Force Magnitudes Over Time")
    axes[1].legend()

    # Dominant frequency plot
    axes[2].plot(dominant_frequencies, label="Dominant Frequency (Hz)", color="orange", linewidth=1.5)
    axes[2].set_ylim(0, 50)
    axes[2].set_ylabel("Frequency (Hz)")
    axes[2].set_title("Dominant Frequency Over Time")
    axes[2].legend()

    # Plot dominant frequency magnitude
    axes[3].plot(dominant_magnitudes, label="Dominant Frequency Magnitude", color="orange")
    axes[3].set_ylabel("Magnitude")
    axes[3].set_title("Magnitude of Dominant Frequency")
    axes[3].legend()

    # Displacement plot
    axes[4].plot(displacement, label="Displacement", color="purple", linewidth=1.5)
    axes[4].set_xlabel("Time Steps")
    axes[4].set_ylabel("Displacement")
    axes[4].set_title("Displacement Over Time")
    axes[4].legend()

    # Add vertical dashed lines in all subplots
    for ax in axes:
        for pos in anomalies:
            ax.axvline(pos, linestyle="dashed", color="black", alpha=0.7)

    plt.tight_layout()
    plt.show()
    print("Average time per iteration:", np.mean(times))

    return means, sigmas
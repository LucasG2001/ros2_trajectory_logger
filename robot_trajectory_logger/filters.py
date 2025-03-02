import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from collections import deque
from scipy.signal import spectrogram
from scipy.fft import fft, fftfreq
from helpers import update_plot
from scipy.signal import lombscargle
from tests import bandpass_filter_example, stft_spectogram_example

"""
different filters for gaussian process regression
"""

def compute_length_scale_from_fft_windowed(signal, fs, sc=250, num_bins=None, plot=True, window_type='hann', u_crossing=0.01):
    """
    Computes the characteristic length scale (l) based on the Fourier spectrum with windowing.

    Parameters:
    - signal: numpy array, the input time-domain signal.
    - fs: float, the sampling frequency (Hz).
    - sc: float, cutoff frequency (Hz), default is 250 Hz.
    - num_bins: int, number of frequency bins to use in FFT. Default is signal length.
    - plot: bool, whether to plot the FFT spectrum.
    - window_type: str, type of window function to apply ('hann', 'hamming', 'blackman', 'kaiser').
    - u_crossing: float, threshold to filter out small Fourier coefficients (default = 0.01).

    Returns:
    - l: float, computed length scale.
    """
    # Set the number of bins (default to signal length if not provided)
    N = num_bins if num_bins is not None else 500 #len(signal)
    var = np.var(signal)

    # Compute FFT of the original (unwindowed) signal
    F_train_unwindowed = np.fft.fft(signal, n=N, norm='forward')

    # Apply window function
    if window_type == 'hann':
        window = np.hanning(len(signal))
    elif window_type == 'hamming':
        window = np.hamming(len(signal))
    elif window_type == 'blackman':
        window = np.blackman(len(signal))
    elif window_type == 'kaiser':
        window = np.kaiser(len(signal), beta=8)
    else:
        raise ValueError("Invalid window type. Choose from 'hann', 'hamming', 'blackman', or 'kaiser'.")

    # Normalize window to maintain power
    windowed_signal = signal * window
    window_correction = np.sum(window**2) / len(window)  # Energy normalization

    # Compute FFT with windowed signal
    F_train_windowed = np.fft.fft(windowed_signal, n=N, norm='forward') / np.sqrt(window_correction)

    # Generate frequency bins
    freqs = np.fft.fftfreq(N, d=1/fs)
    df = fs / N  # Frequency step size (Δf)

    # Apply cutoff frequency filter
    mask = (np.abs(freqs) <= sc)
    freqs_filtered = freqs[mask]
    F_filtered_windowed = F_train_windowed[mask]

    # Apply u_crossing threshold: Remove small coefficients
    F_filtered_windowed[np.abs(F_filtered_windowed) < u_crossing] = 0
    print("deleting coefficients below 0.01 ", np.abs(F_filtered_windowed) < u_crossing)

    # Compute the sum term in the formula (integral approximation)
    sum_term = np.sum(4 * (np.pi**2) * (freqs_filtered**2) * np.abs(F_filtered_windowed)**2)

    # Compute length scale (l)
    l = np.sqrt(1 / sum_term) if sum_term > 0 else 1

    # Plot both FFTs if requested
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(freqs, np.abs(F_train_unwindowed), 'b-', alpha=0.6, label="Unwindowed FFT")
        plt.plot(freqs_filtered, np.abs(F_filtered_windowed), 'r-', alpha=0.8, label=f"Windowed FFT ({window_type.capitalize()})")
        plt.axvline(sc, color='k', linestyle='--', label=f"Cutoff: {sc} Hz")
        plt.xlabel("Frequency (Hz)") 
        plt.ylabel("Magnitude")
        plt.title(f"FFT Spectrum: Windowed vs Unwindowed ({window_type.capitalize()} Window)")
        plt.legend()
        plt.grid()
        plt.show()

    return l

def compute_length_scale_example(freq = 1, fs=500):
    """
    Example usage for length scale computation. Computes the length scale from Fourier spectrum. Plots the spectrum.
    Assumes signal length is 1 second.
    freq: Frequency of the signal
    signal: Signal to compute the length scale from
    fs: Sampling frequency
    """
    # Example usage for length scale computation
    t = np.linspace(0, 1, fs, endpoint=False)  # 1-second signal
    signal = np.sin(2*np.pi*freq*t) 
    l = compute_length_scale_from_fft(signal, fs)
    print(f"Computed Length Scale: {l:.4f}")

def real_time_autocorrelation(signal: np.ndarray, fs: float, window_size: int):
    #TODO fix autocorrelation function
    #TODO write unit tests
    #TODO it shouls probably not contain the whole signal until time t but a part of it
    """
    Simulates real-time streaming of a time-series signal,
    computing and plotting the autocorrelation over time.

    Parameters:
    - signal: np.ndarray : Time-series data
    - fs: float : Sampling frequency (Hz)
    - window_size: int : Window size for computing autocorrelation (samples)
    """
    autocorr_values = []
    n = len(signal)
    time_axis = np.arange(n) / fs
   
    
    for i in range(window_size + 1, n):
        observed_signal = signal[window_size : i]
        shifted_signal = signal[0: i - window_size]
        var = np.var(observed_signal) * np.var(shifted_signal)
        print("var", var)  
        if var <= 0.001:
            autocorr_values.append(1.0)  # var = 0 is either constant signal or first measurement
        else:
            autocorr = np.correlate(observed_signal - np.mean(observed_signal), shifted_signal-np.mean(shifted_signal), mode='full') / (n * var) #normalization
            autocorr = autocorr[len(autocorr)//2:]  # Keep only the positive lags
            autocorr_values.append(autocorr[0])  # Store only the zero-lag autocorrelation
    
 # Plot results
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    
    ax[0].plot(time_axis[window_size + 1:], autocorr_values, label='Autocorrelation'+ " " + "Window = "+str(window_size), color='red')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Autocorrelation')
    ax[0].legend()
    ax[0].set_title('Real-time Autocorrelation')
    
    ax[1].plot(time_axis, signal, label='Original Signal', alpha=0.5)
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Signal')
    ax[1].legend()
    ax[1].set_title('Original Signal')
    
    plt.tight_layout()
    plt.show()

def compute_length_scale_from_fft(signal, fs, sc=250, num_bins=None, plot=True):
    """
    Computes the characteristic length scale (l) based on the Fourier spectrum.

    Parameters:
    - signal: numpy array, the input time-domain signal.
    - fs: float, the sampling frequency (Hz).
    - sc: float, cutoff frequency (Hz), default is 250 Hz.
    - num_bins: int, number of frequency bins to use in FFT. Default is signal length.

    Returns:
    - l: float, computed length scale.
    """
    # Set the number of bins (default to signal length if not provided)
    N = num_bins if num_bins is not None else len(signal)
    var = np.var(signal)
 
    # Compute FFT with num_bins points
    F_train = np.fft.fft(signal, n=N, norm='forward')
    
    # Generate frequency bins corresponding to the FFT output
    freqs = np.fft.fftfreq(N, d=1/fs)  # Generate frequency bins
    df = fs / N  # Frequency step size (Δf)

    # Select frequencies within [-sc, sc] (cutoff frequency)
    mask = (np.abs(freqs) <= sc)
    freqs_filtered = freqs[mask]
    F_filtered = F_train[mask]

    # Compute the sum term in the formula (integral approximation)
    sum_term = np.sum(4 * (np.pi**2) * (freqs_filtered**2) * np.abs(F_filtered)**2)

    # Compute length scale (l) as sqrt of the inverse of the integral term
    l = np.sqrt(var / sum_term) if sum_term > 0 else 1

    # Plot the DFT spectrum if requested (using stem plot)
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(freqs, np.abs(F_train), 'b-', label="Magnitude of FFT")  # Use a normal plot instead of stem
        plt.axvline(sc, color='r', linestyle='--', label=f"Cutoff: {sc} Hz")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.title("Discrete Fourier Transform (FFT) Spectrum")
        plt.legend()
        plt.grid()
        plt.show()

    return l


def normalize_array(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    return (arr - mean) / std


def ema_filter(data, alpha=0.1):
    """
    Apply an Exponential Moving Average (EMA) filter to a NumPy array.
    
    Parameters:
        data (np.ndarray): The input array to be filtered.
        alpha (float): The smoothing factor, 0 < alpha <= 1.
                      Smaller alpha gives more smoothing.
                      
    Returns:
        np.ndarray: The EMA-filtered array.
    """
    if not 0 < alpha <= 1:
        raise ValueError("Alpha should be in the range (0, 1].")
    
    ema = np.zeros_like(data)
    ema[0] = data[0]  # Initialize with the first data point
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema

def moving_average_filter(data, window_size=10):
    """
    Apply a Moving Average (MA) filter to a NumPy array.
    
    Parameters:
        data (np.ndarray): The input array to be filtered.
        window_size (int): The size of the moving window (default: 10).
        
    Returns:
        np.ndarray: The MA-filtered array, with edges padded with NaN.
    """
    if window_size < 1:
        raise ValueError("Window size must be greater than or equal to 1.")
    
    cumsum = np.cumsum(np.insert(data, 0, 0))
    ma = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    
    # Pad the start of the array with NaN to match the input length
    return np.concatenate((np.full(window_size - 1, 0), ma))

class RealTimeBandpassFilter:
    def __init__(self, lowcut, highcut, fs, order=2):
        """
        Initializes a real-time Butterworth bandpass filter.

        :param lowcut: Lower cutoff frequency (Hz)
        :param highcut: Upper cutoff frequency (Hz)
        :param fs: Sampling frequency (Hz)
        :param order: Filter order (default: 2 for minimal lag)
        """
        nyquist = 0.5 * fs  # Nyquist frequency
        low = lowcut / nyquist
        high = highcut / nyquist

        # Design the filter in Second-Order Sections (sos) form
        self.sos = signal.butter(order, [low, high], btype='band', output='sos')

        # Initialize filter state (correctly sets past values)
        self.zi = signal.sosfilt_zi(self.sos)  # Initial conditions for each section

    def filter_sample(self, sample):
        """
        Filters a single data sample in real time.

        :param sample: Incoming signal sample (scalar)
        :return: Filtered signal sample (scalar)
        """
        filtered_sample, self.zi = signal.sosfilt(self.sos, [sample], zi=self.zi)
        return filtered_sample[0]  # Extract single value
    

def compute_and_plot_stft(signal_data, fs, nperseg=256, noverlap=None, nfft=None, window="hann", plot=True):
    """
    Computes and visualizes the Short-Time Fourier Transform (STFT) of a given signal.
    
    Parameters:
    - signal_data (numpy array): The 1D array containing the time-domain signal.
    - fs (float): Sampling frequency in Hz.
    - nperseg (int, optional): Number of samples per segment (window size). Default is 256.
    - noverlap (int, optional): Number of overlapping samples between consecutive segments.
                                Default is nperseg // 2 (50% overlap).
    - nfft (int, optional): Number of FFT points. Default is nperseg.
    - window (str or array-like, optional): Window function applied to each segment.
                                            Default is "hann" (Hann window).
    - plot (bool, optional): If True, plots the spectrogram and frequency spectrum.
    
    Returns:
    - f (numpy array): Array of frequency bins.
    - t (numpy array): Array of time bins (center of each window segment).
    - Zxx (numpy array): STFT complex-valued matrix (frequency-time representation).
    """

    # Set default values if not provided
    if noverlap is None:
        noverlap = nperseg // 2  # Default to 50% overlap
    if nfft is None:
        nfft = nperseg  # Default FFT size to segment size

    # Compute STFT using scipy.signal.stft
    f, t, Zxx = signal.stft(
        signal_data,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=False,
        return_onesided=True
    )

    # Only keep frequencies up to Nyquist frequency (the rest will be aliased anyway)
    nyquist_freq = fs / 2
    valid_indices = f <= nyquist_freq
    f = f[valid_indices]
    Zxx = Zxx[valid_indices, :]


    if plot:
        # Plot Spectrogram
        plt.figure(figsize=(12, 6))
        
        # Spectrogram Plot
        plt.subplot(2, 1, 1)
        plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
        plt.colorbar(label="Magnitude")
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (s)")
        plt.title("STFT Spectrogram")

        # Frequency Spectrum at a Given Time
        # Initialize plot
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)  # Leave space for slider
        # Initial time index
        time_index = len(t) // 2  # Start at middlE
        # Plot initial frequency spectrum
        spectrum_line, = ax.plot(f, np.abs(Zxx[:, time_index]), label=f"Spectrum at t = {t[time_index]:.2f}s")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.set_title("Frequency Spectrum at a Specific Time")
        ax.legend()

        # Create slider axis
        ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
        slider = Slider(ax_slider, "Time Index", 0, len(t) - 1, valinit=time_index, valstep=1)

        # Connect slider to the update function
        slider.on_changed(lambda val: update_plot(val, slider, spectrum_line, Zxx, t, ax, fig))

        plt.show()

    return f, t, Zxx

def mad_outlier_detection(point, data, threshold=3.5):
    """
    Detects outliers in a dataset using the Median Absolute Deviation (MAD) method.
    
    Parameters:
    - data: numpy array, the input data.
    - threshold: float, the number of MADs a point must be away from the median to be considered an outlier.
    - point: data point that is checked to be an outlier
    Returns:
    - A boolean indicating if the latest data point is an outlier.
    """
    if len(data) < 2:
        return False, point, 1.5  # Not enough data to determine outliers
    
    b = 1.4826 # distribuition constant for normal distribution
    median = np.median(data)
    mad = b * np.median(np.abs(data - median))

    if median + threshold*mad < point or median-threshold*mad > point:
        return True, median, mad # is outlier
    else:
        return False, median, mad

def real_time_outlier_detection(stream, window_size=100, threshold=3.5):
    """
    Performs real-time outlier detection using a sliding window approach.
    
    Parameters:
    - stream: iterable, incoming data points.
    - window_size: int, size of the analysis window.
    - threshold: float, MAD threshold for detecting outliers.
    """
    window = deque(maxlen=window_size)
    outliers = []
    
    for point in stream:
        window.append(point)
        is_outlier, median, mad = mad_outlier_detection(point, window, threshold)
        outliers.append(is_outlier)
        yield point, is_outlier, median, mad

def plot_real_time_outliers(data, outliers):
    """
    Plots the real-time data with detected outliers highlighted.
    
    Parameters:
    - data: numpy array, the input data.
    - outliers: boolean array, mask of outliers detected.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data, '-', label='Data')
    plt.plot(np.where(outliers)[0], np.array(data)[outliers], 'o', label='Outliers', linewidth=2)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Real-Time Outlier Detection using MAD")
    plt.legend()
    plt.show()

def plot_spectral_intensity(signal, fs, f_min=0, f_max=500):
    """
    Plots the absolute total spectral intensity within a given frequency band over time.

    Parameters:
        signal (numpy array): The input signal.
        fs (float): Sampling frequency of the signal.
        f_min (float): Lower frequency bound.
        f_max (float): Upper frequency bound.
    """
    # Compute the spectrogram
    f, t, Sxx = spectrogram(signal, fs=fs, window="hann", nperseg=16, noverlap=8, scaling='density')

    # Select frequencies within the passband
    mask = (f >= f_min) & (f <= f_max)
    total_intensity = np.sum(Sxx[mask, :], axis=0)  # Sum spectral power in passband

    # Plot the total spectral intensity over time
    plt.figure(figsize=(10, 5))
    plt.plot(t, np.abs(total_intensity), label=f'Total Spectral Intensity ({f_min}-{f_max} Hz)', color='b')
    plt.xlabel('Time [s]')
    plt.ylabel('Spectral Intensity')
    plt.title(f'Total Spectral Intensity in {f_min}-{f_max} Hz Passband')
    plt.legend()
    plt.grid()
    plt.show()

def plot_frequency_bands_over_time(signal, fs, window_size=1.0, overlap=0.5):
    """
    Computes and plots the percentage contribution of different frequency bands over time.

    Parameters:
    - signal: numpy array, the input time-domain signal.
    - fs: float, the sampling frequency (Hz).
    - window_size: float, duration of each STFT window in seconds (default = 1.0s).
    - overlap: float, overlap fraction between windows (default = 50%).
    """
    # Define frequency bands
    bands = {"0-5 Hz": (0, 5), "5-25 Hz": (5, 25), "25-50 Hz": (25, 50), "50-250 Hz": (50, 250)}

    # Compute STFT
    nperseg = int(window_size * fs)  # Convert window size from seconds to samples
    noverlap = int(overlap * nperseg)  # Overlap in samples
    f, t, Zxx = signal.stft(signal, fs, nperseg=nperseg, noverlap=noverlap, window='hann')

    # Compute energy in each frequency band
    band_energies = {}
    for band_name, (f_low, f_high) in bands.items():
        band_mask = (f >= f_low) & (f <= f_high)
        band_energies[band_name] = np.sum(np.abs(Zxx[band_mask, :])**2, axis=0)  # Compute total energy in band

    # Convert energy to percentage contribution at each time step
    total_energy = np.sum(list(band_energies.values()), axis=0)  # Sum energy across all bands
    for band_name in band_energies.keys():
        band_energies[band_name] = (band_energies[band_name] / total_energy) * 100  # Convert to percentage

    # Plot frequency band percentage contributions over time
    plt.figure(figsize=(10, 5))
    for band_name, energy_percent in band_energies.items():
        plt.plot(t, energy_percent, label=band_name)

    plt.xlabel("Time (s)")
    plt.ylabel("Percentage Contribution (%)")
    plt.title("Frequency Band Contribution Over Time")
    plt.legend()
    plt.grid()
    plt.show()



# Example Usage
if __name__ == "__main__":
    
    # bandpass_filter_example()
    # stft_spectogram_example()
    #compute_length_scale_example(freq=1.0, fs=500)
    # Real-time Autocorrelation Example
    np.random.seed(42)
    fs = 100  # Sampling frequency in Hz
    t = np.arange(0, 10, 1/fs)  # 10 seconds of data
    signal = np.sin(2 * np.pi * 1 * t) # + 0.5 * np.random.randn(len(t))  # 1 Hz sine wave with noise
    real_time_autocorrelation(signal, fs=fs, window_size=5)
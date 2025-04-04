from scipy import signal
import numpy as np


from scipy import signal

class RealTimeLowpassFilter:
    def __init__(self, cutoff, fs, order=2):
        """
        Initializes a real-time Butterworth lowpass filter.

        :param cutoff: Cutoff frequency (Hz)
        :param fs: Sampling frequency (Hz)
        :param order: Filter order (default: 2 for minimal lag)
        """
        nyquist = 0.5 * fs  # Nyquist frequency
        normalized_cutoff = cutoff / nyquist

        # Design the filter in Second-Order Sections (sos) form
        self.sos = signal.butter(order, normalized_cutoff, btype='low', output='sos')

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


"""simulates a real-time bandpass filter using a Butterworth filter design."""

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
    


# Example Usage
if __name__ == "__main__":
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
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(t, raw_signal, label="Raw Signal", alpha=0.5)
    plt.plot(t, filtered_signal, label="Filtered Signal", linewidth=2)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Real-Time Butterworth Bandpass Filter (5-15 Hz)")
    plt.show()
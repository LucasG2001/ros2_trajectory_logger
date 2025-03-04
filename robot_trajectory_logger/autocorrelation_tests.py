import numpy as np
import matplotlib.pyplot as plt

def generate_sine_wave(frequencies=[0, 5, 25], duration=1.0, sampling_rate=1000):
    """Generates a sine wave composed of multiple frequencies."""
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = sum(np.sin(2 * np.pi * f * t) for f in frequencies)
    return t, signal

def compute_autocorrelation(signal, shift):
    """Computes the normalized autocorrelation of a signal with its shifted version."""
    shifted_signal = np.roll(signal, shift)
    autocorr = np.correlate(signal, shifted_signal, mode='full')
    mid = len(autocorr) // 2
    
    # Normalize using signal length and variance
    norm_factor = len(signal) * np.var(signal)
    normalized_autocorr = autocorr[mid] / norm_factor if norm_factor != 0 else 0
    
    return normalized_autocorr, shifted_signal  # Return normalized autocorr value at given shift

def plot_results(t, signal, shifted_signals, autocorrelations, shifts, sampling_rate):
    """Plots the original and shifted signals along with the normalized autocorrelation values."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    # Plot original and shifted signals
    axes[0].plot(t, signal, label='Original Signal', alpha=0.7)
    for shift, shifted_signal in zip(shifts, shifted_signals):
        axes[0].plot(t, shifted_signal, label=f'Shifted (Shift={shift})', linestyle='dashed')
    axes[0].set_title('Original and Shifted Signals')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    
    # Plot normalized autocorrelation values
    axes[1].scatter(shifts, autocorrelations, color='r', label='Normalized Autocorr at Shifts')
    axes[1].set_title('Normalized Autocorrelation at Selected Shifts')
    axes[1].set_xlabel('Shift (Samples)')
    axes[1].set_ylabel('Normalized Autocorrelation')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

def main():
    frequencies = [0, 5, 25]  # Modify these frequencies as needed
    duration = 1.0  # seconds
    sampling_rate = 1000  # Hz
    shifts = [20, 50, 100]  # Different shift values to test
    
    t, signal = generate_sine_wave(frequencies, duration, sampling_rate)
    autocorrelations = []
    shifted_signals = []
    
    for shift in shifts:
        autocorr, shifted_signal = compute_autocorrelation(signal, shift)
        autocorrelations.append(autocorr)
        shifted_signals.append(shifted_signal)
        print(f'Normalized Autocorrelation at shift {shift}: {autocorr}')
    
    plot_results(t, signal, shifted_signals, autocorrelations, shifts, sampling_rate)

if __name__ == "__main__":
    main()

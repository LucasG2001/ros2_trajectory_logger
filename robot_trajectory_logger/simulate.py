import numpy as np
import matplotlib.pyplot as plt

# Sampling parameters
fs = 500  # Sampling frequency in Hz
duration = np.random.uniform(3, 6)  # Random duration between 3 and 6 sec
n_samples = int(duration * fs)  # Total number of samples
time = np.linspace(0, duration, n_samples)

# Define sequence durations
seq2_duration = np.random.uniform(0.3, 0.9)
remaining_time = duration - seq2_duration
seq1_duration = remaining_time / 2
seq3_duration = remaining_time / 2

# Compute indices
seq1_end = int(seq1_duration * fs)
seq2_end = seq1_end + int(seq2_duration * fs)

# Function to generate a weighted sum of sine waves
def generate_signal(f_min, f_max, num_components, duration, fs, min_amp = 1, max_amp=3):
    t = np.linspace(0, duration, int(duration * fs))
    signal = np.zeros_like(t)
    for _ in range(num_components):
        f = np.random.uniform(f_min, f_max)
        amp = np.random.uniform(min_amp, max_amp)
        weight = np.random.uniform(0.1, 1)
        signal += weight * amp * np.sin(2 * np.pi * f * t + np.random.uniform(0, 2 * np.pi))
    return signal

# Generate sequences
seq1 = generate_signal(0, 5, 5, seq1_duration, fs, 0.1, 2)
seq2 = generate_signal(0, 5, 2, seq2_duration, fs, 0.1, 3) + generate_signal(5, 25, 3, seq2_duration, fs, 3, 6)
seq3 = generate_signal(0, 5, 5, seq3_duration, fs, 0.1, 2)

# Combine sequences
signal = np.concatenate((seq1, seq2, seq3))

# Generate heteroscedastic noise
noise1 = np.linspace(0.05, 0.5, seq1_end) * np.random.randn(seq1_end)
noise2 = np.linspace(0.5, 0.05, seq2_end - seq1_end) * np.random.randn(seq2_end - seq1_end)
noise3 = np.linspace(0.5, 0.05, n_samples - seq2_end) * np.random.randn(n_samples - seq2_end)

# Add noise to signal
signal += np.hstack((noise1, noise2, noise3))

# Plot the signal
plt.figure(figsize=(12, 6))
plt.plot(time, signal, label='Heteroscedastic Signal')
plt.axvline(seq1_duration, color='r', linestyle='--', label='Seq1 -> Seq2')
plt.axvline(seq1_duration + seq2_duration, color='g', linestyle='--', label='Seq2 -> Seq3')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Simulated Heteroscedastic Process')
plt.legend()
plt.show()
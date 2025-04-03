from scipy.signal import butter, lfilter_zi, lfilter
import json
import numpy as np
import matplotlib.pyplot as plt

class ButterworthLowPass:
    def __init__(self, sample_rate, cutoff_freq, order=2):
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        self.b, self.a = butter(order, normal_cutoff, btype='low', analog=False)

        # Initial filter state (zi holds previous inputs/outputs)
        self.zi = lfilter_zi(self.b, self.a) * 0  # start at zero signal

    def process(self, x):
        """Filter a single input sample."""
        y, self.zi = lfilter(self.b, self.a, [x], zi=self.zi)
        return y[0]

    def reset(self):
        self.zi = lfilter_zi(self.b, self.a) * 0

def load_log_file(filename):
    """
    Load the JSON log file.

    Parameters:
    - filename (str): Path to the JSON log file.

    Returns:
    - List of dictionaries containing the log data.
    """
    with open(filename, 'r') as file:
        data = [json.loads(line.strip()) for line in file]
    return data

def extract_data(data):

    timestamps = list(range(len(data)))

    velocity_desired = []

    for entry in data:
        # Extract velocity desired
        velocity_desired.append(entry['velocity_desired'])

    return timestamps, velocity_desired


if __name__ == "__main__":
    fs = 1000    # sample rate
    fc = 7     # cutoff frequency
    
    # Import data stream and initialize the AODDS model
    # Path to your JSON log file
    logfile = '/home/nilsjohnson/franka_ros2_ws/src/ros2_trajectory_logger/robot_state_log_2025_02_28_1404.json'

    # Load and process the log file
    data = load_log_file(logfile)

    timestamps, velocity_desired = extract_data(data)

    sampling_rate = 1000  # 1000 Hz update rate

    acceleration_desired = sampling_rate * np.diff(velocity_desired, prepend=velocity_desired[1])

    # Simulate streaming signal (replace this with live sensor/audio/etc)
    signal = acceleration_desired

    filt = ButterworthLowPass(fs, fc)

    filtered_output = []

    for sample in signal:
        filtered_output.append(filt.process(sample))

    
    fig, axs = plt.subplots(3, 1, figsize=(18, 18), sharex=True)

    # plot the unfiltered and filtered signal
    # Plot the displacement (Z-axis)
    axs[0].plot(timestamps, acceleration_desired, label="accel", color='blue')
    axs[0].set_xlabel("Timestamps")
    axs[0].set_ylabel("Acceleration [m/s^2]")
    axs[0].legend()
    axs[0].grid(True)

    # Plot the filtered force (desired-axis)
    axs[1].plot(timestamps, filtered_output, label="accel filtered", color='green')
    axs[1].set_xlabel("Timestamps")
    axs[1].set_ylabel("accel filtered [m/s^2]")
    axs[1].legend()
    axs[1].grid(True)

    # plot veolocity
    axs[2].plot(timestamps, velocity_desired, label="velocity", color='red')
    axs[2].set_xlabel("Timestamps")
    axs[2].set_ylabel("Velocity [m/s]")
    axs[2].legend()
    axs[2].grid(True)
    

    # Show the plot
    plt.tight_layout()
    plt.show()




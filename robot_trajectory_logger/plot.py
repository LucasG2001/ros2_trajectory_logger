import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, freqz

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
    """
    Extract data from the list of dictionaries.

    Parameters:
    - data (list): List of dictionaries containing the log data.

    Returns:
    - timestamps (list): List of timestamps.
    - forces (dict): Dictionary of forces (x, y, z).
    - torques (dict): Dictionary of torques (x, y, z).
    - reference_positions (dict): Dictionary of reference positions (x, y, z).
    - euler_angles (dict): Dictionary of Euler angles (roll, pitch, yaw).
    - ee_positions (dict): Dictionary of end-effector positions (x, y, z).
    - ee_orientations (dict): Dictionary of end-effector orientations (roll, pitch, yaw).
    """
    timestamps = list(range(len(data)))  # Assuming each entry represents a time step

    forces = {'x': [], 'y': [], 'z': []}
    torques = {'x': [], 'y': [], 'z': []}
    reference_positions = {'x': [], 'y': [], 'z': []}
    euler_angles = {'roll': [], 'pitch': [], 'yaw': []}
    ee_positions = {'x': [], 'y': [], 'z': []}
    ee_orientations = {'roll': [], 'pitch': [], 'yaw': []}
    accelerations = {'x': [], 'y': [], 'z': []}

    for entry in data:
        # Extract force data
        forces['x'].append(entry['f_ext']['force']['x'])
        forces['y'].append(entry['f_ext']['force']['y'])
        forces['z'].append(entry['f_ext']['force']['z'])

        # Extract torque data
        torques['x'].append(entry['f_ext']['torque']['x'])
        torques['y'].append(entry['f_ext']['torque']['y'])
        torques['z'].append(entry['f_ext']['torque']['z'])

        # Extract reference position data
        reference_positions['x'].append(entry['reference_position']['x'])
        reference_positions['y'].append(entry['reference_position']['y'])
        reference_positions['z'].append(entry['reference_position']['z'])

        # Extract Euler angles data
        euler_angles['roll'].append(entry['euler_angles']['roll'])
        euler_angles['pitch'].append(entry['euler_angles']['pitch'])
        euler_angles['yaw'].append(entry['euler_angles']['yaw'])

        # Extract end-effector position data
        ee_positions['x'].append(entry['ee_pose']['position']['x'])
        ee_positions['y'].append(entry['ee_pose']['position']['y'])
        ee_positions['z'].append(entry['ee_pose']['position']['z'])

        # Extract end-effector orientation data
        ee_orientations['roll'].append(entry['ee_pose']['orientation']['roll'])
        ee_orientations['pitch'].append(entry['ee_pose']['orientation']['pitch'])
        ee_orientations['yaw'].append(entry['ee_pose']['orientation']['yaw'])


    return timestamps, forces, torques, reference_positions, euler_angles, ee_positions, ee_orientations

""" 
updating_limits = False

def on_xlim_changed(ax, other_axes):
    global updating_limits
    if updating_limits:
        return  # Prevent recursion
    
    updating_limits = True  # Set flag to indicate that we're updating limits
    xlim = ax.get_xlim()  # Get the current x-axis limits
    for other_ax in other_axes:
        other_ax.set_xlim(xlim)  # Apply the same limits to other axes
    updating_limits = False  # Reset flag after updating limits
    plt.draw()

def on_ylim_changed(ax, other_axes):
    global updating_limits
    if updating_limits:
        return  # Prevent recursion
    
    updating_limits = True  # Set flag to indicate that we're updating limits
    ylim = ax.get_ylim()  # Get the current y-axis limits
    for other_ax in other_axes:
        other_ax.set_ylim(ylim)  # Apply the same limits to other axes
    updating_limits = False  # Reset flag after updating limits
    plt.draw() 
"""

def moving_average(data, window_size):
    """ Computes the moving average of the given data using a window of the specified size. """
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, 'same')  # 'same' returns the convolution at each point of the original array

def calculate_accelerations(positions, delta_t):
    accelerations = {'x': [], 'y': [], 'z': []}

    for axis in ['x', 'y', 'z']:
        # Calculate the first derivative of positions to get velocities
        velocities = np.diff(positions[axis]) / delta_t
        # Calculate the second derivative of velocities to get accelerations
        axis_accelerations = np.diff(velocities) / delta_t
        # Set accelerations higher than 16 to 0
        axis_accelerations = np.where(np.abs(axis_accelerations) > 16, 0, axis_accelerations)
        accelerations[axis] = axis_accelerations

    return accelerations

def apply_ema(data, alpha):
    ema_data = {}
    for key in data:
        ema_data[key] = []
        for i in range(len(data[key])):
            if i == 0:
                ema_data[key].append(data[key][i])
            else:
                ema_data[key].append(alpha * data[key][i] + (1 - alpha) * ema_data[key][i - 1])
    return ema_data

def truncate_to_shortest_length(*args):
    # Initialize the minimum length with a large number
    min_length = float('inf')
    
    # Iterate through each argument to find the minimum length
    for arg in args:
        if isinstance(arg, dict):
            # Find the shortest length for each dictionary
            for key in arg:
                min_length = min(min_length, len(arg[key]))
        elif isinstance(arg, list) or isinstance(arg, range):
            min_length = min(min_length, len(arg))

    # Truncate each argument to the minimum length
    truncated_args = []
    for arg in args:
        if isinstance(arg, dict):
            truncated_arg = {key: value[:min_length] for key, value in arg.items()}
            truncated_args.append(truncated_arg)
        else:
            truncated_args.append(arg[:min_length])
    
    return truncated_args

def compute_deltas(data, window_size, sampling_rate):
    """
    Computes the change in data over specified window size.

    Parameters:
    - data (list): Input data list.
    - window_size (float): Window size in milliseconds.
    - sampling_rate (int): Sampling rate in Hz.

    Returns:
    - List of deltas.
    """
    window_points = int(window_size / 1000 * sampling_rate)
    return [data[i + window_points] - data[i] if i + window_points < len(data) else None for i in range(len(data))]



def plot_data(timestamps, forces, torques, reference_positions, euler_angles, ee_positions, ee_orientations,accelerations):
    """
    Plot the extracted data.

    Parameters:
    - timestamps (list): List of timestamps.
    - forces (dict): Dictionary of forces (x, y, z).
    - torques (dict): Dictionary of torques (x, y, z).
    - reference_positions (dict): Dictionary of reference positions (x, y, z).
    - euler_angles (dict): Dictionary of Euler angles (roll, pitch, yaw).
    - ee_positions (dict): Dictionary of end-effector positions (x, y, z).
    - ee_orientations (dict): Dictionary of end-effector orientations (roll, pitch, yaw).
    """
    fig, axs = plt.subplots(7, 3, figsize=(18, 18))

    # Calculate differences (diff of forces)
    force_diff = {
    'x': 1000 *np.diff(forces['x'], prepend=forces['x'][1]),
    'y': 1000* np.diff(forces['y'], prepend=forces['y'][1]),
    'z': 1000 * np.diff(forces['z'], prepend=forces['z'][1])
    }
    # low pass filter
    for i in range (0, len(force_diff['x'])-1):
        force_diff['x'][i+1] = force_diff['x'][i] * 0.99 + 0.01 * force_diff['x'][i+1]
        force_diff['y'][i+1] = force_diff['y'][i] * 0.99 + 0.01 * force_diff['y'][i+1]
        force_diff['z'][i+1] = force_diff['z'][i] * 0.99 + 0.01 * force_diff['z'][i+1]


    velocities = {
    'vx': 1000* np.diff(ee_positions['x'], prepend=ee_positions['x'][1]), # 1 to avoid jump from 0 to somewhere
    'vy': 1000 * np.diff(ee_positions['y'], prepend=ee_positions['y'][1]),
    'vz': 1000 * np.diff(ee_positions['z'], prepend=ee_positions['z'][1])
    }

    # Plot forces and their differences
    # Force X and diff
    axs[0, 0].plot(timestamps, forces['x'], 'r', label='Force X')
    axs[0, 0].plot(timestamps, force_diff['x'], 'b', label='Diff Force X')
    axs[0, 0].set_title('Force X and Diff')
    axs[0, 0].legend()
    axs[0, 0].set_xlim(left=10)

    # Force Y and diff
    axs[0, 1].plot(timestamps, forces['y'], 'r', label='Force Y')
    axs[0, 1].plot(timestamps, force_diff['y'], 'b', label='Diff Force Y')
    axs[0, 1].set_title('Force Y and Diff')
    axs[0, 1].legend()
    axs[0, 1].set_xlim(left=10)
    # Force Z and diff
    axs[0, 2].plot(timestamps, forces['z'], 'r', label='Force Z')
    axs[0, 2].plot(timestamps, force_diff['z'], 'b', label='Diff Force Z')
    axs[0, 2].set_title('Force Z and Diff')
    axs[0, 2].legend()
    axs[0, 2].set_xlim(left=10)
    # Plot forces
    #axs[0, 0].plot(timestamps, forces['x'], 'r', label='Force X')
    #axs[0, 1].plot(timestamps, forces['y'], 'r', label='Force Y')
    #axs[0, 2].plot(timestamps, forces['z'], 'r', label='Force Z')

    # Plot torques
    axs[1, 0].plot(timestamps, torques['x'], 'r', label='Torque X')
    axs[1, 1].plot(timestamps, torques['y'], 'r', label='Torque Y')
    axs[1, 2].plot(timestamps, torques['z'], 'r', label='Torque Z')
    # Plot reference positions and end-effector positions
    axs[2, 0].plot(timestamps, reference_positions['x'], 'r', label='Ref Position X')
    axs[2, 0].plot(timestamps, ee_positions['x'], 'b', label='EE Position X')
    axs[2, 1].plot(timestamps, reference_positions['y'], 'r', label='Ref Position Y')
    axs[2, 1].plot(timestamps, ee_positions['y'], 'b', label='EE Position Y')
    axs[2, 2].plot(timestamps, reference_positions['z'], 'r', label='Ref Position Z')
    axs[2, 2].plot(timestamps, ee_positions['z'], 'b', label='EE Position Z')
    
    # plot velocities
    axs[3, 0].plot(timestamps, velocities['vx'], 'g', label='EE Velocity X')
    axs[3, 1].plot(timestamps, velocities['vy'], 'b', label='EE Velocity Y')
    axs[3, 2].plot(timestamps, velocities['vz'], 'b', label='EE Velocity Z')
    axs[3, 0].set_xlim(left=10)
    axs[3, 1].set_xlim(left=10)
    axs[3, 2].set_xlim(left=10)
    axs[3, 0].set_ylim([-0.4, 0.4])
    axs[3, 1].set_ylim([-0.4, 0.4])
    axs[3, 2].set_ylim([-0.4, 0.4])


    # Plot Euler angles and end-effector orientations
    axs[4, 0].plot(timestamps, euler_angles['roll'], 'r', label='Ref Roll')
    axs[4, 0].plot(timestamps, ee_orientations['roll'], 'b', label='EE Roll')
    axs[4, 1].plot(timestamps, euler_angles['pitch'], 'r', label='Ref Pitch')
    axs[4, 1].plot(timestamps, ee_orientations['pitch'], 'b', label='EE Pitch')
    axs[4, 2].plot(timestamps, euler_angles['yaw'], 'r', label='Ref Yaw')
    axs[4, 2].plot(timestamps, ee_orientations['yaw'], 'b', label='EE Yaw')

    # Set titles for plots
    axs[0, 0].set_title('Force X')
    axs[0, 1].set_title('Force Y')
    axs[0, 2].set_title('Force Z')
    axs[1, 0].set_title('Torque X')
    axs[1, 1].set_title('Torque Y')
    axs[1, 2].set_title('Torque Z')
    # axs[2, 0].set_title('Position X')
    # axs[2, 1].set_title('Position Y')
    # axs[2, 2].set_title('Position Z')

    """     
    axs[3, 0].set_title('Orientation Roll')
    axs[3, 1].set_title('Orientation Pitch')
    axs[3, 2].set_title('Orientation Yaw') 
    """

    # Add labels and legends
    for ax in axs.flat:
        ax.set(xlabel='Timestamp', ylabel='Value')
        ax.legend()

    """    
    # Synchronize zoom/pan events
    for ax in axs.flat:
        other_axes = [other_ax for other_ax in axs.flat if other_ax != ax]
        ax.callbacks.connect('xlim_changed', lambda event_ax: on_xlim_changed(event_ax, other_axes))
        ax.callbacks.connect('ylim_changed', lambda event_ax: on_ylim_changed(event_ax, other_axes))
    """

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25)                # increased the hight spacing in between the subplots

    # Show plot
    plt.show(block=False)

def plot_force_fft(forces, sampling_rate):
    """
    Plot the FFT of the force data.

    Parameters:
    - forces (dict): Dictionary of forces (x, y, z).
    - sampling_rate (int): Sampling rate in Hz.
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))

    for i, axis in enumerate(['x', 'y', 'z']):
        force = forces[axis]
        n = len(force)
        freq = np.fft.fftfreq(n, d=1/sampling_rate)
        fft = np.fft.fft(force)

        axs[i].plot(freq[:n//2], np.abs(fft)[:n//2])
        axs[i].set_title(f'FFT of Force {axis.upper()}')
        axs[i].set_xlabel('Frequency (Hz)')
        axs[i].set_ylabel('Magnitude')

    plt.tight_layout()
    plt.show(block=False)

def plot_force_vs_position(timestamps, forces, positions):
    baseline = 0.2303  # Baseline position (z-axis)

    # Subtract baseline from each position using list comprehension
    normalized_positions = [p - baseline for p in positions['z']]

    plt.figure(figsize=(10, 6))
    plt.plot(normalized_positions, forces['z'], 'r-', label='Force vs. Position')
    plt.title('Force as a Function of Position')
    plt.xlabel('Position Z (mm)')
    plt.ylabel('Force Z (N)')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

def plot_delta_force_vs_position(delta_x_z, delta_force_z):
    plt.figure(figsize=(10, 6))
    plt.scatter(delta_x_z, delta_force_z, color='b', label='Delta Force vs. Delta Position')
    plt.title('Change in Force vs. Change in Position over 30ms')
    plt.xlabel('Delta Position (mm)')
    plt.ylabel('Delta Force (N)')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

def plot_force_vs_delta_displacement(delta_displacement, smoothed_forces_z):
    """
    Plot Force vs Delta Displacement.
    
    Parameters:
    - delta_displacement (list or np.array): The delta displacement data (from np.diff()).
    - smoothed_forces_z (list or np.array): The smoothed forces data for the z-axis.
    """
    # Ensure both arrays have the same length by truncating the forces array
    smoothed_forces_z = smoothed_forces_z[:-1]  # Truncate the last element of smoothed forces to match delta_displacement
    plt.figure(figsize=(10, 6))
    plt.plot(delta_displacement, smoothed_forces_z, marker='o', label="Force-Displacement Curve")
    plt.xlabel("Displacement (mm)")
    plt.ylabel("Force (N)")
    plt.title("Force vs Displacement (Cumulative)")
    plt.grid(True)
    plt.legend()
    plt.show(block=False)

def normalize_positions(positions, baseline):
    """ Normalize positions by a baseline value. """
    return {key: [p - baseline for p in pos] for key, pos in positions.items()}


def butter_lowpass_filter(data, low, high, fs, order):
    """
    Apply a Butterworth low-pass filter to the data.

    Parameters:
    - data (dict): Dictionary of data to filter.
    - cutoff (float): Cutoff frequency in Hz.
    - fs (int): Sampling rate in Hz.
    - order (int): Order of the filter.

    Returns:
    - Dictionary of filtered data.
    """
    filtered_data = {}
    for key in data:
        b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype='band')
        filtered_data[key] = filtfilt(b, a, data[key])
    return filtered_data

def calculate_phase_distortion(low, high, fs, order):
    # Design a Butterworth bandpass filter
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype='band')
    
    # Compute the frequency response of the filter
    # worN=8000 specifies the number of frequency points to compute.
    # 
    w, h = freqz(b, a, worN=8000)
    
    # Calculate the phase response of the filter
    # np.angle(h) returns the phase angles in radians
    # np.unwrap removes discontinuities in the phase response
    phase_response = np.unwrap(np.angle(h))
    
    return w, phase_response

def plot_phase_distortion(w, phase_response, fs):
    plt.figure(figsize=(10, 6))
    plt.plot(w * fs / (2 * np.pi), phase_response, 'b')
    plt.title('Phase Response of the Butterworth Filter')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')
    plt.grid(True)
    plt.legend()
    plt.show(block=False)


if __name__ == "__main__":
    # Path to your JSON log file
    logfile = 'robot_state_log_2024_11_11_1120.json'

    # Load and process the log file
    data = load_log_file(logfile)
    timestamps, forces, torques, reference_positions, euler_angles, ee_positions, ee_orientations = extract_data(data)

    sampling_rate = 500  # 1000 Hz update rate

    plot_force_fft(forces, sampling_rate)

    # Define the time step delta_t
    delta_t = 1.0 / 500.0  # Assuming 1000Hz update rate

    smoothed_position = apply_ema(ee_positions, alpha=0.005)

    # Calculate velocities and accelerations
    accelerations = calculate_accelerations(ee_positions, delta_t)
    
    # Initialize smoothed_forces to be the same as forces before applying moving average
    smoothed_forces = forces.copy()  # Ensure this is a deep copy if forces contains mutable objects


    # Apply moving average filter to the force data over 20 points
    # window_size = 40
    # for axis in ['x', 'y', 'z']:
    #     smoothed_forces[axis] = moving_average(forces[axis], window_size)

    # Apply Butterworth filter to the force data
    smoothed_forces = butter_lowpass_filter(forces, low=2, high=8, fs=sampling_rate, order=2)
    
    # Calculate phase distortion
    w, phase_response = calculate_phase_distortion(low=2, high=8, fs=sampling_rate, order=2)

    # Plot phase distortion
    plot_phase_distortion(w, phase_response, sampling_rate)

    # Truncate all data arrays to the shortest length (due to velocity/acceleration calculation)
    timestamps, forces, torques, reference_positions, euler_angles, ee_positions, ee_orientations, accelerations = truncate_to_shortest_length(
        timestamps, smoothed_forces, torques, reference_positions, euler_angles, ee_positions, ee_orientations, accelerations
    )

    # Truncate smoothed forces to match the truncated timestamps length
    for key in smoothed_forces:
        smoothed_forces[key] = smoothed_forces[key][:len(timestamps)]

    print(accelerations['x'])

    # Now plot the data
    plot_data(timestamps, smoothed_forces, torques, reference_positions, euler_angles, ee_positions, ee_orientations, accelerations)
    
    # Extract and truncate forces and positions for force vs. position plot
    truncated_timestamps, truncated_forces_z, truncated_positions_z = truncate_to_shortest_length(
        timestamps, smoothed_forces['z'], ee_positions['z']
    )
    #plot_force_vs_position(truncated_timestamps, {'z': truncated_forces_z}, {'z': truncated_positions_z})

    # Plot deltas
    sampling_rate = 500  # 1000 Hz update rate
    delta_force_z = compute_deltas(smoothed_forces['z'], 30, sampling_rate)
    delta_x_z = compute_deltas(ee_positions['z'], 30, sampling_rate)
    timestamps, delta_force_z, delta_x_z = truncate_to_shortest_length(timestamps, delta_force_z, delta_x_z)
    #plot_delta_force_vs_position(delta_x_z, delta_force_z)
    

    # Initialize smoothed_forces to be the same as forces before applying moving average
    smoothed_displacement = ee_positions.copy()  # Ensure this is a deep copy if forces contains mutable objects

    # Apply moving average filter to the force data over 40 points
    window_size = 40
    for axis in ['x', 'y', 'z']:
        smoothed_displacement[axis] = moving_average(ee_positions[axis], window_size)

    delta_displacement = abs(np.diff(smoothed_displacement['z'])) 
    #plot_force_vs_delta_displacement(delta_displacement, smoothed_forces['z'])

# Ensure all plots stay open
plt.show()
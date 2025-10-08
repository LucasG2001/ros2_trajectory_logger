import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
from scipy.spatial.transform import Rotation as R


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

def extract_data_gp(data):
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

    timestamps = list(range(len(data)))

    forces = {'x': [], 'y': [], 'z': []}
    torques = {'x': [], 'y': [], 'z': []}
    reference_positions = {'x': [], 'y': [], 'z': []}
    euler_angles = {'roll': [], 'pitch': [], 'yaw': []}
    ee_positions = {'x': [], 'y': [], 'z': []}
    ee_orientations = {'roll': [], 'pitch': [], 'yaw': []}
    accelerations = {'x': [], 'y': [], 'z': []}
    velocity_desired = []
    gp_means = []
    gp_lower_bounds = []
    gp_upper_bounds = []
    trigger_values = []


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

        # Extract velocity desired
        velocity_desired.append(entry['velocity_desired'])

        # Extract gp data
        gp_means.append(entry['means'])
        gp_lower_bounds.append(entry['lower_bounds'])
        gp_upper_bounds.append(entry['upper_bounds'])

        # extract trigger values
        trigger_values.append(entry['trigger_values'])

    return timestamps, forces, torques, reference_positions, euler_angles, ee_positions, ee_orientations, velocity_desired, gp_means, gp_lower_bounds, gp_upper_bounds, trigger_values

if __name__ == "__main__":
    # Path to your JSON log file
    logfile = '/home/lucas/franka_ros2_ws/src/ros2_trajectory_logger/robot_state_log_2025_04_16_1436.json'
    
    # Load and process the log file
    data = load_log_file(logfile)
    timestamps, forces, torques, reference_positions, euler_angles, ee_positions, ee_orientations, velocity_desired, gp_means, gp_lower_bounds, gp_upper_bounds, trigger_values = extract_data_gp(data)

    fig, ax = plt.subplots(figsize=(18, 6))  # Just one subplot now

    # Plot velocity desired
    ax.plot(timestamps, velocity_desired, label="velocity", color='blue')

    # Plot mean
    ax.plot(timestamps, gp_means, label="mean", color='green')

    # Plot lower and upper bounds
    ax.plot(timestamps, gp_lower_bounds, label="lower bound", color='red')
    ax.plot(timestamps, gp_upper_bounds, label="upper bound", color='orange')

    # Plot trigger values
    ax.plot(timestamps, trigger_values, label="trigger values", color='purple')

    # Labels, legend, grid
    ax.set_xlabel("Timestamps")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()
import json
import matplotlib.pyplot as plt
import numpy as np
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
    timestamps = range(len(data))  # Assuming each entry represents a time step

    forces = {'x': [], 'y': [], 'z': []}
    torques = {'x': [], 'y': [], 'z': []}
    reference_positions = {'x': [], 'y': [], 'z': []}
    euler_angles = {'roll': [], 'pitch': [], 'yaw': []}
    ee_positions = {'x': [], 'y': [], 'z': []}
    ee_orientations = {'roll': [], 'pitch': [], 'yaw': []}

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

def plot_data(timestamps, forces, torques, reference_positions, euler_angles, ee_positions, ee_orientations):
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
    fig, axs = plt.subplots(6, 3, figsize=(18, 18))

    # Calculate differences (diff of forces)
    force_diff = {
    'x': np.diff(forces['x'], prepend=forces['x'][0]),
    'y': np.diff(forces['y'], prepend=forces['y'][0]),
    'z': np.diff(forces['z'], prepend=forces['z'][0])
    }

    # Plot forces and their differences
    # Force X and diff
    axs[0, 0].plot(timestamps, forces['x'], 'r', label='Force X')
    axs[0, 0].plot(timestamps, force_diff['x'], 'b', label='Diff Force X')
    axs[0, 0].set_title('Force X and Diff')
    axs[0, 0].legend()

    # Force Y and diff
    axs[0, 1].plot(timestamps, forces['y'], 'r', label='Force Y')
    axs[0, 1].plot(timestamps, force_diff['y'], 'b', label='Diff Force Y')
    axs[0, 1].set_title('Force Y and Diff')
    axs[0, 1].legend()

    # Force Z and diff
    axs[0, 2].plot(timestamps, forces['z'], 'r', label='Force Z')
    axs[0, 2].plot(timestamps, force_diff['z'], 'b', label='Diff Force Z')
    axs[0, 2].set_title('Force Z and Diff')
    axs[0, 2].legend()

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

    # Plot Euler angles and end-effector orientations
    axs[3, 0].plot(timestamps, euler_angles['roll'], 'r', label='Ref Roll')
    axs[3, 0].plot(timestamps, ee_orientations['roll'], 'b', label='EE Roll')
    axs[3, 1].plot(timestamps, euler_angles['pitch'], 'r', label='Ref Pitch')
    axs[3, 1].plot(timestamps, ee_orientations['pitch'], 'b', label='EE Pitch')
    axs[3, 2].plot(timestamps, euler_angles['yaw'], 'r', label='Ref Yaw')
    axs[3, 2].plot(timestamps, ee_orientations['yaw'], 'b', label='EE Yaw')

    # Set titles for plots
    axs[0, 0].set_title('Force X')
    axs[0, 1].set_title('Force Y')
    axs[0, 2].set_title('Force Z')
    axs[1, 0].set_title('Torque X')
    axs[1, 1].set_title('Torque Y')
    axs[1, 2].set_title('Torque Z')
    axs[2, 0].set_title('Position X')
    axs[2, 1].set_title('Position Y')
    axs[2, 2].set_title('Position Z')
    axs[3, 0].set_title('Orientation Roll')
    axs[3, 1].set_title('Orientation Pitch')
    axs[3, 2].set_title('Orientation Yaw')

    # Add labels and legends
    for ax in axs.flat:
        ax.set(xlabel='Timestamp', ylabel='Value')
        ax.legend()

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()

if __name__ == "__main__":
    # Path to your JSON log file
    logfile = 'robot_state_log_2024_10_22_1528.json'

    # Load and process the log file
    data = load_log_file(logfile)
    timestamps, forces, torques, reference_positions, euler_angles, ee_positions, ee_orientations = extract_data(data)

    # Plot the data
    plot_data(timestamps, forces, torques, reference_positions, euler_angles, ee_positions, ee_orientations)

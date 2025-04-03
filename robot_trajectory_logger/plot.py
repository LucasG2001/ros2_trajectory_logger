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

    timestamps = list(range(len(data)))

    forces = {'x': [], 'y': [], 'z': []}
    torques = {'x': [], 'y': [], 'z': []}
    reference_positions = {'x': [], 'y': [], 'z': []}
    euler_angles = {'roll': [], 'pitch': [], 'yaw': []}
    ee_positions = {'x': [], 'y': [], 'z': []}
    ee_orientations = {'roll': [], 'pitch': [], 'yaw': []}
    accelerations = {'x': [], 'y': [], 'z': []}
    joint_velocities = [] 
    jacobianEE = []
    dtjacobianEE = []
    dtFext_desired = []
    f_ext_desired = []
    velocity_desired = []

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

        # Extract dtFext_desired, prjection on the correct axis happens in the .cpp file of cartesian impedance controller
        dtFext_desired.append(entry['dt_Fext_desired'])

        # Extract f_ext_desired
        f_ext_desired.append(entry['f_ext_desired'])

        # Extract velocity desired
        velocity_desired.append(entry['velocity_desired'])

    return timestamps, forces, torques, reference_positions, euler_angles, ee_positions, ee_orientations, dtFext_desired, f_ext_desired, velocity_desired

def butter_band_filter(data, high,low, fs, order):
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
    
    b, a = butter(order, [high / (fs / 2), low / (fs / 2)], btype='band')
    filtered_data = filtfilt(b, a, data)

    return filtered_data

def plot_force_fft(data, sampling_rate):
    """
    Plot the FFT of the force data.

    Parameters:
    - data (list or np.array): Data to perform FFT on.
    - sampling_rate (int): Sampling rate in Hz.
    """
    plt.figure(figsize=(10, 6))
        
    n = len(data)
    freq = np.fft.fftfreq(n, d=1/sampling_rate)
    fft = np.fft.fft(data)

    plt.plot(freq[:n//2], np.abs(fft)[:n//2])
    plt.title('FFT of k_z')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Path to your JSON log file
    logfile = '/home/nilsjohnson/franka_ros2_ws/src/ros2_trajectory_logger/robot_state_log_2025_02_28_1404.json'
    
    # Load and process the log file
    data = load_log_file(logfile)
    timestamps, forces, torques, reference_positions, euler_angles, ee_positions, ee_orientations, dtFext_desired, f_ext_desired, velocity_desired, displacement_desired = extract_data(data)

    sampling_rate = 1000  # 1000 Hz update rate

    # Convert positions to numpy arrays
    ee_positions_x = np.array(ee_positions['x'])
    ee_positions_y = np.array(ee_positions['y'])
    ee_positions_z = np.array(ee_positions['z'])

    # Convert Euler angles to Rotation objects
    rotations = R.from_euler('xyz', np.column_stack((ee_orientations['roll'],
                                                    ee_orientations['pitch'],
                                                    ee_orientations['yaw'])), degrees=False)

    # Reference orientation at index 1000
    ref_rotation = rotations[0]

    # Compute relative rotations
    relative_rotations = ref_rotation.inv() * rotations

    # Get angle (in radians) of each relative rotation
    orientation_error = relative_rotations.magnitude()

    # convert to degrees
    orientation_error = np.degrees(orientation_error)

    # Compute total displacement from the first position
    displacement_total = np.sqrt(
    (ee_positions_x - ee_positions_x[0])**2 + 
    (ee_positions_y - ee_positions_y[0])**2 + 
    (ee_positions_z - ee_positions_z[0])**2
    )

    fig, axs = plt.subplots(5, 1, figsize=(18, 18), sharex=True)

    # Plot the displacement (Z-axis)
    axs[0].plot(timestamps, displacement_total, label="Displacement", color='blue')
    axs[0].set_xlabel("Timestamps")
    axs[0].set_ylabel("Displacement [mm]")
    axs[0].legend()
    axs[0].grid(True)

    # Plot the filtered force (desired-axis)
    axs[1].plot(timestamps, f_ext_desired, label="Force", color='green')
    axs[1].set_xlabel("Timestamps")
    axs[1].set_ylabel("Force [N]")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(timestamps, dtFext_desired, label="dt_Force", color='red')
    axs[2].set_xlabel("Timestamps")
    axs[2].set_ylabel("dt_Force [N/s]")
    axs[2].legend()
    axs[2].grid(True)

    axs[3].plot(timestamps, velocity_desired, label="vel", color='orange')
    axs[3].set_xlabel("Timestamps")
    axs[3].set_ylabel("Velocity [m/s]")
    axs[3].legend()
    axs[3].grid(True)

    axs[4].plot(timestamps, orientation_error, label="orientation error", color='orange')
    axs[4].set_xlabel("Timestamps")
    axs[4].set_ylabel("Orientation Error")
    axs[4].legend()
    axs[4].grid(True)
    

    """     difference_array = []

    desired_value = -4.5

    # subtract the deisred value from all values in the dt_Fext_z array and save the new values in the difference_array
    for value in f_ext_desired:
        difference_array.append(abs(value - desired_value))
    
    # find the index of the second smallest value in the difference_array
    index = np.argsort(difference_array)[2]

    # at my index value set a red cross in the plot
    axs[1].plot(index, f_ext_desired[index], 'rx')

    # I want to draw a vertical dotted line at the index value in all plots
    for ax in axs:
        ax.axvline(index, color='k', linestyle='--') 

    print(dtFext_desired[index])"""
    

    plt.tight_layout()
    plt.show()


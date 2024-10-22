import json

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression



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



def extract_data(data, start = 2000, end = 4700):

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

    # Limit data to the specified range

    limited_data = data[start:end]

    timestamps = list(range(start, end))



    forces = {'x': [], 'y': [], 'z': []}

    torques = {'x': [], 'y': [], 'z': []}

    reference_positions = {'x': [], 'y': [], 'z': []}

    euler_angles = {'roll': [], 'pitch': [], 'yaw': []}

    ee_positions = {'x': [], 'y': [], 'z': []}

    ee_orientations = {'roll': [], 'pitch': [], 'yaw': []}

    accelerations = {'x': [], 'y': [], 'z': []}



    for entry in limited_data:

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



def perform_linear_regression(x, F_ext):

    """

    Performs linear regression to estimate F_h and k for each axis.

    

    Parameters:

    - x (np.array): Displacement data (independent variable).

    - F_ext (np.array): External force data (dependent variable).

    

    Returns:

    - F_h (float): The estimated value of F_h for the axis.

    - k (float): The estimated value of k (stiffness) for the axis.

    """

    # Prepare the independent variable matrix: [1, -x]

    X = np.vstack([np.ones(len(x)), -x]).T  # Transpose to match the shape (n_samples, 2)

    

    # Perform linear regression

    model = LinearRegression(fit_intercept=False)  # No intercept because we are already handling F_h

    model.fit(X, F_ext)

    

    # The coefficients of the model are F_h and k

    #F_h, k = model.coef_

    # Extract the slope (k) and intercept (a)
    k = model.coef_[0]
    F_h = model.intercept_

    return F_h, k



def sliding_window_regression(displacement, force, window_size=40, step_size=40):

    """

    Applies linear regression on each sliding window of data.

    

    Parameters:

    - displacement (np.array): Displacement data for an axis.

    - force (np.array): Force data for an axis.

    - window_size (int): Number of data points to use for each regression (e.g., 40).

    - step_size (int): Step size to move the window forward (e.g., 40 for non-overlapping windows).

    

    Returns:

    - F_h_list (list): List of F_h estimates for each window.

    - k_list (list): List of k (stiffness) estimates for each window.

    """

    F_h_list = []

    k_list = []

    

    # Slide through the data with a window of size `window_size`

    for start in range(0, len(displacement) - window_size + 1, step_size):

        # Extract the window of data

        x_window = displacement[start:start + window_size]

        F_ext_window = force[start:start + window_size]

        

        # Perform linear regression on the window

        F_h, k = perform_linear_regression(x_window, F_ext_window)

        

        # Store the results

        F_h_list.append(F_h)

        k_list.append(k)

    

    return F_h_list, k_list





if __name__ == "__main__":

    # Path to your JSON log file

    logfile = 'robot_state_log_2024_10_22_1523.json'

    

    # Load and process the log file

    data = load_log_file(logfile)

    timestamps, forces, torques, reference_positions, euler_angles, ee_positions, ee_orientations = extract_data(data)



    # Convert lists to numpy arrays for further processing

    force_z = np.array(forces['z'])

    displacement_z = np.array(ee_positions['z'])



    # Define window size (40 measurements) and step size (you can use 40 for non-overlapping)

    window_size = 40  # Number of measurements per window

    step_size = 40    # Move 40 points at a time (non-overlapping)



    # Perform sliding window linear regression for Z axis

    F_h_z_list, k_z_list = sliding_window_regression(displacement_z, -force_z, window_size=window_size, step_size=step_size)



    # Print results

    print(f"Estimated F_h values for each window (Z-axis): {F_h_z_list}")

    print(f"Estimated k values (stiffness) for each window (Z-axis): {k_z_list}")



    # Plot results

    plt.figure(figsize=(10, 6))

    plt.plot(F_h_z_list, label="F_h (Z-axis)")

    plt.plot(k_z_list, label="k (stiffness Z-axis)")

    plt.xlabel("Window Index")

    plt.ylabel("Estimated Values")

    plt.legend()

    plt.grid(True)

    plt.show()
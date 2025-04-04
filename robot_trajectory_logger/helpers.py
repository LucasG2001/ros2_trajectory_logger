import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

"""
Helpers for miscellaneous functions.
"""

def update_plot(val, slider, spectrum_line, y, t, ax, fig):
    """
    Update function for the interactive slider.

    Parameters:
    - val: Current slider value.
    - slider: The slider widget.
    - spectrum_line: The plot line object.
    - y: The y-data array.
    - t: The time vector. (or x)
    - ax: The plot axis.
    - fig: The figure object.
    """
    time_idx = int(slider.val)
    spectrum_line.set_ydata(abs(y[:, time_idx]))  # Update Y-axis data
    spectrum_line.set_label(f"Spectrum at t = {t[time_idx]:.2f}s")  # Update label
    ax.legend()
    fig.canvas.draw_idle()  # Redraw the figure

    
# Step 1: Load data from JSON file
def load_data(json_file_path, sampling_freq = 500, time_window = 5, plot=False):
    """
    Extracts force magnitudes and linear positions from a JSON file.
    Calculates the velocity norms based on the sampling time.
    
    Args:
        json_file_path (str): Path to the JSON file.
        sampling_time (float): Sampling time in seconds (default: 1/500).
        time_window (int): Number of seconds to consider (default: 5).        
    Returns:
        f_magnitude (numpy.ndarray): Magnitudes of forces extracted.
        positions (numpy.ndarray): Linear positions extracted.
        velocities (numpy.ndarray): Norms of velocities calculated.
    """
    T = 1/ sampling_freq
    indices = time_window * sampling_freq
    f_magnitude = []
    displacement = []
    positions = []
    # Read the JSON file line by line
    with open(json_file_path, 'r') as file:
        for line in file:
            # Parse each line as a JSON object
            data = json.loads(line)
            
            # Extract force components
            force = data["f_ext"]["force"]
            f_magnitude.append(force["z"])
            
            # Extract linear displacements
            position = data["ee_pose"]["position"]
            positions.append([position["x"], position["y"], position["z"]])
            displacement.append(np.sqrt([position["x"]**2 + position["y"]**2 + position["z"]**2]))
    
    # Convert to numpy arrays
    f_magnitude = np.array(f_magnitude)
    displacement = displacement - displacement[0]  # Set initial displacement to zero
    # Step 2: Calculate velocities as norms of position differences
    # Velocity = norm(pos[k+1] - pos[k]) / sampling_time
    position_differences = np.diff(positions, axis=0)  # Differences between consecutive positions
    velocities = np.concatenate(([0], np.linalg.norm(position_differences, axis=1) / T))  # Norms divided by sampling time

    return f_magnitude[0: indices], displacement[0: indices], velocities[0: indices] * 10
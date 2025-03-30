# ros2_trajectory_logger

This repository is used to gather data from the Franka Emika FR3 during usage. This branch is set up to gather data from the drill controller and automatically start and stop log files with the foot pedal. The log files will be saved in the directory where the `ros2 run robot_trajectory_logger robot_trajectory_logger` command is executed. 

## log_data.py

This script is used to gather the inputs from the standard messages as well as custom messages from the `messages_fr3` repository and create the log file. If you want to gather new signals from the FR3 during operation, you can modify this script accordingly. The architecture of message subscription and data gathering over callback can be copied from the preexisting examples. The log file will be automatically started and stopped via foot pedal actuation which sends a message over the PlannerService and is handled by the handle_service function. Stopping and resarting a logfile within the same minute will not work as the log file names will then be the same. The current timestamp in the name is set to a precision of up to a minute.

## plot.py

This script is used to plot the data gathered by `log_data.py`. It provides visual representations of the logged data with additional band-pass filter and fft transform function integrated for enhanced data analysis. Make sure to adapt the sampling rate int he main function if it is changed in the `log_data.py` script. The following varibales represent the most important data from the drill controller:

- `ee_positions`: end-effector position in x,y,z
- `ee-orientation`: end-effector roll,pitch and yaw
- `dt_Fext_desired`: derivative of resultant force acting on end-effector in desired drilling direction
- `f_ext_desired`: resultant force acting on end-effector in desired drilling direction
- `velocity_desired`: velocity in desired drilling direction

When adding new signals the current architecture can be used from the extract_data function and adapted accordingly if signal names are changed.

## linear_regression.py

This script can be used to create a linear_regression depending on the desired inputs.

# ros2_trajectory_logger

This repository is used to gather data from the Franka Emika FR3 during usage. This branch is set up to gather data from the drill controller and automatically start and stop log files with the foot pedal.

## log_data.py

This script is used to gather the inputs from the standard messages as well as custom messages from the `messages_fr3` repository and create the log file. If you want to gather new signals from the FR3 during operation, you can modify this script accordingly.

## plot.py

This script is used to plot the data gathered by `log_data.py`. It provides visual representations of the logged data for analysis.

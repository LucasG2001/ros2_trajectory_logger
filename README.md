# ros2_trajectory_logger
This repo is used to gather data from the franka emika FR3 druing usage. This branch is setup to gather data from the drill controller as well as automatically start and stop log files with the foot pedal actuation. In the setup process of the drill_control repo you should have already cloned this repo. There you will find all the steps needed to start the logger during the drilling process. The sections below will give a better understanding of the individual scripts as well as the signals used in them.

## log_data.py
This script is used to gather the inputs from the standard messages as well as custom messages from the messages_fr3 repo and create the logfile. If you want to gather new signals from the FR3 during usage then these messages need to be subscribed to and the data gathered with a callback in this script. The architecture of subscribing to a message and creating a callback can be copied from the already existing messages and callbacks. The logfiles will automatically start and stop with the foot pedal actuation as this sends a message over the PlannerService which is handeld in the handle_service function.

## plot.py
This script is used to plot the data gathered in the log_data.py

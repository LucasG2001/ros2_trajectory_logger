import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import os
from messages_fr3.srv import PlannerService
from robot_trajectory_logger.SpikeDetector import SpikeDetector

"""
This node loads the logfiles from the bone drilling experiments and sequentially sends them to ROS2 to test functionality with other nodes
"""

class DataStreamer(Node):
    def __init__(self, file_path, sampling_frequency, cutoff_time, passband):
        super().__init__('robot_trajectory_logger')

        # Pose publishers
        self.displacement_publisher = self.create_publisher(Float64, '/displacement_value', 1)       
        self.f_ext_publisher = self.create_publisher(Float64, '/f_ext_desired', 1)
        self.velocity_publisher = self.create_publisher(Float64, '/velocity_desired', 1)

        # Initialize spike detector with passed arguments
        self.spike_detector = SpikeDetector(
            logfile=file_path,
            fs=sampling_frequency,
            time_window=cutoff_time,
            passband=passband
        )

        # Timer
        self.send_data_timer = self.create_timer(1.0 / 1000.0, self.send_data)
        # counter 
        self.counter = 0

        # send 'a' command over PlannerService
        self.planner_service_client = self.create_client(PlannerService, 'planner_service')

        # wait for the service to be available
        while not self.planner_service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Planner servie not available, waiting...')
        
        # create request and send the 'a' command
        request = PlannerService.Request()
        request.command = 'a'
        self.planner_service_client.call_async(request)
        self.get_logger().info('Sent command "a" to planner service')


    def send_data(self):
        if self.counter < self.spike_detector.logfile_length:
            i = self.counter
            self.displacement_publisher.publish(Float64(data=self.spike_detector.displacement[i]))
            self.f_ext_publisher.publish(Float64(data=self.spike_detector.drilling_force[i]))
            self.velocity_publisher.publish(Float64(data=self.spike_detector.velocities[i]))

            #print(f"Published data at index {i}")
            self.counter += 1
            # print the data every 1000ms
            if self.counter % 1000 == 0:
                # print displacement
                self.get_logger().info(f"Displacement: {self.spike_detector.displacement[i]}")
        
        else:
            i = self.counter
            #self.get_logger().info("End of data stream reached.")
        

    
def main(args=None):
    rclpy.init(args=args)
    folder_path = "/home/nilsjohnson/franka_ros2_ws/src/ros2_trajectory_logger/Logs"
    sampling_frequency = 1000    
    cutoff_time = 10.5
    passband = (5, 50)
    # file_paths = []
    # for filename in os.listdir(folder_path):
    #     file_path = os.path.join(folder_path, filename)
    #     if os.path.isfile(file_path):  # Ensure it's a file
    #         #print(f"adding file: {file_path}")
    #         # hard coded file paths for testing
    #         file_path = "/home/nilsjohnson/franka_ros2_ws/src/ros2_trajectory_logger/Logs/robot_state_log_2025_02_28_1402.json"
    #         file_paths.append(file_path)
    #         #print(f"File paths: {file_paths[0]}")

    file_path = "/home/nilsjohnson/franka_ros2_ws/src/ros2_trajectory_logger/Logs/robot_state_log_2025_02_28_1011.json"
    # show the file path name
    node = DataStreamer(file_path, sampling_frequency, cutoff_time, passband)
    rclpy.spin(node)
    node.destroy_node() 
    rclpy.shutdown()


if __name__ == '__main__':
    main()
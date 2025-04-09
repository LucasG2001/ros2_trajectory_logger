from SpikeDetector import SpikeDetector
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import os

"""
This node loads the logfiles from the bone drilling experiments and sequentially sends them to ROS2 to test functionality with other nodes
"""

class DataStreamer(Node):
    def __init__(self, file_path, sampling_frequency, cutoff_time, passband):
        super().__init__('robot_trajectory_logger')

        # Pose publishers
        self.displacement_publisher = self.create_publisher(Float64, '/data_streamer/displacement', 1)       
        self.f_ext_publisher = self.create_publisher(Float64, '/f_ext_desired', 1)
        self.velocity_publisher = self.create_publisher(Float64, '/velocity_desired', 1)

        # Initialize spike detector with passed arguments
        self.spike_detector = SpikeDetector(
            file_path=file_path,
            fs=sampling_frequency,
            time_window=cutoff_time,
            passband=passband
        )

        # Timer
        self.send_data_timer = self.create_timer(1.0 / 1000.0, self.send_data)
        # counter 
        self.counter = 0

    def send_data(self):
        i = self.counter
        self.displacement_publisher.publish(Float64(data=self.spike_detector.displacement[i]))
        self.f_ext_publisher.publish(Float64(data=self.spike_detector.drilling_force[i]))
        self.velocity_publisher.publish(Float64(data=self.spike_detector.velocities[i]))

    
def main(args=None):
    rclpy.init(args=args)
    folder_path = "Logs"
    sampling_frequency = 1000    
    cutoff_time = 10.5
    passband = (5, 50)
    file_paths = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):  # Ensure it's a file
            print(f"adding file: {file_path}")
            file_paths.append(file_path)

    node = DataStreamer(file_paths[0], sampling_frequency, cutoff_time, passband)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
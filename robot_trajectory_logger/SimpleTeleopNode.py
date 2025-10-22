import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool, Int16
from franka_msgs.msg import FrankaRobotState
from rclpy.action import ActionClient
from franka_msgs.action import Grasp, Move
from geometry_msgs.msg import Wrench
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import csv
import numpy as np
from robot_trajectory_logger.wiggle_ee import wiggle_pose

class SimpleTeleopNode(Node):

    def __init__(self):
        super().__init__('simple_teleop_node')

        # Publishers
        self.cartesian_pub = self.create_publisher(
            Pose,
            '/cartesian_impedance_controller/reference_pose',
            1
        )
        self.float_mode_pub = self.create_publisher(
            Int16,
            '/cartesian_impedance_controller/control_mode',
            1
        )

        # Subscribers
        self.pose_subscriber = self.create_subscription(
            FrankaRobotState,
            '/franka_robot_state_broadcaster/robot_state',
            self.robot_state_callback,
            1)

        self.get_logger().info("Simple teleop node started.")

        # Initialize states
        self.ee_pose = Pose()

    def robot_state_callback(self, msg: FrankaRobotState):
        print("Received robot state message.")
        self.f_ext = msg._o_f_ext_hat_k._wrench
        self.ee_pose.position = msg.o_t_ee._pose._position
        self.ee_pose.orientation = msg.o_t_ee._pose._orientation
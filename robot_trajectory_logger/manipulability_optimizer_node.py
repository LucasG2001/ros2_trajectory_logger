#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Vector3
from your_package.msg import GoalPose, JointConfig  # Replace 'your_package' with your ROS package name
from ManipulabilityOptimizer import ManipulabilityMaximizer  # Ensure RobotOptimizer class is in your Python path

class RobotOptimizerNode:
    def __init__(self):
        # ROS Parameters and setup
        urdf_path = rospy.get_param("~urdf_path", "fr3.urdf")
        ee_frame_name = rospy.get_param("~ee_frame_name", "fr3_hand")
        self.joint_limits_lower = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159, -0.04, -0.04])
        self.joint_limits_upper = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159, 0.04, 0.04])
        self.q0 = np.array([0.33506416, 0.19438218, -0.34938642, -2.4187224, 1.59775894, 1.5097755, -1.34924279, 0., 0.])
        
        # Set up ROS subscriber and publisher
        rospy.Subscriber("goal_pose", GoalPose, self.goal_pose_callback)
        self.joint_config_pub = rospy.Publisher("optimal_joint_config", JointConfig, queue_size=10)

        # Initialize optimizer with default parameters (to be updated upon receiving goal pose)
        self.optimizer = None

    def goal_pose_callback(self, msg):
        # Parse the GoalPose message
        desired_translation = np.array([msg.translation.x, msg.translation.y, msg.translation.z])
        desired_rotation_euler = np.array([msg.rotation_euler.x, msg.rotation_euler.y, msg.rotation_euler.z])
        force_direction = np.array([0, 0, 1, 0, 0, 0])  # Example force direction, customize as needed

        # Initialize optimizer with new goal pose
        self.optimizer = RobotOptimizer("fr3.urdf", "fr3_hand", desired_translation, desired_rotation_euler, force_direction)

        # Run the optimization
        result = self.optimizer.optimize(self.q0, self.joint_limits_lower, self.joint_limits_upper)
        
        # Prepare and publish the result
        if result.success:
            optimal_joint_config_msg = JointConfig()
            optimal_joint_config_msg.joint_angles = result.x.tolist()
            self.joint_config_pub.publish(optimal_joint_config_msg)
            rospy.loginfo("Published optimal joint configuration.")
        else:
            rospy.logwarn("Optimization failed: " + result.message)

    def spin(self):
        rospy.spin()

if __name__ == "__main__":
    rospy.init_node("robot_optimizer_node")
    node = RobotOptimizerNode()
    node.spin()

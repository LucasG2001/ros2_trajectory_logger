#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool
from franka_msgs.msg import FrankaRobotState 
from geometry_msgs.msg import Pose, Wrench
import time

def quaternion_to_euler(quaternion):
        # Create a Rotation object from the quaternion
        rotation = R.from_quat(quaternion)
        # Convert to Euler angles (roll, pitch, yaw)
        euler_angles = rotation.as_euler('xyz', degrees=False)  # radians
        return euler_angles

class SimpleTeleopNode(Node):

    def __init__(self):
        super().__init__('simple_teleop_node')

        self.cartesian_pub = self.create_publisher(
            Pose,
            '/joint_impedance_controller/reference_pose',
            10
        )
        self.float_mode_pub = self.create_publisher(
            Bool,
            '/joint_impedance_controller/control_mode',
            10
        )
         # Subscribe to the robot state
        self.pose_subscriber = self.create_subscription(
            FrankaRobotState,  # Replace with the correct message type for franka_robot_state
            '/franka_robot_state_broadcaster/robot_state',
            self.robot_state_callback,
            10)
        self.current_pose = Pose()
        self.get_logger().info("Simple teleop node started.")

        # Initialize state and variables
        self.reference_pose = Pose()
        self.ee_pose = Pose()
        self.f_ext = Wrench()
        self.reference_pose.position.x = 0.38105
        self.reference_pose.position.y = -0.2179
        self.reference_pose.position.z = 0.09762
        self.reference_pose.orientation.w = 0.0
        self.reference_pose.orientation.x = 1.0
        self.reference_pose.orientation.y = 0.0
        self.reference_pose.orientation.z = 0.0
        self.ee_euler_angles = quaternion_to_euler([self.ee_pose.orientation._x, self.ee_pose.orientation._y, self.ee_pose.orientation._z, self.ee_pose.orientation._w])
        self.reference_euler_angles = quaternion_to_euler([self.reference_pose.orientation._x, self.reference_pose.orientation._y, self.reference_pose.orientation._z, self.reference_pose.orientation._w])

    

    def robot_state_callback(self, msg: FrankaRobotState):
        print("Received robot state message")
        self.f_ext = msg._o_f_ext_hat_k._wrench  # Assuming this is the correct attribute
        self.ee_pose.position = msg.o_t_ee._pose._position
        self.ee_pose.orientation = msg.o_t_ee._pose._orientation
        quaternion = [self.ee_pose.orientation._x, self.ee_pose.orientation._y, self.ee_pose.orientation._z, self.ee_pose.orientation._w]
        self.ee_euler_angles = quaternion_to_euler(quaternion)

        
    

    def run(self):
        while rclpy.ok():
            cmd = input(
                "\nEnter input (0: move to point, 1: free-float, 2: go to neutral, ENTER: print pose, s: stiff, x: exit): "
            )

            if cmd == "0":
                target = Pose()
                target.position.x = 0.46789
                target.position.y = -0.211576
                target.position.z = 0.0826885 + 0.03
                target.orientation.w = 0.0
                target.orientation.x = 1.0
                target.orientation.y = 0.0
                target.orientation.z = 0.0
                self.cartesian_pub.publish(target)
                self.get_logger().info("Sent predefined pose command.")

                time.sleep(2.5)  # Wait for the robot to reach the pose
                target.position.z -= 0.03  # Move back to original position
                self.cartesian_pub.publish(target)
                self.get_logger().info("Sent back to original pose command.")

            elif cmd == "1":
                msg = Bool()
                msg.data = True
                self.float_mode_pub.publish(msg)
                self.get_logger().info("Sent free-float True.")

            elif cmd == "2":
                target = Pose()
                target.position.x = 0.45
                target.position.y = -0.0
                target.position.z = 0.4
                target.orientation.w = 0.0
                target.orientation.x = 1.0
                target.orientation.y = 0.0
                target.orientation.z = 0.0
                self.cartesian_pub.publish(target)
                self.get_logger().info("Sent neutral pose command.")

            elif cmd == "s":
                msg = Bool()
                msg.data = False
                self.float_mode_pub.publish(msg)
                self.get_logger().info("Sent stiff (free-float=False).")

            elif cmd == "":
                x = self.ee_pose.position.x
                y = self.ee_pose.position.y
                z = self.ee_pose.position.z
                print(f"Current pose:  x={x:.3f}, y={y:.3f}, z={z:.3f}")

            elif cmd.lower() == "x":
                self.get_logger().info("Shutting down.")
                rclpy.shutdown()
                break

            else:
                print("Unknown command.")

            # Process incoming messages
            rclpy.spin_once(self, timeout_sec=0.1)


def main():
    rclpy.init()
    node = SimpleTeleopNode()
    node.run()

if __name__ == '__main__':
    main()

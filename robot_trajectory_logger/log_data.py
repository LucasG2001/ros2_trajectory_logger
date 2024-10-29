import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Wrench, Pose
from sensor_msgs.msg import JointState  # Replace with correct message type for franka_robot_state
from franka_msgs.msg import FrankaRobotState
from messages_fr3.srv import PlannerService
from std_srvs.srv import Trigger
import numpy as np
import json
from datetime import datetime
import os
import time
from scipy.spatial.transform import Rotation as R

def quaternion_to_euler(quaternion):
     # Create a Rotation object from the quaternion
    rotation = R.from_quat(quaternion)
    # Convert to Euler angles (roll, pitch, yaw)
    euler_angles = rotation.as_euler('xyz', degrees=False)  # radians
    return euler_angles

def homogenous_transform_to_pose(transform):
    # transform 4x4 pose matrix into position and orientation
    # Ensure the matrix is 4x4
    assert transform.shape == (4, 4), "Input matrix must be 4x4."

    # Extract translation (x, y, z)
    translation = transform[:3, 3]

    # Extract rotation matrix (top-left 3x3)
    rotation_matrix = transform[:3, :3]

    # Convert rotation matrix to Euler angles (roll, pitch, yaw)
    rotation = R.from_matrix(rotation_matrix)
    euler_angles = rotation.as_euler('xyz', degrees=False)  # roll, pitch, yaw

    # Convert rotation matrix to quaternion [x, y, z, w]
    quaternion = rotation.as_quat()

    # Combine translation and Euler angles into a pose vector
    pose_vector = np.concatenate((translation, euler_angles))

    return pose_vector, quaternion


class RobotTrajectoryLogger(Node):

    def __init__(self):
        super().__init__('robot_trajectory_logger')

        # Add the Pose publisher
        self.pose_publisher = self.create_publisher(Pose, 'admittance_controller/reference_pose', 10)

        # Create a service server for PlannerService
        self.srv = self.create_service(PlannerService, 'planner_service', self.handle_service)

        # Timer for sending trajectory at 20Hz
        self.timer_send_trajectory = self.create_timer(1.0 / 0.3, self.send_trajectory)

        # Timer for logging robot state at 100Hz
        self.timer_log_data = self.create_timer(1.0 / 100.0, self.log_data)

        # Subscribe to the robot state
        self.subscription = self.create_subscription(
            FrankaRobotState,  # Replace with the correct message type for franka_robot_state
            '/franka_robot_state_broadcaster/robot_state',
            self.robot_state_callback,
            10)

        # Initialize state and variables
        self.time_start = time.time()
        self.reference_pose = Pose()
        self.ee_pose = Pose()
        self.f_ext = Wrench()
        self.reference_pose.position.x = 0.5
        self.reference_pose.position.y = 0.0
        self.reference_pose.position.z = 0.5
        self.reference_pose.orientation.w = 0.0
        self.reference_pose.orientation.x = 1.0
        self.reference_pose.orientation.y = 0.0
        self.reference_pose.orientation.z = 0.0
        self.ee_euler_angles = quaternion_to_euler([self.ee_pose.orientation._x, self.ee_pose.orientation._y, self.ee_pose.orientation._z, self.ee_pose.orientation._w])
        self.reference_euler_angles = quaternion_to_euler([self.reference_pose.orientation._x, self.reference_pose.orientation._y, self.reference_pose.orientation._z, self.reference_pose.orientation._w])


        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
        self.log_file = (f"robot_state_log_{timestamp}.json")
        print(self.log_file)

        # Initialize force bias so this can be set via service
        self.bias_force = np.array([0.0, 0.0, 0.0])

    def handle_service(self, request, response):
        command = request.command
        match command:
            case 'a':
                self.logging_active = True
                if self.f_ext is not None:
                    self.bias_force = np.array([self.f_ext.force._x, self.f_ext.force._y, self.f_ext.force._z])
                self.get_logger().info(f'Logging activated. Force bias set to: {self.bias_force}')
                response.success = True
            case 'd':
                self.logging_active = False
                self.get_logger().info('Logging deactivated.')
                response.success = True
            case _:
                response.success = False
        return response
    
    def send_trajectory(self):
        current_time = time.time() - self.time_start
        amplitude = 0.3
        frequency = 0.1

        # Generate sinusoidal trajectory in X direction
        self.reference_pose.position.x = 0.5 + 0*  0.2 * np.sin(2 * np.pi * 0.2 * current_time)
        self.reference_pose.position.y = amplitude * np.sin(2 * np.pi * frequency * current_time)
        self.reference_pose.position.z = 0.4 +  0 * amplitude * np.sin(2 * np.pi * 0.2 * current_time)
        self.get_logger().info(f'Sent trajectory Point at time: {current_time:.2f} seconds')
        # self.pose_publisher.publish(self.reference_pose)

    def robot_state_callback(self, msg: FrankaRobotState):
        self.f_ext = msg._o_f_ext_hat_k._wrench  # Assuming this is the correct attribute
        self.ee_pose.position = msg.o_t_ee._pose._position
        self.ee_pose.orientation = msg.o_t_ee._pose._orientation
        quaternion = [self.ee_pose.orientation._x, self.ee_pose.orientation._y, self.ee_pose.orientation._z, self.ee_pose.orientation._w]
        self.ee_euler_angles = quaternion_to_euler(quaternion)

    def log_data(self):

        if not self.logging_active:
            return
        
        # Ensure data is available before logging
        if self.f_ext is None or self.reference_pose is None or self.ee_euler_angles is None:
            self.get_logger().warn("f_ext is None. Skipping log entry.")
            return
        
        # Adjust the force by subtracting the bias
        adjusted_force = np.array([self.f_ext.force._x, self.f_ext.force._y, self.f_ext.force._z]) - self.bias_force

        # Prepare the data to log
        data_to_log = {
            "f_ext": {
                "force": {
                    "x": adjusted_force[0],
                    "y": adjusted_force[1],
                    "z": adjusted_force[2]
                },
                "torque": {
                    "x": self.f_ext._torque._x,
                    "y": self.f_ext._torque._y,
                    "z": self.f_ext._torque._z
                }
            },
            "reference_position": {
                "x": self.reference_pose.position._x,
                "y": self.reference_pose.position._y,
                "z": self.reference_pose.position._z
            },
            "euler_angles": {
                "roll": self.reference_euler_angles[0],
                "pitch": self.reference_euler_angles[1],
                "yaw": self.reference_euler_angles[2]
            },
            "ee_pose":
            {
                "position": {
                    "x": self.ee_pose.position._x,
                    "y": self.ee_pose.position._y,
                    "z": self.ee_pose.position._z
                },
                "orientation": {
                    "roll": self.ee_euler_angles[0],
                    "pitch": self.ee_euler_angles[1],
                    "yaw": self.ee_euler_angles[2]
                }
            }
        }

        # Log data to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(data_to_log) + '\n')


def main(args=None):
    rclpy.init(args=args)
    node = RobotTrajectoryLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

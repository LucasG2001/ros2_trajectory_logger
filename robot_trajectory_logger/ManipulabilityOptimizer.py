import rclpy
from rclpy.node import Node
import pinocchio as pin
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from messages_fr3.msg import JointConfig , PoseDirection
from franka_msgs.msg import FrankaRobotState

# TODO: add joint configuration to desired pose message 

class JointOptimizer(Node):
    def __init__(self, urdf_path, ee_frame_name):
        super().__init__('joint_optimizer')
        # Load the robot model and create data for kinematics/dynamics calculations
        # Define joint limits and initial configuration
        self.joint_limits_lower = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159, -0.04, -0.04])
        self.joint_limits_upper = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159, 0.04, 0.04])
        self.fd = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId(ee_frame_name)

        # initialize self.q0 as an empty std::array<double, 7> 
        self.q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        
        self.joint_config_publisher = self.create_publisher(JointConfig, '/joint_config', 10)

        self.subscription = self.create_subscription(PoseDirection, '/pose_direction', self.pose_direction_callback, 10)

        self.subscription = self.create_subscription(
            FrankaRobotState,  # Replace with the correct message type for franka_robot_state
            '/franka_robot_state_broadcaster/robot_state',
            self.robot_state_callback,
            10)

        self.joint_config = JointConfig()
    

    def objective_function(self, q):
        # Forward kinematics and Jacobian calculation
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        pin.computeJointJacobians(self.model, self.data, q)
        
        # Retrieve the Jacobian in the base frame
        J = pin.getFrameJacobian(self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        
        # Objective function value calculation
        return -np.dot(self.f_d, np.linalg.inv(J @ J.T) @ self.f_d)  # Minimize the negative to maximize the value

    def pose_constraint(self, q):
        # Calculate forward kinematics for pose constraint
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        
        # Calculate position and orientation error
        current_pose = self.data.oMf[self.ee_frame_id]
        position_error = current_pose.translation - self.desired_pose.translation
        orientation_error = pin.log3(self.desired_pose.rotation.T @ current_pose.rotation)
        
        # Combine errors into a single vector
        return np.concatenate([position_error, orientation_error])

    def optimize(self, q0, joint_limits_lower, joint_limits_upper, max_iterations=3000):
        # Define bounds from joint limits
        bounds = [(low, high) for low, high in zip(joint_limits_lower, joint_limits_upper)]
        constraints = {'type': 'eq', 'fun': self.pose_constraint}

        # Initial objective function evaluation
        initial_objective = self.objective_function(q0)
        print("Initial objective is:", initial_objective)

        # Run optimization
        result = minimize(
            self.objective_function, 
            q0, 
            constraints=constraints,
            bounds=bounds,
            method='SLSQP',  
            options={'maxiter': max_iterations, 'disp': True}
        )
        
        # Process optimization results
        if result.success:
            print("Optimal joint configuration:", result.x)
            print("Optimal joint configuration in degrees:", result.x * 180 / np.pi)
            
            # publish the optimal joint configuration 
            
            self.joint_config.joint1 = result.x[0]
            self.joint_config.joint2 = result.x[1]
            self.joint_config.joint3 = result.x[2]
            self.joint_config.joint4 = result.x[3]
            self.joint_config.joint5 = result.x[4]
            self.joint_config.joint6 = result.x[5]
            self.joint_config.joint7 = result.x[6]

            self.joint_config_publisher.publish(self.joint_config)

            # Extract optimal end-effector pose
            optimal_ee_pose = self.data.oMf[self.ee_frame_id]
            position = optimal_ee_pose.translation
            rotation_matrix = optimal_ee_pose.rotation
            euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz')
            
            print("Optimal EE position:", position)
            print("Optimal EE orientation (Euler angles):", euler_angles)
            print("Maximized objective value:", -result.fun)
        else:
            print("Optimization failed:", result.message)
        
        return result
    
    def pose_direction_callback(self, request):
        # Set desired pose in SE(3)
        # Extract desired pose and force direction from the service request
        desired_rotation_matrix = R.from_euler('xyz', [request.roll, request.pitch, request.yaw]).as_matrix()
        self.desired_pose = pin.SE3(desired_rotation_matrix, np.array([request.x, request.y, request.z]))
        self.f_d = np.append(np.array([request.directionx, request.directiony, request.directionz]), [0.0, 0.0, 0.0])

        # print the desired pose and force direction
        print("Desired pose:", self.desired_pose)
        print("Desired force direction:", self.f_d)
        
        # Run the optimization
        self.optimize(self.q0, self.joint_limits_lower, self.joint_limits_upper)

    def robot_state_callback(self, msg: FrankaRobotState):
        self.q0 = np.append(np.array(msg.measured_joint_state.position), [0.0, 0.0])


############ END OF CLASS DEFINITION ################################################

def main(args=None):
        # Define parameters
        urdf_path = '/home/nilsjohnson/franka_ros2_ws/src/ros2_trajectory_logger/fr3.urdf'
        ee_frame_name = 'fr3_hand'
        rclpy.init(args=args)
        node = JointOptimizer(urdf_path, ee_frame_name)
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
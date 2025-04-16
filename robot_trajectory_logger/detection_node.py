import rclpy
from rclpy.node import Node
from franka_msgs.msg import FrankaRobotState
from messages_fr3.srv import PlannerService
from std_msgs.msg import Float64, Bool, Float64MultiArray
import numpy as np
import json
from datetime import datetime
import GPy
import time
from scipy.spatial.transform import Rotation as R
from collections import deque


class BreakthroughDetection(Node):

    def __init__(self):
        super().__init__('robot_trajectory_logger')

        # Add the Pose publisher
        self.pose_publisher = self.create_publisher(Bool, 'cartesian_impedance_control/trigger', 1)

        self.gp_publisher = self.create_publisher(Float64MultiArray, '/gp_values', 1)

        self.trigger_publisher = self.create_publisher(Bool, '/trigger', 1)

        # Create a service server for PlannerService
        self.srv = self.create_service(PlannerService, 'planner_service', self.handle_service)

        
        # Subscribe to the robot state
        self.subscription = self.create_subscription(
            FrankaRobotState,  # Replace with the correct message type for franka_robot_state
            '/franka_robot_state_broadcaster/robot_state',
            self.robot_state_callback,
            1)
        
        # subscription only when testing
        self.displacement_subscription = self.create_subscription(
            Float64,  # Replace with the correct message type for franka_robot_state
            '/displacement_value',
            self.displacement_callback,
            1)
        
          # Subscribe to the drilling force (already direction-corrected)
        self.force_subscription = self.create_subscription(
            Float64,  # Replace with the correct message type for franka_robot_state
            '/f_ext_desired',
            self.drilling_force_callback,
            1)
        
          # Subscribe to the drilling velocity (already direction-corrected)
        self.velocity_subscription = self.create_subscription(
            Float64,  # Replace with the correct message type for franka_robot_state
            '/velocity_desired',
            self.velocity_callback,
            1)
        
        # initialize timer that calls our process for detecting breakthrough at at 1000Hz
        self.process_timer = self.create_timer(1.0 / 1000.0, self.process_data)
        # Initialize state and variables
        self.counter = None
        self.velocity = 0
        self.velocity_buffer = deque([0.0] * 10, maxlen=10) # FIFO buffer for velocity
        self.displacement = 0
        self.initial_position = None
        self.f_ext = 0
        self.ee_pos = np.array([0.0, 0.0, 0.0])
        self.feature_size = 10 # number of previous measurements to use for prediction
        self.refit_interval = 200  # Refit every "refit_interval" steps
        self.sample_size = 20 # number of samples to use for training per interval
        self.has_triggered = False # keep track of change points
        self.logging_active = False # if logging is false we do not detect
        self.lower_bound = None
        self.upper_bound = None # bounds for anomaly detection
        self.trigger_counter = 0 # counter for trigger messages
        # Initialize the Gaussian Process model
        # Define GP kernel (Sum of RBF and a Constant term)
        self.kernel = GPy.kern.RBF(input_dim=self.feature_size, variance=1.0, lengthscale=np.ones(self.feature_size) * 1, ARD=True)
        self.kernel.lengthscale.fix()
        self.kernel.variance.constrain_bounded(1e-2, 0.2)    # Variance between 0.5 and 10
        # Initialize GP model
        self.gp = GPy.models.GPRegression(np.zeros((1, self.feature_size)), np.zeros((1, self.feature_size)), self.kernel, noise_var=1e-3)
        # Constrain noise variance
        self.gp.Gaussian_noise.variance.fix()  # Noise variance between 1e-4 and 0.5
        print("Initial hyperparameters:\n", self.gp)
        # buffer for measurements
        self.X_train = np.zeros([self.refit_interval, self.feature_size])  # n_samples x n_features
        self.y_train = np.zeros([self.refit_interval, 1])  # n_samples x n_targets
        # Initialize force bias so this can be set via service
        self.bias_force = 0.0  #!!! bias force set to zero when testing with the data_streamer node and pre-recorded data as these already have the bias force deducted!!!

    def handle_service(self, request, response):
        command = request.command
        match command:
            case 'a':
                self.logging_active = True
                if self.f_ext is not None and self.counter is None:
                    # !!! commented out for testing purposes when using pre-recorded data !!
                    #self.bias_force = np.array([self.f_ext.force._x, self.f_ext.force._y, self.f_ext.force._z])
                    #self.initial_position = self.ee_pos
                    self.displacement = 0 # initialize displacement
                    self.counter = 0 # initialize counter
                    self.lower_bound = -1
                    self.upper_bound = 1
                self.get_logger().info(f'Logging activated. Force bias set to: {self.bias_force}')
                response.success = True
            case 'd':
                self.logging_active = False
                self.counter = 0 # reset counter
                self.get_logger().info('Logging deactivated.')
                response.success = True
            case _:
                response.success = False
        return response
    
    # Callback for the robot state
    def robot_state_callback(self, msg: FrankaRobotState):
        self.ee_pos = np.array([msg.o_t_ee._pose._position._x, msg.o_t_ee._pose._position._y, msg.o_t_ee._pose._position._z])
        if self.initial_position is not None:
            self.displacement = np.linalg.norm(self.ee_pos - self.initial_position)
        else:
            self.initial_position = self.ee_pos
            

    def displacement_callback(self, msg: Float64):
        self.displacement = msg.data
        # self.get_logger().info(f"Displacement: {self.displacement}")
        
    def drilling_force_callback(self, msg: Float64):
        # self.get_logger().info(f"Force: {msg.data}")
        # self.get_logger().info(f"Bias_Force: {self.bias_force}")
        self.f_ext = msg.data - self.bias_force
        # self.get_logger().info(f"Drilling force: {self.f_ext}")

    def velocity_callback(self, msg: Float64):
        self.velocity = msg.data
        #self.get_logger().info(f"Velocity: {self.velocity}")

        # check for anomaly
        if (self.logging_active == True):
            if self.velocity < self.lower_bound or self.velocity > self.upper_bound:
                # avoid false positives by only counting anomalies when inside the bone
                # also avoid detecting anomalies when the process has not been fitted yet
                if self.displacement < -0.002 and self.f_ext < 0.0 and self.velocity < 0.0: 
                    if self.has_triggered == False:
                        self.has_triggered = True
                        self.trigger_counter += 1

                        if self.trigger_counter == 1:
                            self.get_logger().info(f"Anomaly detected at {self.displacement} mm with force {self.f_ext} N and velocity {self.velocity} m/s")
                        
                        # send the trigger message once the second breakthrough is detected
                        if self.trigger_counter > 1:
                            self.get_logger().info(f"Second anomaly detected at {self.displacement} mm with force {self.f_ext} N and velocity {self.velocity} m/s")
                            # publish trigger message
                            trigger_msg = Bool()
                            trigger_msg.data = True
                            self.trigger_publisher.publish(trigger_msg)


            if self.trigger_counter <= 1:
                trigger_msg = Bool()
                trigger_msg.data = False
                self.trigger_publisher.publish(trigger_msg)
            else:
                trigger_msg = Bool()
                trigger_msg.data = True
                self.trigger_publisher.publish(trigger_msg)

            if self.has_triggered == True and self.velocity > 0.0: # reset trigger when velocity reaches 0 again
                self.has_triggered = False
                
            # Store training data in buffer, refill for every interval
            if self.displacement < -0.001: # has_triggered == False:
                self.X_train[self.counter % self.refit_interval, :] = list(self.velocity_buffer)
                self.y_train[self.counter % self.refit_interval, :] = np.array([[self.velocity]])

            
            # update buffer
            self.velocity_buffer.append(self.velocity)
            # self.get_logger().info(f"Velocity: {self.velocity}")

    def process_data(self):
        if (self.logging_active == True):
            """
            Perform Gaussian Process Regression with self-correlation on the velocity vector data using GPy library. 
            Simulates a Datastream of the Force and uses the previous values to predict the next value(s).
            """
            start_time = time.time()
            # Parameters
            subsampling_factor = self.refit_interval//self.sample_size # subsampling factor for training data
            # Storage for predictions and Logging 
            means = []
            sigmas = []
            times = []
            # Feature vector: last `feature size` points
            feature_vector = np.array([list(self.velocity_buffer)])
            # Predict next sequence
            y_pred, y_std = self.gp.predict(feature_vector) # predict disturbance
            mean_prediction = (y_pred.flatten()[-1])
            sigma_prediction =(np.sqrt(y_std.flatten()[-1]))  # Extract only the last prediction
            # for logging
            means.append(mean_prediction)
            sigmas.append(sigma_prediction)

            # check for anomaly (change point)
            self.lower_bound = mean_prediction - 1.99 * sigma_prediction 
            self.upper_bound = mean_prediction + 1.99 * sigma_prediction
            
            # print calculated values all 1000ms
            if self.counter % 1000 == 0:
                self.get_logger().info(f"Mean: {mean_prediction}, Lower: {self.lower_bound}, Upper: {self.upper_bound}")
                self.get_logger().info(f"Displacement: {self.displacement}")
                self.get_logger().info(f"Drilling force: {self.f_ext}")
                # self.get_logger().info(f"Trigger: {self.has_triggered}")
                # print("Times:", times)
                # print("Means:", means)
                # print("Sigmas:", sigmas)
            
            # Publish GP values for live plotting
            gp_values = Float64MultiArray()
            gp_values.data = [mean_prediction, self.lower_bound, self.upper_bound]
            self.gp_publisher.publish(gp_values)

            # Refit GP every `refit_interval` steps
            if self.counter % self.refit_interval == 0 and self.has_triggered == False:
                # Update kernel hyperparameters
                # Set new data and refit (same hyperparameters) but only on a subset of data
                self.gp.set_XY(self.X_train[::subsampling_factor], self.y_train[::subsampling_factor])
                self.gp.optimize()
                #print("Updated hyperparameters:\n", self.gp)
                end_time = time.time()
                times.append(end_time - start_time)

            self.counter += 1
            
            # print(f"Counter: {self.counter}")
 


     


def main(args=None):
    rclpy.init(args=args)
    node = BreakthroughDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

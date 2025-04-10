import rclpy
from rclpy.node import Node
from messages_fr3.srv import PlannerService, SetStiffness, SetMode, ControllerActivation

class PlannerClient(Node):

    def __init__(self):
        super().__init__('planner_client')
        # Create a client for the PlannerService service
        self.planner_client = self.create_client(PlannerService, 'planner_service')
        # Create a client for the SetStiffness service
        self.stiffness_client = self.create_client(SetStiffness, 'set_stiffness')
        # Create a client for the SetMode service
        self.mode_client = self.create_client(SetMode, 'set_mode')
        # Create a client for the ControllerActivation service
        self.activation_client = self.create_client(ControllerActivation, 'controller_activation')
        # Wait for the services to become available
        while not self.planner_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('PlannerService not available, waiting again...')
        while not self.stiffness_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('SetStiffness service not available, waiting again...')
        while not self.mode_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('SetMode service not available, waiting again...')
        while not self.activation_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('ControllerActivation service not available, waiting again...')
        self.get_logger().info('All services are now available.')
        
        # Initialize the request objects
        self.planner_request = PlannerService.Request()
        self.stiffness_request = SetStiffness.Request()
        self.mode_request = SetMode.Request()
        self.activation_request = ControllerActivation.Request()

        # Initialize objects
        self.mode = False
        self.controller_active = False
        self.stiffness_request.a = 1500.0
        self.stiffness_request.b = 1500.0
        self.stiffness_request.c = 0.0
        self.stiffness_request.d = 100.0
        self.stiffness_request.e = 100.0
        self.stiffness_request.f = 15.0


    def send_planner_request(self, command):
        # Set the request command
        self.planner_request.command = command
        # Call the service asynchronously
        self.future = self.planner_client.call_async(self.planner_request)
        # Wait for the service call to complete
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

    def send_stiffness_request(self, a, b, c, d, e, f):
        # Set the stiffness values
        self.stiffness_request.a = a
        self.stiffness_request.b = b
        self.stiffness_request.c = c
        self.stiffness_request.d = d
        self.stiffness_request.e = e
        self.stiffness_request.f = f
        # Call the service asynchronously
        self.future = self.stiffness_client.call_async(self.stiffness_request)
        # Wait for the service call to complete
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
    
    def send_mode_request(self, mode):
        # Set the mode request
        self.mode_request.mode = mode
        # Call the service asynchronously
        self.future = self.mode_client.call_async(self.mode_request)
        try:
            # Wait for the service call to complete
            rclpy.spin_until_future_complete(self, self.future)
            response = self.future.result()
            if response.success:
                return response
            else:
                # Log specific error message if available
                error_message = getattr(response, 'error_message', 'Unknown error')  # Adjust 'error_message' as needed
                self.get_logger().error(f"Failed to set mode. Reason: {error_message}")
                return response
        except Exception as e:
            self.get_logger().error(f"Service call failed due to an exception: {str(e)}")
            return None

    
    def send_activation_request(self, controller_activation):
        # Set the mode request
        self.activation_request.controller_activation = controller_activation
        # Call the service asynchronously
        self.future = self.mode_client.call_async(self.activation_request)
        # Wait for the service call to complete
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    planner_client = PlannerClient()

    while rclpy.ok():
        # Get user input
        user_input = input("Enter command (e.g., 'ff' to enter free float mode, 's' to activate the controller with the current pose and fixed orientation, 'k' to update stiffness values, 'a' to activate logging, 'd' to deactivate logging, 'b' to set the bias on the force): ")
        parts = user_input.split()  # Split the input into parts
        if len(parts) >= 1:  # Check if there is at least one part
            command = parts[0]  # The first part is the command
            if command == 'k':  # Check if the command is 'k' and there are 6 stiffness values
                a = 1500.0
                b = 1500.0
                c = 0.0
                d = 100.0
                e = 100.0
                f = 15.0
                response = planner_client.send_stiffness_request(a, b, c, d, e, f)
                if response.success:
                    planner_client.get_logger().info(f"Stiffness values set successfully: {a}, {b}, {c}, {d}, {e}, {f}")
                else:
                    planner_client.get_logger().error("Failed to set stiffness values.")
            elif command == 'ff':
                response = planner_client.send_mode_request(True)
                if response and response.success:
                    planner_client.get_logger().info("Free float mode activated.")
                else:
                    planner_client.get_logger().error("Failed to activate free float mode.")

            elif command == 's':
                response = planner_client.send_activation_request(True)
                if response.success:
                    planner_client.get_logger().info("Controller activated.")
                else:
                    planner_client.get_logger().error("Failed to activate controller.")
            else:   
                response = planner_client.send_planner_request(command)
                if response.success:
                    planner_client.get_logger().info(f"Command '{command}' executed successfully.")
                else:
                    planner_client.get_logger().error(f"Failed to execute command '{command}'.")

    planner_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
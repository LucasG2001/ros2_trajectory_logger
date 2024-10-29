import rclpy
from rclpy.node import Node
from messages_fr3.srv import PlannerService, SetStiffness, SetMode

class PlannerClient(Node):

    def __init__(self):
        super().__init__('planner_client')
        # Create a client for the PlannerService service
        self.planner_client = self.create_client(PlannerService, 'planner_service')
        # Create a client for the SetStiffness service
        self.stiffness_client = self.create_client(SetStiffness, 'set_stiffness')
        # Create a client for the SetMode service
        self.mode_client = self.create_client(SetMode, 'set_mode')
        # Wait for the services to become available
        while not self.planner_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('PlannerService not available, waiting again...')
        while not self.stiffness_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('SetStiffness service not available, waiting again...')
        # Initialize the request objects
        self.planner_request = PlannerService.Request()
        self.stiffness_request = SetStiffness.Request()

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
    
    def send_mode_request(self, free_float):
        # Set the mode request
        self.mode_request.free_float = free_float
        # Call the service asynchronously
        self.future = self.mode_client.call_async(self.mode_request)
        # Wait for the service call to complete
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    planner_client = PlannerClient()

    while rclpy.ok():
        # Get user input
        user_input = input("Enter command (e.g., 'ff' to enter free float mode,'k' to update stiffness values, 'a' to activate logging, 'd' to deactivate logging, 'b' to set the bias on the force): ")
        parts = user_input.split()  # Split the input into parts
        if len(parts) >= 1:  # Check if there is at least one part
            command = parts[0]  # The first part is the command
            if command == 'k':  # Check if the command is 'k' and there are 6 stiffness values
                    a = 1500
                    b = 1500
                    c = 0
                    d = 100
                    e = 100
                    f = 15
                    response = planner_client.send_stiffness_request(a, b, c, d, e, f)
                    if response.success:
                        planner_client.get_logger().info(f"Stiffness values set successfully: {a}, {b}, {c}, {d}, {e}, {f}")
                    else:
                        planner_client.get_logger().error("Failed to set stiffness values.")
            elif command == 'ff':
                response = planner_client.send_mode_request(free_float=True)
                if response.success:
                    planner_client.get_logger().info("Free float mode activated.")
                else:
                    planner_client.get_logger().error("Failed to activate free float mode.")
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
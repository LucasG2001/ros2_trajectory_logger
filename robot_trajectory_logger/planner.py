import rclpy
from rclpy.node import Node
from custom_msgs.srv import PlannerService  # Import the custom service

class PlannerClient(Node):

    def __init__(self):
        super().__init__('planner_client')
        # Create a client for the PlannerService service
        self.client = self.create_client(PlannerService, 'planner_service')
        # Wait for the service to become available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        # Initialize the request object
        self.request = PlannerService.Request()

    def send_request(self, command):
        # Set the request command
        self.request.command = command
        # Call the service asynchronously
        self.future = self.client.call_async(self.request)
        # Wait for the service call to complete
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    planner_client = PlannerClient()

    while rclpy.ok():
        # Get user input
        user_input = input("Enter command (e.g., 'k' to update stiffness matrix, 'a' to activate logging, 'd' to deactivate logging, 'b' to set the bias on the force): ")
        parts = user_input.split()  # Split the input into parts
        if len(parts) >= 1:  # Check if there is at least one part
            command = parts[0]  # The first part is the command
            response = planner_client.send_request(command)
            if response.success:
                planner_client.get_logger().info(f"Command '{command}' executed successfully.")
            else:
                planner_client.get_logger().error(f"Failed to execute command '{command}'.")

    planner_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
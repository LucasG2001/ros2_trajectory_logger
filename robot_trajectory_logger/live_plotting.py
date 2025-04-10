import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from std_msgs.msg import Float64MultiArray, Float64


class LivePlotting(Node):

    def __init__(self):

        super().__init__('live_plotting_node')

        # Subscribe to the drilling velocity (already direction-corrected)
        self.subscription = self.create_subscription(
            Float64,  # Replace with the correct message type for franka_robot_state
            '/velocity_desired',
            self.velocity_callback,
            1)
        

        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/gp_values',
            self.gp_callback,
            1)

        # initialize buffers for live plotting
        self.means = []
        self.sigmas = []
        self.lower_bounds = []
        self.upper_bounds = []
        self.time_steps = []
        self.velocity = 0.0

        # Initialize live plot
        self.fig, self.ax = plt.subplots()
        self.line_mean, = self.ax.plot([], [], label="Mean Prediction")
        self.line_lower, = self.ax.plot([], [], label="Lower Bound", linestyle="--")
        self.line_upper, = self.ax.plot([], [], label="Upper Bound", linestyle="--")
        self.line_velocity, = self.ax.plot([], [], label="Velocity", linestyle=":")
        self.ax.legend()
        self.ax.set_title("Live Prediction Plot")
        self.ax.set_xlabel("Time Step")
        self.ax.set_ylabel("Prediction")

    def velocity_callback(self, msg: Float64):
        self.velocity = msg.data

        # print velocity every 1000ms
        if len(self.time_steps) % 1000 == 0:
            self.get_logger().info(f"Velocity: {self.velocity}")
        
    
    def gp_callback(self, msg: Float64MultiArray):
        mean = msg.data[0]
        lower = msg.data[1]
        upper = msg.data[2]

        # print mean, lower, upper as it arrives to see if something is incomming
        #self.get_logger().info(f"Mean: {mean}, Lower: {lower}, Upper: {upper}")

        self.means.append(mean)
        self.lower_bounds.append(lower)
        self.upper_bounds.append(upper)
        self.time_steps.append(len(self.time_steps))  # Keep X-axis synced


    def update_plot(self, frame):
    # Update plot data
        self.line_mean.set_data(self.time_steps, self.means)
        self.line_lower.set_data(self.time_steps, self.lower_bounds)
        self.line_upper.set_data(self.time_steps, self.upper_bounds)
        self.line_velocity.set_data(self.time_steps, [self.velocity] * len(self.time_steps))

        # Adjust plot limits
        self.ax.relim()
        self.ax.autoscale_view()

        return self.line_mean, self.line_lower, self.line_upper, self.line_velocity

    def start_live_plot(self):
        # Start live plotting
        ani = FuncAnimation(self.fig, self.update_plot, interval=1)
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = LivePlotting()
    node.start_live_plot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
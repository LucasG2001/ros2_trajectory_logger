import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from std_msgs.msg import Float64MultiArray, Float64, Bool
import threading


class LivePlotting(Node):
    def __init__(self):
        super().__init__('live_plotting_node')

        self.velocity_subscription = self.create_subscription(
            Float64, '/velocity_desired', self.velocity_callback, 1)
        
        self.gp_values_subscription = self.create_subscription(
            Float64MultiArray, '/gp_values', self.gp_callback, 1)
    
        self.trigger_subscription = self.create_subscription(
            Bool, '/trigger', self.trigger_callback, 1)

        self.means = []
        self.sigmas = []
        self.lower_bounds = []
        self.upper_bounds = []
        self.time_steps = []
        self.velocity = []
        self.trigger_values = []


        self.fig, self.ax = plt.subplots()
        self.line_mean, = self.ax.plot([], [], label="Mean Prediction")
        self.line_lower, = self.ax.plot([], [], label="Lower Bound", linestyle="--")
        self.line_upper, = self.ax.plot([], [], label="Upper Bound", linestyle="--")
        self.line_velocity, = self.ax.plot([], [], label="Velocity")
        # self.line_trigger, = self.ax.plot([], [], label="Trigger Times")
        self.ax.legend()
        self.ax.set_title("Live Prediction Plot")
        self.ax.set_xlabel("Time Step")
        self.ax.set_ylabel("Prediction")


    def velocity_callback(self, msg: Float64):
        self.velocity.append(msg.data)

    def trigger_callback(self, msg: Bool):
        # if the trigger is False, append a zero to the trigger_values list and if it is True, append a 1
        if msg.data:
            self.trigger_values.append(0.2)
        else:
            self.trigger_values.append(0)
            

    def gp_callback(self, msg: Float64MultiArray):
        
        if len(msg.data) < 3:
            self.get_logger().warn("Received incomplete GP data.")
            return

        mean = msg.data[0]
        lower = msg.data[1]
        upper = msg.data[2]

        current_step = len(self.time_steps)

        self.means.append(mean)
        self.lower_bounds.append(lower)
        self.upper_bounds.append(upper)
        self.time_steps.append(current_step)

        # Ensure all lists stay aligned
        min_len = min(len(self.means), len(self.lower_bounds), len(self.upper_bounds), len(self.time_steps))
        self.means = self.means[:min_len]
        self.lower_bounds = self.lower_bounds[:min_len]
        self.upper_bounds = self.upper_bounds[:min_len]
        self.time_steps = self.time_steps[:min_len]
        self.trigger_values = self.trigger_values[:min_len]



    def update_plot(self, frame):
        min_len = min(
            len(self.time_steps),
            len(self.means),
            len(self.lower_bounds),
            len(self.upper_bounds),
            len(self.velocity)
            #len(self.trigger_values)
        )

        if min_len == 0:
            return self.line_mean, self.line_lower, self.line_upper, self.line_velocity

        x = self.time_steps[:min_len]
        self.line_mean.set_data(x, self.means[:min_len])
        self.line_lower.set_data(x, self.lower_bounds[:min_len])
        self.line_upper.set_data(x, self.upper_bounds[:min_len])
        self.line_velocity.set_data(x, self.velocity[:min_len])
        # self.line_trigger.set_data(x, self.trigger_values[:min_len]) 

        self.ax.relim()
        self.ax.autoscale_view()
        return self.line_mean, self.line_lower, self.line_upper, self.line_velocity




    def start_live_plot(self):
        # Make the plot window larger
        manager = plt.get_current_fig_manager()
        manager.resize(1200, 800)  # Set the size to 1200x800 pixels

        ani = FuncAnimation(self.fig, self.update_plot, interval=100)
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = LivePlotting()

    # Start rclpy.spin in a separate thread
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # Run plotting in main thread (important for GUI)
    node.start_live_plot()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool, Int16
from franka_msgs.msg import FrankaRobotState
from rclpy.action import ActionClient
from franka_msgs.action import Grasp, Move
from geometry_msgs.msg import Wrench
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import csv
import numpy as np

fixed_offset = [0.41, 0.0, 0.026]  # Fixed offset (fixation center) in meters
#!/usr/bin/env python3


class SimpleTeleopNode(Node):

    def __init__(self):
        super().__init__('simple_teleop_node')

        # Publishers
        self.cartesian_pub = self.create_publisher(
            Pose,
            #'/joint_impedance_controller/reference_pose',
            '/cartesian_impedance_controller/reference_pose',
            1
        )
        self.float_mode_pub = self.create_publisher(
            Int16,
            #'/joint_impedance_controller/control_mode',
            '/cartesian_impedance_controller/control_mode',
            1
        )

        # Action client for gripper grasp
        # Gripper action clients
        self.move_client = ActionClient(self, Move, '/fr3_gripper/move')
        self.grasp_client = ActionClient(self, Grasp, '/fr3_gripper/grasp')


        # Subscribers
        self.pose_subscriber = self.create_subscription(
            FrankaRobotState,
            '/franka_robot_state_broadcaster/robot_state',
            self.robot_state_callback,
            1)

        self.get_logger().info("Simple teleop node started.")

        # Initialize state
        self.ee_pose = Pose()
        self.f_ext = Wrench()

        # Load poses from CSV
        csv_path = os.path.join(os.getcwd(), "points.csv")
        if not os.path.exists(csv_path):
            self.get_logger().error(f"CSV file not found at {csv_path}")
            return


        df = pd.read_csv(csv_path)
        print(df.head())
        # Expect columns: Label,X (m),Y (m),Z (m)
        self.poses = []
        for _, row in df.iterrows():
            p = Pose()
            p.position.x = float(row["X (m)"]) + fixed_offset[0]
            p.position.y = float(row["Y (m)"]) + fixed_offset[1]
            p.position.z = float(row["Z (m)"]) + fixed_offset[2]
            # fixed orientation (0,1,0,0)
            p.orientation.w = float(row["qw"])
            p.orientation.x = float(row["qx"])
            p.orientation.y = float(row["qy"])
            p.orientation.z = float(row["qz"])
            self.poses.append(p)


        if len(self.poses) < 16:
            self.get_logger().error("CSV must have at least 16 poses (8 first + 8 last).")
            print("length was ", len(self.poses))
            print(self.poses)
            return

        # Prepare sequence: alternating between first 8 and last 8
        first_eight = self.poses[:4] 
        last_eight = self.poses[-4:][::-1]
        self.sequence = []
        for i in range(4):
            self.sequence.append((first_eight[i], True))   # grasp after these
            self.sequence.append((last_eight[i], False))   # no grasp after these

        self.get_logger().info(f"Prepared {len(self.sequence)} poses for execution.")

        # Initialize logs
        self.force_log = [] # List of (timestamp, fx, fy, fz)
        self.event_log = [] # List of (timestamp, event_type)

    def wait(self, duration):
        """Wait for `duration` seconds while keeping the node spinning."""
        start_time = time.time()
        while time.time() - start_time < duration:
            rclpy.spin_once(self, timeout_sec=0.05)  # adjust timeout as needed

     # Helper to create a neutral pose
    def make_neutral_pose(self):
        neutral = Pose()
        neutral.position.x = 0.45
        neutral.position.y = 0.0
        neutral.position.z = 0.45
        neutral.orientation.x = 1.0
        neutral.orientation.y = 0.0
        neutral.orientation.z = 0.0
        neutral.orientation.w = 0.0
        return neutral


    def robot_state_callback(self, msg: FrankaRobotState):
        # print("Received robot state message.")
        self.f_ext = msg._o_f_ext_hat_k._wrench
        self.ee_pose.position = msg.o_t_ee._pose._position
        self.ee_pose.orientation = msg.o_t_ee._pose._orientation
        # Log force and time
        now = time.time()
        self.force_log.append((now, self.f_ext.force.x, self.f_ext.force.y, self.f_ext.force.z))

    def send_grasp(self):
        if not self.grasp_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Grasp action server not available!")
            return

        goal_msg = Grasp.Goal()
        goal_msg.width = 0.0        # fully close
        goal_msg.speed = 0.1
        goal_msg.force = 200.0
        goal_msg.epsilon.inner = 0.02
        goal_msg.epsilon.outer = 0.02

        self.get_logger().info("Sending grasp action goal...")
        send_goal_future = self.grasp_client.send_goal_async(goal_msg)

        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Grasp goal rejected!")
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result

        if result.success:
            self.get_logger().info("Grasp succeeded.")
        else:
            self.get_logger().warn("Grasp failed.")

    def send_move(self, width):
        open_msg = Move.Goal()
        open_msg.width = 0.06    
        open_msg.speed = 0.1
        self.get_logger().info("Sending open action goal...")
        send_goal_future = self.move_client.send_goal_async(open_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("open goal rejected!")
            return
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        if result.success:
            self.get_logger().info("gripper open succeeded.")
        else:
            self.get_logger().warn("gripper close failed.")

    def run(self):
        if not hasattr(self, "sequence"):
            self.get_logger().error("No sequence prepared. Exiting run loop.")
            return

        pose_errors = []  # Store errors for logging and plotting

        # Return to neutral at start
        self.send_move(0.06)  # ensure gripper is open
        self.cartesian_pub.publish(self.make_neutral_pose())
        self.get_logger().info("Returned to neutral pose.")
        self.wait(3.0)

        for i, (pose, do_grasp) in enumerate(self.sequence):
            # Approach from above
            pre_pose = Pose()
            pre_pose.position.x = pose.position.x
            pre_pose.position.y = pose.position.y
            pre_pose.position.z = pose.position.z + 0.06
            pre_pose.orientation = pose.orientation

            # Grasp or release
            if do_grasp:
                # self.float_mode_pub.publish(Int16(data=0))  # switch to grasped mode
                self.cartesian_pub.publish(pre_pose)
                self.wait(3.5)
                # LOG
                error = [self.ee_pose.position.x - pose.position.x,
                self.ee_pose.position.y - pose.position.y]
                pose_errors.append({'step': i, 'x_error': error[0], 'y_error': error[1]})
                self.get_logger().info(f"pose error {error}")
                # End Log
                self.cartesian_pub.publish(pose)  # go back down
                self.get_logger().info(f"Published pose {i+1}/{len(self.sequence)}")
                self.wait(3.5)
                # LOG
                error = [
                self.ee_pose.position.x - pose.position.x,
                self.ee_pose.position.y - pose.position.y]
                pose_errors.append({'step': i, 'x_error': error[0], 'y_error': error[1]})
                # end log
                self.send_grasp()
                self.wait(1.0)
                pre_pose.position.z += 0.04  # small lift before going up
                self.cartesian_pub.publish(pre_pose)  # lift back up
                self.wait(3.0)
            else:
                self.cartesian_pub.publish(pre_pose)
                self.wait(3.5)
                # LOG
                error = [self.ee_pose.position.x - pose.position.x,
                self.ee_pose.position.y - pose.position.y]
                pose_errors.append({'step': i, 'x_error': error[0], 'y_error': error[1]})
                self.get_logger().info(f"pose error {error}")
                # move down to pose
                self.cartesian_pub.publish(pose)  # go back down
                self.event_log.append((time.time(), 'insertion'))
                self.get_logger().info(f"Published pose {i+1}/{len(self.sequence)}")
                self.wait(0.5)
                self.float_mode_pub.publish(Int16(data=1))  # switch to insertion mode
                self.wait(3.0)
                # LOG
                error = [
                self.ee_pose.position.x - pose.position.x,
                self.ee_pose.position.y - pose.position.y]
                pose_errors.append({'step': i, 'x_error': error[0], 'y_error': error[1]})
                # end log
                # open gripper
                self.send_move(0.06)
                self.wait(1.0)
                self.float_mode_pub.publish(Int16(data=0))  # switch back to floating mode
                self.cartesian_pub.publish(pre_pose)  # lift back up
                self.wait(3.0)
                
            

        # Return to neutral at the end
        self.cartesian_pub.publish(self.make_neutral_pose())
        self.get_logger().info("Returned to neutral pose.")

        # Save errors to CSV
        csv_path = os.path.join(os.getcwd(), "pose_errors.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['step', 'x_error', 'y_error']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in pose_errors:
                writer.writerow(row)
        self.get_logger().info(f"Pose errors saved to {csv_path}")

        # Prepare data in mm
        steps_pre = [row['step'] for idx, row in enumerate(pose_errors) if idx % 2 == 0]
        x_errors_pre = [row['x_error'] * 1000 for idx, row in enumerate(pose_errors) if idx % 2 == 0]
        y_errors_pre = [row['y_error'] * 1000 for idx, row in enumerate(pose_errors) if idx % 2 == 0]

        steps_post = [row['step'] for idx, row in enumerate(pose_errors) if idx % 2 == 1]
        x_errors_post = [row['x_error'] * 1000 for idx, row in enumerate(pose_errors) if idx % 2 == 1]
        y_errors_post = [row['y_error'] * 1000 for idx, row in enumerate(pose_errors) if idx % 2 == 1]

        # Calculate averages
        avg_x_pre = np.mean([abs(x) for x in x_errors_pre])
        avg_y_pre = np.mean([abs(y) for y in y_errors_pre])
        avg_x_post = np.mean([abs(x) for x in x_errors_post])
        avg_y_post = np.mean([abs(y) for y in y_errors_post])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Pre-Grasp subplot
        ax1.scatter(steps_pre, x_errors_pre, color='blue', marker='o', label='X Error Pre-Grasp')
        ax1.scatter(steps_pre, y_errors_pre, color='red', marker='o', label='Y Error Pre-Grasp')
        for step, err in zip(steps_pre, x_errors_pre):
            ax1.annotate(f"{err:.2f}", (step, err), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
        for step, err in zip(steps_pre, y_errors_pre):
            ax1.annotate(f"{err:.2f}", (step, err), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
        ax1.set_ylabel('Error (mm)')
        ax1.set_title(f'Pre-Grasp Errors (Integrator, - ) (avg X: {avg_x_pre:.2f} mm, avg Y: {avg_y_pre:.2f} mm)')
        ax1.legend()
        ax1.grid(True)
        ymin, ymax = ax1.get_ylim()
        ax1.set_yticks(np.arange(np.floor(ymin/0.4)*0.4, np.ceil(ymax/0.4)*0.4+0.4, 0.4))

        # Post-Grasp subplot
        ax2.scatter(steps_post, x_errors_post, color='blue', marker='x', label='X Error Post-Grasp')
        ax2.scatter(steps_post, y_errors_post, color='red', marker='x', label='Y Error Post-Grasp')
        for step, err in zip(steps_post, x_errors_post):
            ax2.annotate(f"{err:.2f}", (step, err), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
        for step, err in zip(steps_post, y_errors_post):
            ax2.annotate(f"{err:.2f}", (step, err), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Error (mm)')
        ax2.set_title(f'Post-Grasp Errors (Integrator, - ) (avg X: {avg_x_post:.2f} mm, avg Y: {avg_y_post:.2f} mm)')
        ax2.legend()
        ax2.grid(True)
        ymin, ymax = ax2.get_ylim()
        ax2.set_yticks(np.arange(np.floor(ymin/0.4)*0.4, np.ceil(ymax/0.4)*0.4+0.4, 0.4))

        plt.suptitle('Pose Errors Pre- and Post-Grasp')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()

        # Plot forces if available
        if self.force_log:
            force_log_np = np.array(self.force_log)
            t0 = force_log_np[0, 0]
            times = force_log_np[:, 0] - t0
            fx = force_log_np[:, 1]
            fy = force_log_np[:, 2]
            fz = force_log_np[:, 3]

            # Compute EMA of Fz
            alpha = 0.1
            ema_fz = []
            for i, val in enumerate(fz):
                if i == 0:
                    ema_fz.append(val)
                else:
                    ema_fz.append(alpha * val + (1 - alpha) * ema_fz[-1])
            ema_fz = np.array(ema_fz)

            plt.figure(figsize=(12, 6))
            plt.plot(times, fx, label='Fx')
            plt.plot(times, fy, label='Fy')
            plt.plot(times, fz, label='Fz')
            plt.plot(times, ema_fz, color='red', linewidth=2, label='EMA Fz (α=0.1)')

            # Mark events
            for t, event in self.event_log:
                t_rel = t - t0
                if event == 'pose_published':
                    plt.axvline(t_rel, color='green', linestyle='--', alpha=0.7, label='Pose Published')
                elif event == 'wait_finished':
                    plt.axvline(t_rel, color='magenta', linestyle=':', alpha=0.7, label='Wait Finished')
                elif event == 'insertion or pullout complete':
                    plt.axvline(t_rel, color='orange', linestyle='-.', alpha=0.7, label='insertion')

            # Avoid duplicate legend entries
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

            plt.xlabel('Time (s)')
            plt.ylabel('Force (N)')
            plt.title('External Force (f_ext) with Pose and Wait Events')
            plt.grid(True)
            plt.tight_layout()
            plt.show()


def main():
    rclpy.init()
    node = SimpleTeleopNode()
    node.run()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

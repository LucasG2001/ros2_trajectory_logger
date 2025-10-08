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

fixed_offset = [0.41, 0.0, -0.03]  # Fixed offset (fixation center) in meters
#!/usr/bin/env python3


def df_to_poses(df):
    poses = []
    # Quaternion for 90 deg rotation around Z axis
    rot_z90 = R.from_euler('z', 90, degrees=True)
    # Quaternion for 180 deg rotation around X axis
    rot_x180 = R.from_euler('x', 180, degrees=True)
    # Compose the two rotations: first z, then x
    rot_total = rot_x180 * rot_z90
    for _, row in df.iterrows():
        p = Pose()
        p.position.x = float(row["X (m)"]) + fixed_offset[0]
        p.position.y = float(row["Y (m)"]) + fixed_offset[1]
        p.position.z = float(row["Z (m)"]) + fixed_offset[2]

        # Original quaternion from CSV
        quat_orig = [
            -float(row["qx"]),
            -float(row["qy"]),
            -float(row["qz"]),
            float(row["qw"])
        ]

        # Quaternion multiplication: q_new = q_total * q_orig
        rot_orig = R.from_quat(quat_orig)
        rot_new = rot_total * rot_orig
        quat_new = rot_new.as_quat()  # [x, y, z, w]

        p.orientation.x = quat_new[0]
        p.orientation.y = quat_new[1]
        p.orientation.z = quat_new[2]
        p.orientation.w = quat_new[3]
        poses.append(p)
    return poses

def hover_pose(pose: Pose, z_offset: float = 0.06) -> Pose:
        """
        Return a copy of the given pose, offset in Z by z_offset.
        """
        pre = Pose()
        pre.position.x = pose.position.x
        pre.position.y = pose.position.y
        pre.position.z = pose.position.z + z_offset
        pre.orientation = pose.orientation
        return pre


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

        # Initialize states
        self.ee_pose = Pose()
        self.f_ext = Wrench()

        # Load and split CSV by identifier
        csv_path = os.path.join(os.getcwd(), "robot_trajectory_logger/assembly_poses.csv")
        if not os.path.exists(csv_path):
            self.get_logger().error(f"CSV file not found at {csv_path}")
            return
        df_all = pd.read_csv(csv_path)

        df_pins = df_all[df_all['identifier'] == 'pins'].reset_index(drop=True)
        df_jigs = df_all[df_all['identifier'] == 'jigs'].reset_index(drop=True)
        df_uprofiles = df_all[df_all['identifier'] == 'UPROFILE'].reset_index(drop=True)
        print(f"Loaded {len(df_pins)} pin poses, {len(df_jigs)} jig poses, {len(df_uprofiles)} u-profile poses.")

        self.sequence = []


        # --- D2: pick first D2 (between B-points), place at last D (U-profiles)---
        if len(df_uprofiles) >= 2:
            d2_pick = df_to_poses(df_uprofiles.iloc[0:2])
            d2_place = df_to_poses(df_uprofiles.iloc[-2:])
            print("appending u profiles")
            for pick_pose, place_pose in zip(d2_pick, d2_place):
                self.sequence.append((pick_pose, True))
                self.sequence.append((place_pose, False))
            print(self.sequence)
            print(self.sequence)

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
        goal_msg.epsilon.inner = 0.04
        goal_msg.epsilon.outer = 0.04

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

    def send_move(self, width=0.06):
        open_msg = Move.Goal()
        open_msg.width = width   
        open_msg.speed = 0.15
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

    

    def pick_and_place(self, pick_pose: Pose, place_pose: Pose):
        """
        Executes a pick and place routine:
        - Hover over pick_pose, descend, grasp, and lift.
        - Hover over place_pose, descend, release, and lift.
        """

        pose_errors = []

        # -------------------
        # PICK
        # -------------------
        pre_pick = hover_pose(pick_pose)
        self.float_mode_pub.publish(Int16(data=0))  # switch to stiff mode

        # move to pre-pick
        self.cartesian_pub.publish(pre_pick)
        self.wait(3.5)
        error = [self.ee_pose.position.x - pick_pose.position.x,
                self.ee_pose.position.y - pick_pose.position.y]
        pose_errors.append({'step': 'pre-pick', 'x_error': error[0], 'y_error': error[1]})
        self.get_logger().info(f"pose error {error}")

        # move down to pick
        self.cartesian_pub.publish(pick_pose)
        self.wait(3.5)
        error = [self.ee_pose.position.x - pick_pose.position.x,
                self.ee_pose.position.y - pick_pose.position.y]
        pose_errors.append({'step': 'pick', 'x_error': error[0], 'y_error': error[1]})
        self.get_logger().info(f"pose error {error}")

        # grasp
        self.send_grasp()
        self.wait(1.0)

        # lift back up
        pre_pick.position.z += 0.06
        self.cartesian_pub.publish(pre_pick)
        self.wait(3.0)

        # -------------------
        # PLACE
        # -------------------
        pre_place = hover_pose(place_pose)

        # move to pre-place
        self.cartesian_pub.publish(pre_place)
        self.wait(3.5)
        error = [self.ee_pose.position.x - place_pose.position.x,
                self.ee_pose.position.y - place_pose.position.y]
        pose_errors.append({'step': 'pre-place', 'x_error': error[0], 'y_error': error[1]})
        self.get_logger().info(f"pose error {error}")

        # move down to place
        self.cartesian_pub.publish(place_pose)
        self.event_log.append((time.time(), 'insertion'))
        self.get_logger().info(f"Published place pose")
        self.wait(1.5)
        self.float_mode_pub.publish(Int16(data=1))  # switch to insertion mode
        self.wait(2.0)
        error = [self.ee_pose.position.x - place_pose.position.x,
                self.ee_pose.position.y - place_pose.position.y]
        pose_errors.append({'step': 'place', 'x_error': error[0], 'y_error': error[1]})
        self.get_logger().info(f"pose error {error}")

        # release
        self.send_move(0.025)
        self.float_mode_pub.publish(Int16(data=0))  # switch back to stiff mode
        self.wait(1.0)

        # lift back up
        pre_place.position.z += 0.06
        self.cartesian_pub.publish(pre_place)
        self.wait(3.0)

        # -------------------
        # Return to neutral
        # -------------------
        self.cartesian_pub.publish(self.make_neutral_pose())
        self.get_logger().info("Returned to neutral pose.")

        return pose_errors

    def run(self):
        if not hasattr(self, "sequence"):
            self.get_logger().error("No sequence prepared. Exiting run loop.")
            return

        pose_errors = []  # Store errors for logging and plotting

        # Return to neutral at start
        self.send_move(0.025)  # ensure gripper is open
        self.cartesian_pub.publish(self.make_neutral_pose())
        self.get_logger().info("Returned to neutral pose.")
        self.wait(3.0)

        # Iterate over sequence in pairs (pick, place)
        for i in range(0, len(self.sequence), 2):
            if i + 1 >= len(self.sequence):
                break
            pick_pose, do_grasp_pick = self.sequence[i]
            print(f'pick_pose: {pick_pose}')
            place_pose, do_grasp_place = self.sequence[i + 1]
            print(f'place_pose: {place_pose}')
            # Only pick if do_grasp_pick is True and place if do_grasp_place is False
            if do_grasp_pick and not do_grasp_place:
                errors = self.pick_and_place(pick_pose, place_pose)
                pose_errors.extend(errors)
            else:
                self.get_logger().warn(f"Unexpected sequence at step {i}: pick={do_grasp_pick}, place={do_grasp_place}")

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

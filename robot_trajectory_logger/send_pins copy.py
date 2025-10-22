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
from robot_trajectory_logger.wiggle_ee import wiggle_pose, wiggle_out

fixed_offset = [0.4, 0.0, 0.029]  # Fixed offset (fixation center) in meters
#!/usr/bin/env python3


def ee_pose_from_part_pose(part_pos, part_quat, l, d):
    """
    Compute gripper pose in world frame given:
    - part_pos: (3,) part position in world
    - part_quat: (4,) part quaternion in world [x,y,z,w]
    - l: translation along part y-axis in local grasp
    - d: translation along part z-axis in local grasp

    The resulting orientation is wrapped modulo 90° around world Z.
    """

    # --- 1) l_P (local grasp)
    l_P = np.eye(4)
    l_P[:3, 3] = [0, 0, 0]

    # --- 2) g_T_gl (local -> gripper)
    g_T_gl = np.eye(4)

    # --- 3) Base rotation: part orientation * flip X180
    R_P = R.from_quat(part_quat).as_matrix()
    R_x180 = R.from_euler('x', 180, degrees=True).as_matrix()
    R_wg = R_P @ R_x180

    # --- 4) Extract yaw and apply modulo 90
    euler = R.from_matrix(R_wg).as_euler('xyz', degrees=True)
    roll, pitch, yaw = euler
    yaw_mod = np.mod(yaw, 90.0)  # e.g. 110° → 20°, 185° → 5°

    # rebuild rotation with wrapped yaw
    R_wg = R.from_euler('xyz', [roll, pitch, yaw_mod], degrees=True).as_matrix()

    # --- 5) Build homogeneous transform
    w_T_wg = np.eye(4)
    w_T_wg[:3, :3] = R_wg
    w_T_wg[:3, 3] = part_pos

    # --- 6) Combine with local offsets (optional)
    T_total = w_T_wg @ g_T_gl @ l_P

    gripper_pos = T_total[:3, 3]
    gripper_quat = R.from_matrix(T_total[:3, :3]).as_quat()

    return gripper_pos, gripper_quat


def df_to_poses(df, rotx = True, rotz = False):
    poses = []
    for _, row in df.iterrows():

        part_position = np.array([row["X (m)"], row["Y (m)"], row["Z (m)"]])
        part_orientation = np.array([float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])])

        desired_pos, desired_orientation_quat = ee_pose_from_part_pose(part_position, part_orientation, l=0.0, d=0.0)
        p = Pose()
        p.position.x = desired_pos[0] 
        p.position.y = desired_pos[1]  
        p.position.z = desired_pos[2]
        p.orientation.x = desired_orientation_quat[0]
        p.orientation.y = desired_orientation_quat[1]
        p.orientation.z = desired_orientation_quat[2]
        p.orientation.w = desired_orientation_quat[3]

        if p.position.x + fixed_offset[0] > 0.4:
            poses.append(p)
    
    print(f"length of poses is {len(poses)}")
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
        self.fixed_offset = np.array([0.4, 0.0, 0.029])

        # Subscribers
        self.pose_subscriber = self.create_subscription(
            FrankaRobotState,
            '/franka_robot_state_broadcaster/robot_state',
            self.robot_state_callback,
            1)
        
        self.wheel_subscriber = self.create_subscription(
            Pose,
            '/wheel_center',
            self.wheel_callback,
            1)

        self.get_logger().info("Simple teleop node started.")

        # Initialize state
        self.ee_pose = Pose()
        self.f_ext = Wrench()

        # Load and split CSV by identifier
        csv_path = os.path.join(os.getcwd(), "assembly_poses.csv")
        if not os.path.exists(csv_path):
            self.get_logger().error(f"CSV file not found at {csv_path}")
            return
        df_all = pd.read_csv(csv_path)

        df_pins = df_all[df_all['identifier'] == 'PIN'].reset_index(drop=True)
        print(f"Loaded {len(df_pins)} pin poses.")

        self.sequence = []

        # Convert df_pins to Pose objects
        df_pins = df_to_poses(df_pins)
        # Split df_pins into two non-overlapping halves
        half = len(df_pins) // 2
        pins_pick = df_pins[2:half]
        pins_place = df_pins[half+2:]

        # Build sequence: pick → place
        for pick_pose, place_pose in zip(pins_pick, pins_place):
            self.sequence.append((pick_pose, True))    # Pick
            self.sequence.append((place_pose, False))  # Place

        self.get_logger().info(f"Prepared {len(self.sequence)} poses for execution.")

        # Initialize logs
        self.force_log = [] # List of (timestamp, fx, fy, fz)
        self.event_log = [] # List of (timestamp, event_type)

    def wheel_callback(self, msg: Pose):
        self.fixed_offset[0] = msg.position.x
        self.fixed_offset[1] = msg.position.y
        self.fixed_offset[2] = msg.position.z

    def wait(self, duration):

        """Wait for `duration` seconds while keeping the node spinning."""
        start_time = time.time()
        while time.time() - start_time < duration:
            rclpy.spin_once(self, timeout_sec=0.01)  # adjust timeout as needed

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

        # move to pre-pick
        self.cartesian_pub.publish(pre_pick)
        self.wait(3.0)
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
        self.float_mode_pub.publish(Int16(data=2))
        self.wait(0.1)
        self.send_grasp()
        self.wait(1.0)
        self.cartesian_pub.publish(self.ee_pose) # stay
        self.wait(0.5)
        pre_pick = self.ee_pose
        # lift back up
        # wiggle_out(base_pose=self.ee_pose, z_dist=0.08, duration=3.0, rate=200, publisher=self.cartesian_pub)
        pre_pick.position.z += 0.06 
        self.float_mode_pub.publish(Int16(data=0))
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
        self.float_mode_pub.publish(Int16(data=1)) # control mode = 1 (LOWER STIFFNESS)
        self.cartesian_pub.publish(place_pose)
        self.wait(1.5) # halfway through
        self.get_logger().info(f"Published place pose")
        wiggle_pose(base_pose=place_pose, amplitude=0.025, duration=3.0, rate=200, publisher=self.cartesian_pub)
        self.wait(1.5)
        self.float_mode_pub.publish(Int16(data=2)) # low stiffness in hole
        self.cartesian_pub.publish(self.ee_pose) # stay
        self.event_log.append((time.time(), 'insertion'))
        error = [self.ee_pose.position.x - place_pose.position.x,
                self.ee_pose.position.y - place_pose.position.y]
        pose_errors.append({'step': 'place', 'x_error': error[0], 'y_error': error[1]})
        self.get_logger().info(f"pose error {error}")

        # release
        self.send_move(0.06)
        self.wait(1.0)
        self.float_mode_pub.publish(Int16(data=0)) # control mode = 0 (Hgh STIFFNESS)

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
        self.send_move(0.06)  # ensure gripper is open
        self.cartesian_pub.publish(self.make_neutral_pose())
        self.get_logger().info("Returned to neutral pose.")
        self.wait(3.0)

        # Iterate over sequence in pairs (pick, place)
        for i in range(0, len(self.sequence), 2):
            if i + 1 >= len(self.sequence):
                break
            pick_pose, do_grasp_pick = self.sequence[i]  
            pick_pose.position.x += self.fixed_offset[0] 
            pick_pose.position.y += self.fixed_offset[1]
            pick_pose.position.z += self.fixed_offset[2]    
            place_pose, do_grasp_place = self.sequence[i + 1]
            place_pose.position.x += self.fixed_offset[0]            
            place_pose.position.y += self.fixed_offset[1]
            place_pose.position.z += self.fixed_offset[2] 
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

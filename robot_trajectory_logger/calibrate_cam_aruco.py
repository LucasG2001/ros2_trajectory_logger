#!/usr/bin/env python3
"""
simple_teleop_node.py

Sends neutral pose -> rotated pose, then continuously records frames from a ZED camera
and robot poses for 10 seconds. Detects a single ArUco marker per frame, saves marker midpoint
(in camera coords) and orientation (as quaternion in camera coords) and saves robot poses.
After 10 seconds stops recording, writes CSVs and computes best-fit transform T such that:
    P_robot = T * P_cam
(using a rigid-body least squares fit / Umeyama (no scale) / SVD).
"""

import rclpy
from rclpy.node import Node
import numpy as np
import time
import csv
import math
from math import radians
from geometry_msgs.msg import Pose
from std_msgs.msg import Int16
from robot_trajectory_logger.SimpleTeleopNode import SimpleTeleopNode
from robot_trajectory_logger.send_uprofiles import make_neutral_pose
from franka_msgs.msg import FrankaRobotState

# Additional dependencies
import cv2
# Make sure opencv-contrib-python is installed (for aruco)
# ZED bindings
try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except Exception:
    ZED_AVAILABLE = False
    # We'll still provide code but it will error at runtime if ZED SDK is missing.

from scipy.spatial.transform import Rotation as R  # for quaternion math; requires scipy

# ------------------------
# Helper functions
# ------------------------

def pose_to_numpy(pose: Pose):
    """Return (position: 3,) and (quat: 4,) from geometry_msgs Pose"""
    p = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
    q = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w], dtype=float)
    return p, q

def numpy_to_pose(p: np.ndarray, q: np.ndarray) -> Pose:
    pose = Pose()
    pose.position.x = float(p[0]); pose.position.y = float(p[1]); pose.position.z = float(p[2])
    pose.orientation.x = float(q[0]); pose.orientation.y = float(q[1]); pose.orientation.z = float(q[2]); pose.orientation.w = float(q[3])
    return pose

def rotate_orientation_around_world_z(pose_in: Pose, angle_deg) -> Pose:
    """
    Rotate a Pose's orientation by angle_deg about world Z axis.
    Position is left the same; orientation is composed with the rotation.
    """
    p = np.array([pose_in.position.x, pose_in.position.y, pose_in.position.z])
    q = np.array([
        pose_in.orientation.x,
        pose_in.orientation.y,
        pose_in.orientation.z,
        pose_in.orientation.w
    ])
    Rz = R.from_euler('z', angle_deg, degrees=True)
    q_rot = (Rz * R.from_quat(q)).as_quat()
    pose_out = Pose()
    pose_out.position = pose_in.position
    pose_out.orientation.x, pose_out.orientation.y, pose_out.orientation.z, pose_out.orientation.w = q_rot
    return pose_out

def quaternion_from_two_vectors(v_from, v_to):
    """Return quaternion representing rotation from v_from to v_to"""
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)
    axis = np.cross(v_from, v_to)
    cosangle = np.dot(v_from, v_to)
    if np.linalg.norm(axis) < 1e-8:
        # parallel or anti-parallel
        if cosangle > 0:
            return np.array([0,0,0,1], dtype=float)
        else:
            # 180 deg rotation: find orthogonal axis
            axis = np.array([1,0,0], dtype=float)
            if abs(v_from[0]) > 0.9:
                axis = np.array([0,1,0], dtype=float)
            axis = axis - v_from * np.dot(axis, v_from)
            axis = axis / np.linalg.norm(axis)
            return R.from_rotvec(np.pi * axis).as_quat()
    angle = math.acos(np.clip(cosangle, -1.0, 1.0))
    axis = axis / np.linalg.norm(axis)
    return R.from_rotvec(angle * axis).as_quat()

def umeyama_rigid(A: np.ndarray, B: np.ndarray):
    """
    Estimate rigid transform T (4x4) that maps points A to B (P_B = T * P_A).
    A, B are (N,3) corresponding points.
    This enforces rotation + translation, no scale.
    Returns T (4x4).
    """
    assert A.shape == B.shape
    N = A.shape[0]
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    A_c = A - centroid_A
    B_c = B - centroid_B
    H = A_c.T @ B_c
    U, S, Vt = np.linalg.svd(H)
    R_ = Vt.T @ U.T
    if np.linalg.det(R_) < 0:
        Vt[-1,:] *= -1
        R_ = Vt.T @ U.T
    t = centroid_B - R_ @ centroid_A
    T = np.eye(4)
    T[0:3,0:3] = R_
    T[0:3,3] = t
    return T

# ------------------------
# Node
# ------------------------

class MarkerCalibrationNode(SimpleTeleopNode):

    def __init__(self):
        super().__init__()

        self.robot_pose_lock = False  # simple lock if needed

        # ZED camera handle and parameters
        self.zed = None
        self.zed_cam_ready = False

        # Data storage
        self.robot_poses_list = []   # list of dicts: {index, t, position(3), quat(4)}
        self.marker_poses_list = []  # list of dicts: {index, t, cam_position(3), cam_quat(4)}
        self.index = 0

        self.robot_poses_list = []   # list of dicts: {index, t, position(3), quat(4)}
        self.marker_poses_list = []  # list of dicts: {index, t, cam_position(3), cam_quat(4)}
        self.index = 0

    # ZED helpers
    def init_zed(self, svo_filename: str = None):
        if not ZED_AVAILABLE:
            self.get_logger().error("ZED SDK Python bindings not available (pyzed.sl). Install ZED SDK Python API.")
            return False
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        init_params.coordinate_units = sl.UNIT.METER
        if svo_filename is not None:
            # We will enable recording later. But if you want to open SVO file, set this:
            pass
        cam = sl.Camera()
        status = cam.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f"Failed to open ZED camera: {status}")
            return False
        self.zed = cam
        self.zed_cam_ready = True
        self.get_logger().info("ZED camera initialized.")
        return True

    def start_zed_recording(self, svo_path: str):
        """
        Start SVO recording. Uses RecordingParameters.
        """
        if not self.zed_cam_ready:
            self.get_logger().error("ZED not ready for recording.")
            return False
        rec_param = sl.RecordingParameters(svo_path, sl.SVO_COMPRESSION_MODE.H265)
        status = self.zed.enable_recording(rec_param)
        if status != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f"Failed to enable ZED recording: {status}")
            return False
        self.get_logger().info(f"ZED recording started: {svo_path}")
        return True

    def stop_zed_recording(self):
        if not self.zed_cam_ready:
            return
        status = self.zed.disable_recording()
        if status != sl.ERROR_CODE.SUCCESS:
            self.get_logger().warn(f"Failed to properly disable ZED recording: {status}")
        else:
            self.get_logger().info("ZED recording stopped.")

    # convert pixel corners to 3D using ZED point cloud
    def pixel_to_point3(self, u, v):
        """
        Given pixel coordinates (u,v), return 3D camera coordinates (x,y,z) in meters.
        Uses the ZED measure and point cloud; for robustness, this function queries the point cloud value at the pixel.
        """
        mat = sl.Mat()
        err = self.zed.retrieve_measure(mat, sl.MEASURE.XYZRGBA)  # point cloud
        # Convert to numpy and sample at integer pixel coordinates
        if err != sl.ERROR_CODE.SUCCESS:
            return None
        # mat.get_data() returns bytes depending on version; use get_value for pixel
        try:
            x = mat.get_value(int(v), int(u))[0]  # row=v, col=u
            y = mat.get_value(int(v), int(u))[1]
            z = mat.get_value(int(v), int(u))[2]
            if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z):
                return None
            return np.array([x, y, z], dtype=float)
        except Exception:
            # fallback: convert whole mat to numpy array (slower)
            try:
                np_pc = mat.get_data()
                # shape should be (height, width, 4) for XYZRGBA
                val = np_pc[int(v), int(u), 0:3]
                if np.any(np.isnan(val)) or np.any(np.isinf(val)):
                    return None
                return val.astype(float)
            except Exception:
                return None

    def run_teleop_and_record(self, duration_s=10.0, svo_out="output.svo"):
        """
        Main routine: publish neutral pose, rotated pose, start recording, loop grabbing frames for duration_s,
        detect ArUco, save robot and marker poses, stop recording, write CSVs, compute transform.
        """
        # 1) send neutral pose
        neutral = make_neutral_pose()
        neutral.position.z = neutral.position.z - 0.25
        self.get_logger().info("Publishing neutral pose.")
        self.cartesian_pub.publish(neutral)
        time.sleep(0.25)

        # 2) send rotated pose (35 deg about world Z)
        rotated = rotate_orientation_around_world_z(neutral, 35.0)
        self.get_logger().info("Publishing rotated pose (+35 deg about world Z).")
        self.cartesian_pub.publish(rotated)
        time.sleep(0.25)

        # 3) initialize ZED and start recording
        ok = self.init_zed(svo_filename=None)
        if not ok:
            self.get_logger().error("ZED initialization failed. Aborting recording.")
            return

        # Prepare ArUco detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_detector =cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        # start recording SVO
        if not self.start_zed_recording(svo_out):
            self.get_logger().error("Failed to start SVO recording; continuing without recording.")
        # Grab loop
        self.get_logger().info(f"Starting continuous capture for {duration_s} seconds.")
        start_t = time.perf_counter()
        end_t = start_t + duration_s

        # We'll also oscillate the published pose slowly over the duration in xyz and orientation,
        # but since user asked not to use timers, we'll just publish modified poses inside the loop.
        freq = 0.25  # Hz
        lin_amp = 0.2  # 7 cm amplitude (within 5-10 cm requested)
        rot_amp_deg = 25.0  # degrees

        # Prepare image buffer
        image = sl.Mat()
        # For obtaining point cloud as Mat, we call retrieve_measure when needed

        # Index counter
        idx = 0
        while time.perf_counter() < end_t and rclpy.ok():
            # 1) grab from ZED (blocking)
            grab_status = self.zed.grab()
            rclpy.spin_once(self, timeout_sec=0.0)
            if grab_status != sl.ERROR_CODE.SUCCESS:
                # could be temporary, skip frame
                self.get_logger().warning(f"ZED grab failed with status {grab_status}; skipping frame.")
                continue

            # Retrieve left image for ArUco detection
            self.zed.retrieve_image(image, sl.VIEW.LEFT)
            # Convert sl.Mat to OpenCV image (BGR)
            try:
                image_ocv = image.get_data()
            except Exception:
                # fallback conversion
                image_ocv = np.asarray(image.get_data()).copy()

            if image_ocv is None:
                self.get_logger().warning("Empty image from ZED; skipping frame.")
                continue

            # convert to grayscale
            if len(image_ocv.shape) == 3 and image_ocv.shape[2] == 4:
                img_bgr = cv2.cvtColor(image_ocv, cv2.COLOR_BGRA2BGR)
            else:
                img_bgr = image_ocv
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # ArUco detection
            corners_list, ids, rejected = aruco_detector.detectMarkers(gray)
            display_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # for visualization
            if ids is not None and len(ids) > 0:
                # Draw the detected marker boundaries and IDs
                cv2.aruco.drawDetectedMarkers(display_img, corners_list, ids)
                
                # ---- same logic for midpoint and 3D extraction ----
                corners = corners_list[0].reshape(-1, 2)  # 4x2
                mid_pixel = corners.mean(axis=0)
                u_mid, v_mid = float(mid_pixel[0]), float(mid_pixel[1])

                # draw midpoint as a small red dot
                cv2.circle(display_img, (int(u_mid), int(v_mid)), 5, (0, 0, 255), -1)
            if ids is None or len(ids) == 0:
                # No marker detected — still record robot pose with NaNs for marker
                marker_mid_cam = np.array([np.nan, np.nan, np.nan], dtype=float)
                marker_quat_cam = np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)
            else:
                # Use the first detected marker (there should be only one)
                corners = corners_list[0].reshape(-1, 2)  # 4x2: (u,v) corners
                # mid pixel:
                mid_pixel = corners.mean(axis=0)  # (u, v)
                u_mid, v_mid = float(mid_pixel[0]), float(mid_pixel[1])
                # Convert corners and midpoint to 3D camera coordinates using ZED point cloud
                # We'll try to get average of corner points in 3D for robustness
                pts3d = []
                for (u, v) in corners:
                    pt = self.pixel_to_point3(u, v)
                    if pt is not None:
                        pts3d.append(pt)
                mid3d = None
                if len(pts3d) >= 1:
                    pts3d = np.array(pts3d)
                    mid3d = pts3d.mean(axis=0)
                    marker_mid_cam = mid3d
                else:
                    marker_mid_cam = np.array([np.nan, np.nan, np.nan], dtype=float)

                # For marker orientation: we can define the marker's +X as vector from corner0 to corner1 (in camera frame)
                try:
                    if len(pts3d) >= 2:
                        v_x = pts3d[1] - pts3d[0]
                        # marker z should point out from marker plane; approximate with cross product of two edges
                        if pts3d.shape[0] >= 3:
                            v_y = pts3d[2] - pts3d[0]
                            v_z = np.cross(v_x, v_y)
                        else:
                            # approximate z from camera center to marker normal (camera looking direction)
                            v_z = np.array([0,0,1], dtype=float)
                        # create orthonormal frame: x, y, z
                        x_axis = v_x / np.linalg.norm(v_x)
                        z_axis = v_z / np.linalg.norm(v_z)
                        y_axis = np.cross(z_axis, x_axis)
                        y_axis = y_axis / np.linalg.norm(y_axis)
                        R_marker = np.column_stack((x_axis, y_axis, z_axis))
                        quat_cam = R.from_matrix(R_marker).as_quat()
                        marker_quat_cam = quat_cam
                    else:
                        marker_quat_cam = np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)
                except Exception:
                    marker_quat_cam = np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)

            # Get current robot pose (best-effort) from last callback
            try:
                rp_pos, rp_quat = pose_to_numpy(self.ee_pose)
                robot_pos = rp_pos
                robot_quat = rp_quat
            except Exception:
                robot_pos = np.array([np.nan, np.nan, np.nan], dtype=float)
                robot_quat = np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)

              ### NEW DISPLAY CODE ###
            # Add index and timestamp overlay
            cv2.putText(display_img, f"Frame: {idx}", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display_img, f"Time: {time.perf_counter()-start_t:.1f}s", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Show image in window
            cv2.imshow("Marker Detection", display_img)
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                self.get_logger().info("User interrupted (ESC/q).")
                break
            # Save entries
            tstamp = time.time()
            self.robot_poses_list.append({
                'index': idx,
                'time': tstamp,
                'px': float(robot_pos[0]) if np.isfinite(robot_pos[0]) else np.nan,
                'py': float(robot_pos[1]) if np.isfinite(robot_pos[1]) else np.nan,
                'pz': float(robot_pos[2]) if np.isfinite(robot_pos[2]) else np.nan,
                'qx': float(robot_quat[0]) if np.isfinite(robot_quat[0]) else np.nan,
                'qy': float(robot_quat[1]) if np.isfinite(robot_quat[1]) else np.nan,
                'qz': float(robot_quat[2]) if np.isfinite(robot_quat[2]) else np.nan,
                'qw': float(robot_quat[3]) if np.isfinite(robot_quat[3]) else np.nan
            })
            self.marker_poses_list.append({
                'index': idx,
                'time': tstamp,
                'cx': float(marker_mid_cam[0]) if np.isfinite(marker_mid_cam[0]) else np.nan,
                'cy': float(marker_mid_cam[1]) if np.isfinite(marker_mid_cam[1]) else np.nan,
                'cz': float(marker_mid_cam[2]) if np.isfinite(marker_mid_cam[2]) else np.nan,
                'mqx': float(marker_quat_cam[0]) if np.isfinite(marker_quat_cam[0]) else np.nan,
                'mqy': float(marker_quat_cam[1]) if np.isfinite(marker_quat_cam[1]) else np.nan,
                'mqz': float(marker_quat_cam[2]) if np.isfinite(marker_quat_cam[2]) else np.nan,
                'mqw': float(marker_quat_cam[3]) if np.isfinite(marker_quat_cam[3]) else np.nan
            })

            # Oscillate the published pose slowly (linear 5-10cm, rotational 10deg) at 0.5Hz
            elapsed = time.perf_counter() - start_t
            lin_offset = lin_amp * np.sin(2.0 * math.pi * freq * elapsed)  # along x direction
            rot_offset_deg = rot_amp_deg * math.sin(2.0 * math.pi * freq * elapsed)
            # Apply to rotated base pose
            base_pos, base_quat = pose_to_numpy(rotated)
            new_pos = base_pos + np.array([lin_offset*0.75, lin_offset, lin_offset*0.5])
            base_R = R.from_quat(base_quat)
            rot_R = R.from_euler('z', rot_offset_deg, degrees=True)
            new_quat = (rot_R * base_R).as_quat()
            new_pose = numpy_to_pose(new_pos, new_quat)
            self.cartesian_pub.publish(new_pose)

            idx += 1
            # small sleep to avoid 100% busy loop — but we must continuously grab frames, so just a tiny sleep
            time.sleep(0.005)

        # End of loop
        self.get_logger().info("Finished capture loop. Stopping ZED recording and saving data.")
        # Stop recording
        try:
            self.stop_zed_recording()
        except Exception as e:
            self.get_logger().warn(f"Exception when stopping recording: {e}")

        # Release ZED resources
        try:
            self.zed.close()
        except Exception:
            pass

        # Write CSVs
        robot_csv = "robot_poses.csv"
        marker_csv = "marker_poses.csv"
        self.get_logger().info(f"Saving robot poses to {robot_csv} and marker poses to {marker_csv}.")
        with open(robot_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['index','time','px','py','pz','qx','qy','qz','qw'])
            writer.writeheader()
            for r in self.robot_poses_list:
                writer.writerow(r)
        with open(marker_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['index','time','cx','cy','cz','mqx','mqy','mqz','mqw'])
            writer.writeheader()
            for m in self.marker_poses_list:
                writer.writerow(m)

        # Post-process: fit transformation mapping camera marker midpoints to robot positions.
        # Collect correspondences where both are valid
        cam_pts = []
        robot_pts = []
        for r, m in zip(self.robot_poses_list, self.marker_poses_list):
            if np.isfinite(m['cx']) and np.isfinite(r['px']):
                cam_pts.append([m['cx'], m['cy'], m['cz']])
                robot_pts.append([r['px'], r['py'], r['pz']])
        cam_pts = np.array(cam_pts, dtype=float)
        robot_pts = np.array(robot_pts, dtype=float)
        if cam_pts.shape[0] >= 3:
            T = umeyama_rigid(cam_pts, robot_pts)  # maps cam -> robot
            print(f"homogenous transform is {T}")
            print(f"camera pose in robot frame is {T @ np.array([0.0, 0.0, 0.0, 1.0])}")
            self.get_logger().info(f"Computed transform T (camera->robot):\n{T}")
            # Save T to CSV
            T_csv = "T_cam_to_robot.csv"
            np.savetxt(T_csv, T, delimiter=',')
            self.get_logger().info(f"Saved transform to {T_csv}.")
        else:
            self.get_logger().warn("Not enough valid correspondences to compute transform (need >=3).")

        self.get_logger().info("All done.")

# ------------------------
# Main
# ------------------------

def main(args=None):
    rclpy.init(args=args)
    node = MarkerCalibrationNode()
    try:
        # Run the 10-second teleop & record routine synchronously (no timers)
        node.run_teleop_and_record(duration_s=10.0, svo_out="output.svo")
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

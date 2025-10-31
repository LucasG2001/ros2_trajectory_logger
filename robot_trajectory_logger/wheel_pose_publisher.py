#!/usr/bin/env python3
"""
marker_tracking_node.py

Continuously tracks a single ArUco marker using the ZED camera,
transforms its 3D midpoint pose to the robot frame using the precomputed
T_chess and cam_0 transforms from transform.yaml, and publishes it to
/wheel_center as geometry_msgs/Pose.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
import numpy as np
import cv2
import yaml
from scipy.spatial.transform import Rotation as R
import time

try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except Exception:
    ZED_AVAILABLE = False


# ------------------------
# Helper functions
# ------------------------

def load_transform_from_yaml(yaml_path="transform.yaml", key="T_chess_cam2"):
    """Load 4x4 transform matrix from YAML (handles nested 'transforms' key)."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # Support both nested and flat formats
    if "transforms" in data:
        data = data["transforms"]

    if key not in data:
        raise KeyError(f"Key '{key}' not found under 'transforms' in {yaml_path}")

    T = np.array(data[key], dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"Invalid transform shape for '{key}'; expected 4x4.")
    return T


def pixel_to_point3(zed, u, v):
    """Return 3D camera coordinates (x,y,z) for a given pixel using ZED point cloud."""
    mat = sl.Mat()
    err = zed.retrieve_measure(mat, sl.MEASURE.XYZRGBA)
    if err != sl.ERROR_CODE.SUCCESS:
        return None
    try:
        val = mat.get_value(int(v), int(u))
        if val[1] != sl.ERROR_CODE.SUCCESS:
            return None
        x, y, z, _ = val[0]
        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z):
            return None
        return np.array([x, y, z], dtype=float)
    except Exception:
        try:
            np_pc = mat.get_data()
            val = np_pc[int(v), int(u), 0:3]
            if np.any(np.isnan(val)) or np.any(np.isinf(val)):
                return None
            return val.astype(float)
        except Exception:
            return None


def cam_to_robot(T_cam_to_robot, p_cam):
    """Apply homogeneous transform to a 3D point in camera frame."""
    p_cam_h = np.hstack((p_cam, 1.0))
    p_robot_h = T_cam_to_robot @ p_cam_h
    return p_robot_h[:3]


# ------------------------
# Node
# ------------------------

class WheelPublisherNode(Node):

    def __init__(self):
        super().__init__("marker_tracking_node")

        # Load transforms from YAML
        try:
            T_cam_0 = load_transform_from_yaml("transform.yaml", "T_chess_cam2")
            T_chess = load_transform_from_yaml("transform.yaml", "T_robot_chess")
            self.T_cam_to_robot = T_chess @ T_cam_0
            self.get_logger().info("Loaded transforms from transform.yaml and computed T_total = T_chess @ T_cam_0")
        except Exception as e:
            self.get_logger().error(f"Failed to load transform(s): {e}")
            raise SystemExit

        # Publisher
        self.pose_pub = self.create_publisher(Pose, "/wheel_center", 1)

        # ZED camera setup
        if not ZED_AVAILABLE:
            self.get_logger().error("ZED SDK not available. Install pyzed.sl.")
            raise SystemExit

        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        init_params.coordinate_units = sl.UNIT.METER
        self.zed = sl.Camera()
        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f"Failed to open ZED: {status}")
            raise SystemExit
        self.get_logger().info("ZED camera opened successfully.")

        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Reusable mats
        self.image = sl.Mat()
        self.get_logger().info("Marker tracking node ready.")

    def spin_forever(self):
        """Main tracking loop."""
        i = 0
        msg = Pose()
        msg.orientation.w = 1.0  # orientation not used

        while rclpy.ok():
            grab_status = self.zed.grab()
            if grab_status != sl.ERROR_CODE.SUCCESS:
                self.get_logger().warn(f"ZED grab failed: {grab_status}")
                continue

            # Retrieve left image
            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            frame = self.image.get_data()
            if frame is None:
                continue
            img_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # Detect ArUco
            corners_list, ids, _ = self.detector.detectMarkers(gray)
            if ids is None or len(ids) == 0:
                cv2.imshow("ZED Marker Tracking", gray)
                if cv2.waitKey(1) == 27:  # ESC
                    break
                continue

            # Use first marker
            corners = corners_list[0].reshape(-1, 2)
            cv2.polylines(img_bgr, [corners.astype(int)], True, (0, 255, 0), 2)
            for c in corners:
                cv2.circle(img_bgr, tuple(c.astype(int)), 4, (0, 0, 255), -1)

            mid_pixel = corners.mean(axis=0)
            u, v = int(mid_pixel[0]), int(mid_pixel[1])
            mid3d = pixel_to_point3(self.zed, u, v)
            if mid3d is None:
                self.get_logger().warn("Invalid 3D point from ZED; skipping.")
                cv2.imshow("ZED Marker Tracking", img_bgr)
                if cv2.waitKey(1) == 27:
                    break
                continue

            # Transform to robot coordinates
            p_robot = cam_to_robot(self.T_cam_to_robot, mid3d)
            msg.position.x, msg.position.y, msg.position.z = p_robot
            self.pose_pub.publish(msg)

            if i % 30 == 0:
                print(f"[wheel_center] Robot-frame marker position: {p_robot.round(3)}")

            # Display with overlay
            cv2.putText(img_bgr,
                        f"Marker: ({p_robot[0]:.3f}, {p_robot[1]:.3f}, {p_robot[2]:.3f}) m",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("ZED Marker Tracking", img_bgr)
            if cv2.waitKey(1) == 27:  # ESC to exit
                break

            rclpy.spin_once(self, timeout_sec=0.0)
            i += 1

        self.cleanup()

    def cleanup(self):
        self.get_logger().info("Shutting down...")
        try:
            self.zed.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        self.get_logger().info("Camera closed. Exiting.")


# ------------------------
# Main
# ------------------------

def main(args=None):
    rclpy.init(args=args)
    node = WheelPublisherNode()
    try:
        node.spin_forever()
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

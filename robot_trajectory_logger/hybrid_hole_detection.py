#!/usr/bin/env python3
"""
hybrid_hole_detection_node.py

Improved hybrid hole detector for a plate using ZED:
 - 2D adaptive-threshold + morphology to find dark regions (holes)
 - Contour-based centroid (more accurate marking)
 - Validate using ZED depth and texture confidence (holes -> LOW confidence values ~0)
 - Accept invalid/missing depth as a possible hole
 - Publish PoseArray of detected hole centers (robot frame when 3D available)
 - Live preview with markers
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
import numpy as np
import cv2
import math
import time

try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except Exception:
    ZED_AVAILABLE = False

# Import helpers from your marker_tracking_node.py (must be in same folder / pythonpath)
from wheel_pose_publisher import pixel_to_point3, cam_to_robot, load_transform_from_yaml


class HybridHoleDetectionNode(Node):
    def __init__(self):
        super().__init__("hybrid_hole_detection_node")

        # Load transform (robot <- camera). Update keys to match your yaml.
        try:
            T_robot_chess = load_transform_from_yaml("transform.yaml", "T_robot_chess")
            # Choose the camera transform you use (cam2/cam1). Adjust key if needed.
            T_chess_cam2 = load_transform_from_yaml("transform.yaml", "T_chess_cam2")
            self.T_cam_to_robot = T_robot_chess @ T_chess_cam2
            self.get_logger().info("Loaded transforms and computed T_cam_to_robot.")
        except Exception as e:
            self.get_logger().error(f"Failed to load transform(s): {e}")
            raise SystemExit

        # Publisher
        self.pub = self.create_publisher(PoseArray, "/hole_centers", 1)

        # ZED setup
        if not ZED_AVAILABLE:
            self.get_logger().error("ZED SDK not available. Install pyzed.sl.")
            raise SystemExit

        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        self.zed = sl.Camera()
        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f"Failed to open ZED: {status}")
            raise SystemExit

        # Mats
        self.image = sl.Mat()
        self.depth = sl.Mat()
        self.confidence = sl.Mat()

        # Tunable params (start values; adjust to your setup)
        self.min_area = 40           # px
        self.max_area = 8000         # px
        self.circularity_thresh = 0.3
        self.conf_thresh = 30        # texture confidence threshold: values <= this -> likely hole (your holes ~0)
        self.local_depth_diff_thresh = 0.01  # m; require depth candidate to be different from local median by this
        self.depth_min = 0.02
        self.depth_max = 2.0

        self.get_logger().info("Hybrid hole detection node ready.")

    def spin_forever(self):
        while rclpy.ok():
            if self.zed.grab() != sl.ERROR_CODE.SUCCESS:
                continue

            # retrieve image + depth + confidence
            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
            self.zed.retrieve_measure(self.confidence, sl.MEASURE.CONFIDENCE)

            frame = self.image.get_data()
            if frame is None:
                continue

            img_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # --- Preprocess: CLAHE + small blur to normalize lighting ---
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_clahe = clahe.apply(gray)
            blur = cv2.GaussianBlur(gray_clahe, (5, 5), 0)

            # --- Adaptive threshold (extract darker regions) ---
            # Use THRESH_BINARY_INV so dark holes become white in mask
            th = cv2.adaptiveThreshold(
                blur, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                blockSize=31,
                C=9
            )

            # --- Morphology: remove small specks, close holes inside bigger blobs ---
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
            clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)

            # --- Find contours of candidate dark regions ---
            contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            depth_map = self.depth.get_data()
            conf_map = self.confidence.get_data()
            h, w = clean.shape

            hole_poses = PoseArray()
            hole_poses.header.frame_id = "robot_base"

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_area or area > self.max_area:
                    continue

                perimeter = cv2.arcLength(cnt, True)
                if perimeter <= 0:
                    continue
                circularity = 4 * math.pi * (area / (perimeter * perimeter + 1e-8))
                # allow somewhat non-perfect circles; user can tighten
                if circularity < self.circularity_thresh:
                    # still accept elongated holes if other signals match; don't reject outright
                    pass

                # centroid (use moments for accurate center)
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                c_x = int(M["m10"] / M["m00"])
                c_y = int(M["m01"] / M["m00"])

                # boundary check
                if c_x < 0 or c_x >= w or c_y < 0 or c_y >= h:
                    continue

                # --- Confidence + depth checks ---
                conf_val = float(conf_map[c_y, c_x])  # YOUR semantics: holes ~ 0, plate ~ 100
                depth_val = float(depth_map[c_y, c_x])

                # candidate scoring heuristics
                score = 0
                reason = []

                # 1) texture confidence: low values (<= conf_thresh) indicate holes in your setup
                if conf_val <= self.conf_thresh:
                    score += 2
                    reason.append("low_conf")

                # 2) invalid depth is often a strong sign for hole/recess on stereo
                if not np.isfinite(depth_val) or depth_val <= 0:
                    score += 2
                    reason.append("invalid_depth")
                else:
                    # 3) if depth is valid, compare to local neighborhood median depth
                    # take a small patch around candidate
                    r = max(3, int(math.sqrt(area) / 2))
                    y0 = max(0, c_y - r)
                    y1 = min(h, c_y + r + 1)
                    x0 = max(0, c_x - r)
                    x1 = min(w, c_x + r + 1)
                    patch = depth_map[y0:y1, x0:x1].astype(np.float32)
                    # mask-out invalids
                    good = patch[np.isfinite(patch) & (patch > 0)]
                    if good.size > 0:
                        local_med = np.median(good)
                        # hole depth may be closer or farther depending on geometry; check difference
                        if abs(depth_val - local_med) >= self.local_depth_diff_thresh:
                            score += 1
                            reason.append("depth_diff")
                    else:
                        # neighborhood has no valid depths -> more likely hole area
                        score += 1
                        reason.append("neigh_invalid")

                # 4) shape/circularity bonus
                if circularity >= 0.5:
                    score += 1
                    reason.append("circular")

                # Accept if score meets threshold (tunable)
                if score < 2:
                    # not enough evidence
                    continue

                # --- compute 3D point when possible ---
                p_robot = None
                p_cam = pixel_to_point3(self.zed, c_x, c_y)
                if p_cam is not None:
                    p_robot = cam_to_robot(self.T_cam_to_robot, p_cam)

                # Publish pose (if no 3D, we still publish pose with z=nan to indicate 2D-only)
                p = Pose()
                if p_robot is not None:
                    p.position.x = float(p_robot[0])
                    p.position.y = float(p_robot[1])
                    p.position.z = float(p_robot[2])
                else:
                    # encode 2D-only by projecting approximate z from depth_val if finite,
                    # otherwise set nan (ROS float must be numeric; use 0 and mark in log)
                    if np.isfinite(depth_val) and depth_val > 0:
                        # approximate robot point via cam_to_robot if possible using depth only:
                        # but we already tried pixel_to_point3; fallback to 0 z and warn
                        p.position.x = 0.0
                        p.position.y = 0.0
                        p.position.z = float(depth_val)
                    else:
                        p.position.x = 0.0
                        p.position.y = 0.0
                        p.position.z = 0.0

                p.orientation.w = 1.0
                hole_poses.poses.append(p)

                # --- visualization ---
                # draw contour, centroid, and annotate conf/depth/score
                cv2.drawContours(img_bgr, [cnt], -1, (0, 180, 0), 2)
                cv2.circle(img_bgr, (c_x, c_y), 4, (0, 0, 255), -1)
                text = f"s{score} c{int(conf_val)} d{(0 if not np.isfinite(depth_val) else round(depth_val,3))}"
                cv2.putText(img_bgr, text, (c_x + 6, c_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # Publish PoseArray (might be empty)
            hole_poses.header.stamp = self.get_clock().now().to_msg()
            self.pub.publish(hole_poses)

            # display preview
            cv2.imshow("Hybrid Hole Detection (improved)", img_bgr)
            if cv2.waitKey(1) == 27:
                break

            rclpy.spin_once(self, timeout_sec=0.0)

        self.cleanup()

    def cleanup(self):
        self.get_logger().info("Shutting down...")
        try:
            self.zed.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        self.get_logger().info("Camera closed. Exiting.")


def main(args=None):
    rclpy.init(args=args)
    node = HybridHoleDetectionNode()
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

#!/usr/bin/env python3
"""
hole_detection_node.py

Continuously grabs frames from the ZED camera and detects circular holes
(using Canny + contour detection). Marks their centers in red in a live
OpenCV preview.

Reuses helper functions from marker_tracking_node.py.
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import time

try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except Exception:
    ZED_AVAILABLE = False

# Import helper functions from marker_tracking_node
from wheel_pose_publisher import pixel_to_point3, load_transform_from_yaml


class HoleDetectionNode(Node):
    def __init__(self):
        super().__init__("hole_detection_node")

        if not ZED_AVAILABLE:
            self.get_logger().error("ZED SDK not available. Install pyzed.sl.")
            raise SystemExit

        # Initialize ZED
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        init_params.coordinate_units = sl.UNIT.METER
        self.zed = sl.Camera()
        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f"Failed to open ZED: {status}")
            raise SystemExit

        self.image = sl.Mat()
        self.get_logger().info("ZED camera opened successfully.")
        self.get_logger().info("Hole detection node ready.")

    def spin_forever(self):
        """Main loop: capture frames, detect holes, and display."""
        while rclpy.ok():
            grab_status = self.zed.grab()
            if grab_status != sl.ERROR_CODE.SUCCESS:
                self.get_logger().warn(f"ZED grab failed: {grab_status}")
                continue

            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            frame = self.image.get_data()
            if frame is None:
                continue

            img_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # --- Edge detection ---
            edges = cv2.Canny(blur, 60, 150)

            # --- Contour detection ---
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < 50 or area > 5000:
                    continue  # ignore too small or too large
                perimeter = cv2.arcLength(c, True)
                circularity = 4 * np.pi * (area / (perimeter * perimeter + 1e-6))
                if circularity < 0.6:
                    continue  # not circular enough
                (x, y), radius = cv2.minEnclosingCircle(c)
                if radius > 1:
                    center = (int(x), int(y))
                    cv2.circle(img_bgr, center, 4, (0, 0, 255), -1)

            # --- Display edges + detections ---
            cv2.imshow("ZED Hole Detection", img_bgr)
            if cv2.waitKey(1) == 27:  # ESC
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
    node = HoleDetectionNode()
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

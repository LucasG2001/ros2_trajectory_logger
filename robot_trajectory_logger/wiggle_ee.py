import numpy as np
from geometry_msgs.msg import Pose, Pose
from scipy.spatial.transform import Rotation as R
import time


def wiggle_pose(
    base_pose,
    amplitude: float = 0.02,
    duration: float = 4.0,
    rate: float = 200.0,
    publisher=None,
    angle_deg = 5.0
):
    """
    Generate or publish a 4-second 'wiggle' motion around a base pose.

    Args:
        base_pose: geometry_msgs.msg.Pose or np.ndarray [x,y,z,qx,qy,qz,qw]
        amplitude: translation amplitude (m)
        duration: total time (s)
        rate: sample rate (Hz)
        node: rclpy.Node (used only for timestamp/logging if publishing)
        publisher: rclpy.publisher.Publisher for Pose, or None
        frame_id: TF frame for header if publishing

    Returns:
        np.ndarray (N,7): [x,y,z,qx,qy,qz,qw] poses (if not publishing)
    """
    # --- Base pose parsing
    if isinstance(base_pose, Pose):
        p0 = np.array([
            base_pose.position.x,
            base_pose.position.y,
            base_pose.position.z,
        ])
        q0 = np.array([
            base_pose.orientation.x,
            base_pose.orientation.y,
            base_pose.orientation.z,
            base_pose.orientation.w,
        ])
    else:
        p0 = np.array(base_pose[:3])
        q0 = np.array(base_pose[3:])

    r0 = R.from_quat(q0)

    # --- Time setup
    dt = 1.0 / rate
    t = np.arange(0.0, duration, dt)

    # --- Motion parameters
    fx, fy = 3.0, 4.7
    frx, fry, frz = 2.2, 3.5, 5.1
    Arot = np.deg2rad(angle_deg)

    # --- Generate signals
    dx = amplitude * np.sin(2 * np.pi * fx * t)
    dy = amplitude * np.sin(2 * np.pi * fy * t + np.pi / 3)
    dz = -0.001 + amplitude/25 * np.sin(2 * np.pi * fy * t + np.pi / 3)

    rx = Arot * np.sin(2 * np.pi * frx * t)
    ry = Arot * np.sin(2 * np.pi * fry * t + np.pi / 4)
    rz = Arot * np.sin(2 * np.pi * frz * t + np.pi / 2)

    poses = np.zeros((len(t), 7))

    # --- Loop through samples
    for i, ti in enumerate(t):
        pos = p0 + np.array([dx[i], dy[i], dz[i]])
        quat = (R.from_euler("xyz", [rx[i], ry[i], rz[i]]) * r0).as_quat()
        poses[i, :] = np.hstack((pos, quat))

        if publisher is not None:
            msg = Pose()
            msg.position.x, msg.position.y, msg.position.z = pos
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = quat
            publisher.publish(msg)
            time.sleep(dt)

    # finally go back        
    publisher.publish(base_pose)  

    return None if publisher else poses



import numpy as np
from geometry_msgs.msg import Pose, Pose
from scipy.spatial.transform import Rotation as R
import time


def wiggle_out(
    base_pose,
    z_dist: float = 0.06,
    duration: float = 4.0,
    rate: float = 200.0,
    publisher=None,
):
    """
    Generate or publish a 4-second 'wiggle' motion around a base pose.

    Args:
        base_pose: geometry_msgs.msg.Pose or np.ndarray [x,y,z,qx,qy,qz,qw]
        amplitude: translation amplitude (m)
        duration: total time (s)
        rate: sample rate (Hz)
        node: rclpy.Node (used only for timestamp/logging if publishing)
        publisher: rclpy.publisher.Publisher for Pose, or None
        frame_id: TF frame for header if publishing

    Returns:
        np.ndarray (N,7): [x,y,z,qx,qy,qz,qw] poses (if not publishing)
    """
    # --- Base pose parsing
    if isinstance(base_pose, Pose):
        p0 = np.array([
            base_pose.position.x,
            base_pose.position.y,
            base_pose.position.z,
        ])
        q0 = np.array([
            base_pose.orientation.x,
            base_pose.orientation.y,
            base_pose.orientation.z,
            base_pose.orientation.w,
        ])
    else:
        p0 = np.array(base_pose[:3])
        q0 = np.array(base_pose[3:])

    r0 = R.from_quat(q0)

    # --- Time setup
    dt = 1.0 / rate
    t = np.arange(0.0, duration, dt)

    # --- Motion parameters
    frx, fry, frz = 1.0, 1.5, 0.0
    Arot = np.deg2rad(5)

    # --- Generate signals
    dx = 0.001
    dy = 0.001
    dz = z_dist

    rx = Arot * np.sin(2 * np.pi * frx * t)
    ry = Arot * np.sin(2 * np.pi * fry * t + np.pi / 4)
    rz = 0.0 * np.sin(2 * np.pi * frz * t + np.pi / 2)

    poses = np.zeros((len(t), 7))

    # --- Loop through samples
    for i, ti in enumerate(t):
        pos = p0 + np.array([dx, dy, dz])
        quat = (R.from_euler("xyz", [rx[i], ry[i], rz[i]]) * r0).as_quat()
        poses[i, :] = np.hstack((pos, quat))

        if publisher is not None:
            msg = Pose()
            msg.position.x, msg.position.y, msg.position.z = pos
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = quat
            publisher.publish(msg)
            time.sleep(dt)

    # finally go back  
    out_pose = base_pose
    out_pose.position.z += z_dist      
    publisher.publish(out_pose)  

    return None if publisher else poses

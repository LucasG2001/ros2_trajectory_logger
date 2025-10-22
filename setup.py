from setuptools import find_packages, setup

package_name = 'robot_trajectory_logger'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'admittance_control','custom_msgs'],
    zip_safe=True,
    maintainer='lucas',
    maintainer_email='gimenol@student.ethz.ch',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
             'robot_trajectory_logger = robot_trajectory_logger.log_data:main',
             'joint_impedance_tester = robot_trajectory_logger.joint_impedance_tester:main',
             'planner = robot_trajectory_logger.planner:main', 
             'joint_optimizer = robot_trajectory_logger.ManipulabilityOptimizer:main',
             'data_streamer = robot_trajectory_logger.data_streamer:main',
             'breakthrough_detection = robot_trajectory_logger.detection_node:main', 
             'live_plotting = robot_trajectory_logger.live_plotting:main',  
             'send_csv_positions = robot_trajectory_logger.send_pins:main',
             'Lprofiles = robot_trajectory_logger.send_Lprofiles:main',
             'Uprofiles = robot_trajectory_logger.send_uprofiles:main',
             'send_nuts = robot_trajectory_logger.send_nuts:main', 
             'calibrate_cam_aruco = robot_trajectory_logger.calibrate_cam_aruco:main',
        ],
    },
)

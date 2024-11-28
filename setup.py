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
             'planner = robot_trajectory_logger.planner:main', 
                'joint_optimizer = robot_trajectory_logger.ManipulabilityOptimizer:main',
        ],
    },
)

import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'ctr_teleop'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        (
            os.path.join(
                'share',
                package_name,
                'launch',
            ),
            glob(
                'launch/*.launch.py'
            ),
        ),

        (
            os.path.join(
                'share',
                package_name,
                'config',
            ),
            glob(
                'config/*.rviz'
            ),
        ),
    ],
    
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jackyp',
    maintainer_email='jackpeters06798@gmail.com',
    description='ROS 2 teleoperation, simulation, and RViz visualization for concentric tube robots.',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'joy_receiver = ctr_teleop.joy_receiver:main',
            'cartesian_teleop = ctr_teleop.cartesian_teleop:main',
            'ctr_simulator = ctr_teleop.ctr_simulator:main',
        ],
    },
)



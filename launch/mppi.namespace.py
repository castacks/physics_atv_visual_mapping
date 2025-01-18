from launch import LaunchDescription
from launch_ros.actions import PushRosNamespace
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    inner_launch_file = os.path.join(
        get_package_share_directory('torch_mpc'), 'mppi.launch.py'
    )
    robot_namespace = 'mppi'
    return LaunchDescription([
        PushRosNamespace(robot_namespace),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(inner_launch_file),
            launch_arguments={'robot_namespace': robot_namespace, 'headless_mode':"true"}.items(),
        ),
    ])
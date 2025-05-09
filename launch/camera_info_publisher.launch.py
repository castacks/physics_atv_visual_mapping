from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='physics_atv_visual_mapping',
            executable='camera_info_publisher',
            name='camera_info_publisher',
            output='screen',
            parameters=[{
                'use_sim_time': False
            }]
        )
    ]) 
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare the argument for the config file
    visual_mapping_config = LaunchConfiguration("visual_mapping_config")

    # Use PathJoinSubstitution to dynamically join the package path and the config file
    config_fp = PathJoinSubstitution(
        [
            FindPackageShare("physics_atv_visual_mapping"),
            "config",
            "ros",
            visual_mapping_config,
        ]
    )

    nodes = [
        # Declare the use_sim_time argument
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation (Gazebo) clock if true",
        ),
        # Declare the launch argument with a default value
        DeclareLaunchArgument(
            "visual_mapping_config",
            default_value="sam_pca.yaml",
            description="Config file for visual mapping",
        ),
        # Node definition
        Node(
            package="physics_atv_visual_mapping",
            executable="dino_localmapping",
            name="visual_localmapping",
            output="screen",
            parameters=[
                {"config_fp": config_fp},
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
                {"models_dir": LaunchConfiguration("models_dir")},
            ],
        ),
    ]

    return LaunchDescription(nodes)

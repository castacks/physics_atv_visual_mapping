from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare the argument for the config file
    visual_mapping_config = LaunchConfiguration('visual_mapping_config')
    using_zedros1_bag = False 
    use_sim = True
    # Use PathJoinSubstitution to dynamically join the package path and the config file
    config_fp = PathJoinSubstitution([
        FindPackageShare('physics_atv_visual_mapping'),
        'config',
        'ros',
        visual_mapping_config
    ])
    
    nodes = [
        # Declare the use_sim_time argument
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
        # Declare the launch argument with a default value
        DeclareLaunchArgument(
            'visual_mapping_config',
            default_value='wheelchair.yaml',
            description='Config file for visual mapping'
        ),

        # Node definition
        Node(
            package='physics_atv_visual_mapping',
            executable='dino_localmapping',
            name='visual_localmapping',
            output='screen',
            parameters=[{'config_fp': config_fp}, 
                        {'use_sim_time': LaunchConfiguration('use_sim_time')}],
        ),
        Node(
            package='physics_atv_visual_mapping',
            executable='dino_cost',
            name='dino_cost',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        ),
        # Static Transform Publisher node (ROS2 Zed Driver)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            output='screen',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
            arguments=['0', '0', '0', '0', '0', '0', '1', 'base_link', 'zed_camera_link'],
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            output='screen',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
            arguments=['0', '0', '0', '0', '0', '0', '1', 'map', 'odom'],
        ),

        # RViz node
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
            arguments=['-d', '/wheelsafe_ws/src/wheelsafe_meta/rviz/wheelie.rviz']
        ),
        
    ]
    
    if use_sim:
        # Transform Publisher for base_link given odometry
        nodes.append(Node(
            package='physics_atv_visual_mapping',
            executable='odom_to_tf',
            name='odom_to_tf',
            output='screen',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        ))
    
    if using_zedros1_bag: # Launch descriptions specific when running bags that use Zed Ros 1 driver


        
        # Static Transform Publisher node (ROS1 Zed Driver)
        nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', '1', 'base_link', 'zedx_left_camera_frame'],
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]))
        

    return LaunchDescription(nodes)

import rclpy
import ros2_numpy_cpp
import numpy as np
from rclpy.node import Node

from std_msgs.msg import Header

# Initialize ROS 2
rclpy.init()

# Create a random NumPy array (3x3 image for example)
n_voxels, n_features = 10000, 16;
features = np.ascontiguousarray(np.random.rand(n_voxels, n_features));
indices = np.ascontiguousarray(np.random.randint(0, n_voxels, size=n_voxels, dtype=np.uint64));

# Start ROS 2 node and publish NumPy array
ros2_numpy_cpp.start_ros_node()
ros2_numpy_cpp.publish_numpy(
    features, indices,
    Header(), "sensor_init",
    -60, -60, -30,
    120, 120, 60, 
    0.25, 0.25, 0.25,
    n_voxels, n_features
)

# Spin ROS 2 to keep the publisher running
rclpy.spin(Node("numpy_publisher"))

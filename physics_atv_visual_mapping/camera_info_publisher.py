#!/usr/bin/env python3
"""Publish camera_info for Zed X on Wheelie.
Only does for K and P matrices!!
"""
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from rclpy.clock import Clock

class CameraInfoPublisher(Node):
    def __init__(self):
        super().__init__('camera_info_publisher')
        
        # Create publisher for camera info
        self.camera_info_pub = self.create_publisher(
            CameraInfo,
            '/wheelie/camera_info',
            10)
        
        # Create broadcaster for camera transform
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Camera intrinsics from the config
        self.K = np.array([[366.8225402832031, 0.0, 492.81884765625],
                          [0.0, 366.8225402832031, 317.63922119140625],
                          [0.0, 0.0, 1.0]])
        
        # Create 3x4 projection matrix by adding a column of zeros
        self.P = np.hstack((self.K, np.zeros((3, 1))))
        
        
        # Set up timer for publishing
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz
        self.get_logger().info("Camera info publisher initialized")
        
    def publish_camera_info(self):
        # Create and populate CameraInfo message
        camera_info = CameraInfo()
        camera_info.header.stamp = self.get_clock().now().to_msg()
        camera_info.header.frame_id = "camera_link"
        
        # Set camera matrix (K)
        camera_info.k = self.K.flatten().tolist()
        
        # Set projection matrix (P) - 3x4 matrix
        camera_info.p = self.P.flatten().tolist()
        
        # Set distortion coefficients (assuming no distortion)
        camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Set image dimensions (assuming 1000x500 based on the focal length and principal point)
        camera_info.width = 1000
        camera_info.height = 500
        
        # Set binning and ROI
        camera_info.binning_x = 1
        camera_info.binning_y = 1
        camera_info.roi.x_offset = 0
        camera_info.roi.y_offset = 0
        camera_info.roi.height = camera_info.height
        camera_info.roi.width = camera_info.width
        camera_info.roi.do_rectify = False

        # Publish the camera info
        self.camera_info_pub.publish(camera_info)
        
    def timer_callback(self):
        self.publish_camera_info()

def main(args=None):
    rclpy.init(args=args)
    publisher = CameraInfoPublisher()
    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        pass
    finally:
        publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 
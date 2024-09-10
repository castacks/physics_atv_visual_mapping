#!/usr/bin/python3

# Given current odometry, publish the transform between the map and base_link frame
import rospy
import tf2_ros
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped

class OdometryToTf:
    def __init__(self):
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.br = tf2_ros.TransformBroadcaster()

    def odom_callback(self, odom_msg):
        # Extract the pose (position and orientation) from the odometry message
        pose = odom_msg.pose.pose

        # Create a TransformStamped message
        t = TransformStamped()

        # Set frame names
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "odom"         # Parent frame (fixed frame)
        t.child_frame_id = "base_link"    # Child frame (robot's frame)

        # Set the translation (position)
        t.transform.translation.x = pose.position.x
        t.transform.translation.y = pose.position.y
        t.transform.translation.z = pose.position.z

        # Set the rotation (orientation)
        t.transform.rotation = pose.orientation

        # Broadcast the transform
        self.br.sendTransform(t)

if __name__ == "__main__":
    rospy.init_node('odom_to_tf')
    OdometryToTf()
    rospy.spin()

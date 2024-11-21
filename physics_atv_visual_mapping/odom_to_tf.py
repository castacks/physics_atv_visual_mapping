import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class OdometryToTF(Node):
    def __init__(self):
        super().__init__("odometry_to_tf")

        # Create a transform broadcaster to publish the tf
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribe to the /odom topic
        self.odom_subscriber = self.create_subscription(
            Odometry, "/zedx/zed_node/odom", self.handle_odom, 10
        )

    def handle_odom(self, msg):
        # Create a new TransformStamped message to broadcast
        t = TransformStamped()

        # Set the header (time and frame ID)
        t.header.stamp = msg.header.stamp
        t.header.frame_id = "odom"  # Parent frame (odom)
        t.child_frame_id = "base_link"  # Child frame (base_link)

        # Use the position from the odometry message
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z

        # Use the orientation from the odometry message
        t.transform.rotation = msg.pose.pose.orientation

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)

    # Create the node
    node = OdometryToTF()

    try:
        # Keep the node running
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()

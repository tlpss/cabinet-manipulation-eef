import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class TFToTopicPublisher(Node):
    """A very simple ROS2 node that uses a TF2 transform listener to query the transform
    between two frames and publishes it as a PoseStamped message on a topic.
    This is useful when combining ROS2 with the webserver, as the webserver has no direct access
    to the TF2 library.


    parameters:
        tf2_from_frame: The frame to transform from
        tf2_to_frame: The frame to transform to
        topic_name: The name of the topic to publish the transform on

    Topics:
        'TF2_to_topic_publisher/transform': The transform between the two frames as a PoseStamped message
    """

    node_name = "TF2_to_topic_publisher"

    def __init__(self):
        super().__init__(self.node_name)
        timer_period = 0.01  # seconds
        self.from_frame = self.declare_parameter("tf2_from_frame", "base").get_parameter_value().string_value
        self.to_frame = self.declare_parameter("tf2_to_frame", "tool0").get_parameter_value().string_value
        self.topic_name = (
            self.declare_parameter("topic_name", f"{self.node_name}/transform").get_parameter_value().string_value
        )
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.publisher_ = self.create_publisher(PoseStamped, self.topic_name, 10)
        # tf2 listener based on
        #  https://docs.ros.org/en/rolling/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Listener-Py.html
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.get_logger().info(
            f"Publishing transform between {self.from_frame} and {self.to_frame} on topic {self.topic_name}"
        )

    def timer_callback(self):
        try:
            tranfsorm = self.tf_buffer.lookup_transform(self.from_frame, self.to_frame, rclpy.time.Time())
        except TransformException:
            self.get_logger().info(f"Could not look up transform between {self.from_frame} and {self.to_frame}")
            return

        msg = PoseStamped()
        msg.header.stamp = rclpy.time.Time().to_msg()
        msg.header.frame_id = self.from_frame  # reference frame
        msg.pose.position.x = tranfsorm.transform.translation.x
        msg.pose.position.y = tranfsorm.transform.translation.y
        msg.pose.position.z = tranfsorm.transform.translation.z
        msg.pose.orientation.x = tranfsorm.transform.rotation.x
        msg.pose.orientation.y = tranfsorm.transform.rotation.y
        msg.pose.orientation.z = tranfsorm.transform.rotation.z
        msg.pose.orientation.w = tranfsorm.transform.rotation.w
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    publisher = TFToTopicPublisher()

    rclpy.spin(publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

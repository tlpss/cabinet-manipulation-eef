"""script to quickly test sending and receiving messages with roslibpy.
Requires a running rosbridge server on localhost:9090.
If ROS2 runs in a docker container, make sure to share the network (or expose the port).
"""

from __future__ import print_function

import threading
import time
from typing import Optional

from roslibpy import Message, Ros, Time, Topic
from roslibpy.core import UserDict


class ROS2Header(UserDict):
    """Represents a message header of the ROS 2 type std_msgs/Header."""

    def __init__(self, stamp: Optional[Time] = None, frame_id: Optional[str] = None):
        self.data = {}
        self.data["stamp"] = Time(stamp["secs"], stamp["nsecs"]) if stamp else None
        self.data["frame_id"] = frame_id


def test_topic_with_header() -> None:
    context = dict(wait=threading.Event())

    ros = Ros("127.0.0.1", 9090)
    ros.run()

    listener = Topic(ros, "/points", "geometry_msgs/PointStamped")
    publisher = Topic(ros, "/points", "geometry_msgs/PointStamped")

    def receive_message(message: dict) -> None:
        print(message)
        assert message["header"]["frame_id"] == "base"
        assert message["point"]["x"] == 0.0
        assert message["point"]["y"] == 1.0
        assert message["point"]["z"] == 2.0
        listener.unsubscribe()
        context["wait"].set()

    def start_sending() -> None:
        for i in range(3):
            msg = dict(header=dict(ROS2Header(frame_id="base", stamp=Time.now())), point=dict(x=0.0, y=1.0, z=2.0))
            print(msg.__repr__())
            publisher.publish(Message(msg))
            time.sleep(0.1)

        publisher.unadvertise()

    def start_receiving() -> None:
        listener.subscribe(receive_message)

    t1 = threading.Thread(target=start_receiving)
    t2 = threading.Thread(target=start_sending)

    t1.start()
    t2.start()

    if not context["wait"].wait(10):
        raise Exception

    t1.join()
    t2.join()

    ros.close()


if __name__ == "__main__":
    test_topic_with_header()

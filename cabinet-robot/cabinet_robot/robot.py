from typing import Optional

import roslibpy
from airo_spatial_algebra import SE3Container
from airo_typing import HomogeneousMatrixType
from roslibpy import Message, Ros, Service, Topic
from roslibpy.core import UserDict


class ROS2Header(UserDict):
    """Represents a message header of the ROS 2 type std_msgs/Header."""

    def __init__(self, stamp: roslibpy.Time, frame_id: Optional[str] = None):
        self.data = {}
        self.data["stamp"] = stamp
        self.data["frame_id"] = frame_id


class FZIControlledRobot:
    """class for controlling UR robots over ROS with the FZI controllers.
    The class uses the ROS2 Webserver to communicate over ROS without the need for a ROS2 installation.
    ROS can hence run in a docker container (or a remote machine).

    The class also uses the RTDE interface of the UR to read other values from the robot, as this is easier than using the ROS Topics.
    """

    # controller names as defined in the ros launch/config files of the FZI controllers
    FZI_MOTION_CONTROLLER_NAME = "cartesian_motion_controller"
    FZI_FORCE_CONTROLLER_NAME = "cartesian_force_controller"
    FZI_ADMITTANCE_CONTROLLER_NAME = "cartesian_compliance_controller"

    POSITION_CONTROL_TARGET_POSE_TOPIC_NAME = "/target_pose"
    ADMITTANCE_CONTROL_TARGET_POSE_TOPIC_NAME = "/target_pose"
    WRENCH_TOPIC_NAME = "/force_torque_sensor_broadcaster/wrench"

    CONTROL_FRAME = "base"

    def __init__(self, ros2_ip: str = "127.0.0.1", ros2_port: int = 9090):
        self._ros = Ros(ros2_ip, ros2_port)
        self._ros.run()
        self._admittance_target_pose_publisher = Topic(
            self._ros, self.ADMITTANCE_CONTROL_TARGET_POSE_TOPIC_NAME, "geometry_msgs/PoseStamped"
        )
        self._motion_target_pose_publisher = Topic(
            self._ros, self.POSITION_CONTROL_TARGET_POSE_TOPIC_NAME, "geometry_msgs/PoseStamped"
        )
        self._wrench_listener = Topic(
            self._ros, "/force_torque_sensor_broadcaster/wrench", "geometry_msgs/WrenchStamped"
        )

        self.active_controller = None

        self.wrench_thread = self._wrench_listener.subscribe(self._wrench_callback)
        self._latest_wrench = None

        self.control_manager_service = Service(
            self._ros, "/controller_manager/switch_controller", "controller_manager_msgs/SwitchController"
        )

        self.switch_to_position_control()

    def _wrench_callback(self, message):
        self._latest_wrench = np.array(list(message["wrench"].values()))

    def get_eef_pose(self):
        # TODO: create a ros node that offers a service from wich the pose can be read
        # as the tf2 library is not available with roslibpy
        # and then query that service here
        raise NotImplementedError

    def get_wrench(self):
        return self._latest_wrench

    def servo_to_pose(self, pose: HomogeneousMatrixType) -> None:
        if not self.active_controller == self.FZI_ADMITTANCE_CONTROLLER_NAME:
            RuntimeWarning("Ignoring servo target as the robot is not in admittance control mode.")
            return
        message = self._create_stamped_pose_message_from_pose(pose)
        self._admittance_target_pose_publisher.publish(Message(message))

    def move_to_pose(self, pose):
        if not self.active_controller == self.FZI_MOTION_CONTROLLER_NAME:
            print("switching to position control")
            self.switch_to_position_control()
        message = self._create_stamped_pose_message_from_pose(pose)
        self._motion_target_pose_publisher.publish(Message(message))

    def _create_stamped_pose_message_from_pose(self, pose: HomogeneousMatrixType) -> Message:
        quaternion = SE3Container.from_homogeneous_matrix(pose).orientation_as_quaternion
        message = dict(header=dict(ROS2Header(frame_id=self.control_frame, stamp=roslibpy.Time.now())))
        message["pose"] = dict(
            position=dict(x=pose[0, 3], y=pose[1, 3], z=pose[2, 3]),
            orientation=dict(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]),
        )
        return Message(message)

    def switch_to_admittance_control(self):
        request = roslibpy.ServiceRequest(
            {
                "start_controllers": [self.FZI_ADMITTANCE_CONTROLLER_NAME],
                "stop_controllers": [self.FZI_FORCE_CONTROLLER_NAME, self.FZI_MOTION_CONTROLLER_NAME],
                "strictness": 2,
            }
        )
        self.control_manager_service.call(request, callback=None, timeout=10)
        self.active_controller = self.FZI_ADMITTANCE_CONTROLLER_NAME

    def switch_to_position_control(self):
        request = roslibpy.ServiceRequest(
            {
                "start_controllers": ["cartesian_motion_controller"],
                "stop_controllers": ["cartesian_force_controller", "cartesian_compliance_controller"],
                "strictness": 2,
            }
        )
        self.control_manager_service.call(request, callback=None, timeout=10)
        self.active_controller = self.FZI_MOTION_CONTROLLER_NAME

    def switch_to_force_control(self):
        request = roslibpy.ServiceRequest(
            {
                "start_controllers": ["cartesian_force_controller"],
                "stop_controllers": ["cartesian_motion_controller", "cartesian_compliance_controller"],
                "strictness": 2,
            }
        )
        self.control_manager_service.call(request, callback=None, timeout=10)
        self.active_controller = self.FZI_FORCE_CONTROLLER_NAME


### OPENING CABINETS

# start in position control mode

# move to handle pregrasp pose
# switch to admittance
# grasp
# set the initial cabinet joint estimate to be a revolute joint between the handle and the robot base

# while not open ()
# for N steps
# move along that joint estimate in increments of 2cm.
# if the force becomes too high or the gripper no longer has contact, stop
# collect the gripper pose
# make new estimate of the articulation


if __name__ == "__main__":
    import time

    import numpy as np

    robot = FZIControlledRobot()
    pose = SE3Container.from_euler_angles_and_translation([np.pi, 0, 0], [0.2, -0.3, 0.3]).homogeneous_matrix
    robot.servo_to_pose(pose)
    # print(robot.get_wrench())
    time.sleep(4)

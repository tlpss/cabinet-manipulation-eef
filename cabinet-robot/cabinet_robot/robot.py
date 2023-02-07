import typing
import warnings
from typing import Optional

import roslibpy
from airo_spatial_algebra import SE3Container
from airo_typing import HomogeneousMatrixType
from roslibpy import Message, Ros, Service, Topic
from roslibpy.core import UserDict


class ROS2Header(UserDict):
    """Represents a message header of the ROS 2 type std_msgs/Header. For use with the ROS2 Webserver."""

    def __init__(self, stamp: roslibpy.Time, frame_id: Optional[str] = None):
        self.data = {}
        self.data["stamp"] = stamp
        self.data["frame_id"] = frame_id


class FZIControlledRobot:
    """class for controlling UR robots over ROS with the FZI controllers.
    The class uses the ROS2 Webserver to communicate to the ROS nodes without the need for a local ROS2 installation (ROS2 can run in a docker container)
    and/or without having to wrap your codebase in (a) ROS2 node(s).

    The communication stack is as follows:
    python <--JSON--> ROS2 Webserver <--ROS messages--> ROS2 <--RTDE--> UR robot
    """

    # controller names as defined in the ros launch/config files of the FZI controllers
    FZI_MOTION_CONTROLLER_NAME = "cartesian_motion_controller"
    FZI_FORCE_CONTROLLER_NAME = "cartesian_force_controller"
    FZI_ADMITTANCE_CONTROLLER_NAME = "cartesian_compliance_controller"

    FZI_CONTROLLERS = [FZI_MOTION_CONTROLLER_NAME, FZI_FORCE_CONTROLLER_NAME, FZI_ADMITTANCE_CONTROLLER_NAME]

    POSITION_CONTROL_TARGET_POSE_TOPIC_NAME = "/target_pose"
    ADMITTANCE_CONTROL_TARGET_POSE_TOPIC_NAME = "/target_pose"
    WRENCH_TOPIC_NAME = "/force_torque_sensor_broadcaster/wrench"

    CONTROL_FRAME = "base"
    TCP_FRAME = "tool0"  # TODO: use TCP frame, by adding it to the URDF

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
        self._wrench_listener.subscribe(self._wrench_callback)
        self._latest_wrench = None

        self._tcp_pose_listener = Topic(self._ros, "ur_tcp_pose", "geometry_msgs/PoseStamped")
        self._tcp_pose_listener.subscribe(self._tcp_pose_callback)
        self._latest_tcp_pose = None

        self.control_manager_switch_service = Service(
            self._ros, "/controller_manager/switch_controller", "controller_manager_msgs/SwitchController"
        )
        self.control_manager_list_controllers_service = Service(
            self._ros, "/controller_manager/list_controllers", "controller_manager_msgs/ListControllers"
        )

        self.active_controller = self._get_active_FZI_controller()

    def _wrench_callback(self, message):
        # TODO: fix this to return a proper airo-typing wrench-like object.
        # self._latest_wrench = np.array(list(message["wrench"].values()))
        raise NotImplementedError

    def _tcp_pose_callback(self, message):
        self._latest_tcp_pose = message["pose"]

    def get_tcp_pose(self):
        position = np.array(list(self._latest_tcp_pose["position"].values()))
        orientation = np.array(list(self._latest_tcp_pose["orientation"].values()))
        return SE3Container.from_quaternion_and_translation(orientation, position).homogeneous_matrix

    def get_wrench(self):
        return self._latest_wrench

    def servo_to_pose(self, pose: HomogeneousMatrixType) -> None:
        if self.active_controller not in (self.FZI_ADMITTANCE_CONTROLLER_NAME, self.FZI_MOTION_CONTROLLER_NAME):
            warnings.warn("Ignoring servo target as nor the admittance nor the motion controller is active.")
            return
        message = self._create_stamped_pose_message_from_pose(pose)
        self._admittance_target_pose_publisher.publish(Message(message))

    def move_to_pose(self, pose: HomogeneousMatrixType):
        if not self.active_controller == self.FZI_MOTION_CONTROLLER_NAME:
            print("switching to position control")
            self.switch_to_position_control()
        message = self._create_stamped_pose_message_from_pose(pose)
        self._motion_target_pose_publisher.publish(Message(message))

        waiting_time = 0.0
        # TODO: this still times out sometimes, even though the robot is already at the target pose.
        while np.linalg.norm(self.get_tcp_pose()[:3, 3] - pose[:3, 3]) > 0.02:
            time.sleep(0.1)
            waiting_time += 0.1
            if waiting_time > 10:
                warnings.warn("Waiting for robot to reach target pose timed out.")
                break
        return

    def _create_stamped_pose_message_from_pose(self, pose: HomogeneousMatrixType) -> Message:
        quaternion = SE3Container.from_homogeneous_matrix(pose).orientation_as_quaternion
        message = dict(header=dict(ROS2Header(frame_id=self.CONTROL_FRAME, stamp=roslibpy.Time.now())))
        message["pose"] = dict(
            position=dict(x=pose[0, 3], y=pose[1, 3], z=pose[2, 3]),
            orientation=dict(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]),
        )
        return Message(message)

    def switch_to_admittance_control(self):
        self._switch_controllers(self.FZI_ADMITTANCE_CONTROLLER_NAME)

    def switch_to_position_control(self):
        self._switch_controllers(self.FZI_MOTION_CONTROLLER_NAME)

    def switch_to_force_control(self):
        self._switch_controllers(self.FZI_FORCE_CONTROLLER_NAME)

    def _get_active_FZI_controller(self) -> typing.Union[None, str]:
        request = roslibpy.ServiceRequest({})
        response = self.control_manager_list_controllers_service.call(request, callback=None, timeout=10)
        controllers = response["controller"]
        active_controllers = [controller["name"] for controller in controllers if controller["state"] == "active"]
        active_FZI_controllers = [
            controller for controller in active_controllers if controller in self.FZI_CONTROLLERS
        ]
        if len(active_FZI_controllers) > 1:
            raise RuntimeError(
                "More than one FZI controller is active, this should not be possible as they share the HW interface"
            )

        active_FZI_controller = active_FZI_controllers[0] if len(active_FZI_controllers) > 0 else None
        return active_FZI_controller

    def _switch_controllers(self, controller_to_start: str):
        assert controller_to_start in self.FZI_CONTROLLERS, f"Unknown controller {controller_to_start}"
        active_controller = self._get_active_FZI_controller()

        if active_controller:
            controllers_to_stop = [active_controller]
        else:
            controllers_to_stop = []

        controllers_to_start = [controller_to_start]
        request = roslibpy.ServiceRequest(
            {
                "start_controllers": controllers_to_start,
                "stop_controllers": controllers_to_stop,
                "strictness": 2,
            }
        )
        self.control_manager_switch_service.call(request, callback=None, timeout=10)
        self.active_controller = controller_to_start


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
    pose = SE3Container.from_euler_angles_and_translation([0, np.pi, 0], [0.2, -0.3, 0.3]).homogeneous_matrix
    robot.move_to_pose(pose)
    print(robot.get_wrench())
    robot.switch_to_admittance_control()
    print(robot._get_active_FZI_controller())
    # time.sleep(4)
    pose[0, 3] += 0.1
    robot.servo_to_pose(pose)
    time.sleep(10)
    pose[1, 3] += 0.1
    robot.move_to_pose(pose)
    print(robot.get_tcp_pose())

import time
import typing
import warnings
from typing import Optional

import numpy as np
import roslibpy
from airo_spatial_algebra import SE3Container
from airo_typing import HomogeneousMatrixType, WrenchType
from roslibpy import Message, Ros, Service, Topic
from roslibpy.core import UserDict
from spatialmath import base as sm


class ROS2Header(UserDict):
    """Represents a message header of the ROS 2 type std_msgs/Header. For use with the ROS2 Webserver."""

    def __init__(self, stamp: roslibpy.Time, frame_id: Optional[str] = None):
        self.data = {}
        self.data["stamp"] = stamp
        self.data["frame_id"] = frame_id


class FZIControlledRobot:
    """class for controlling UR robots over ROS with the FZI controllers as configured in the `ure_cartesian_controllers` ROS package.
    The class uses the ROS2 Webserver to communicate to the ROS nodes without the need for a local ROS2 installation (ROS2 can run in a docker container)
    and/or without having to wrap your codebase in (a) ROS2 node(s).
    The communication stack is as follows:
    python <--JSON--> ROS2 Webserver <--ROS messages--> ROS2 <--RTDE--> UR robot

    The FZI controllers are configured to work with the ROS tool0 frame (which corresponds to TCP frame of the TCP offsets are all set to zero in the control box)
    This class will convert these poses to the TCP frame using the manually defined Z-offset. It would be more elegant to add this offset to the URDF and have the FZI controllers work in that frame
    directly.
    """

    # TODO: implement the airo-robots async interface once it is stable.

    # controller names as defined in the ros launch/config files of the FZI controllers
    FZI_MOTION_CONTROLLER_NAME = "cartesian_motion_controller"
    FZI_FORCE_CONTROLLER_NAME = "cartesian_force_controller"
    FZI_ADMITTANCE_CONTROLLER_NAME = "cartesian_compliance_controller"

    FZI_CONTROLLERS = [FZI_MOTION_CONTROLLER_NAME, FZI_FORCE_CONTROLLER_NAME, FZI_ADMITTANCE_CONTROLLER_NAME]

    POSITION_CONTROL_TARGET_POSE_TOPIC_NAME = "/target_pose"
    ADMITTANCE_CONTROL_TARGET_POSE_TOPIC_NAME = "/target_pose"
    WRENCH_TOPIC_NAME = "/force_torque_sensor_broadcaster/wrench"

    CONTROL_FRAME = "base"
    FLANGE_FRAME = "tool0"

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
        self._latest_wrench = message["wrench"]

    def _tcp_pose_callback(self, message):
        self._latest_tcp_pose = message["pose"]

    def get_tcp_pose(self) -> Optional[HomogeneousMatrixType]:
        if self._latest_tcp_pose is None:
            warnings.warn("No TCP pose received yet. Returning None.")
            return None

        position = np.array(list(self._latest_tcp_pose["position"].values()))
        orientation = np.array(list(self._latest_tcp_pose["orientation"].values()))
        return SE3Container.from_quaternion_and_translation(orientation, position).homogeneous_matrix

    def get_wrench(self) -> Optional[WrenchType]:
        """returns the latest wrench as measured in the sensor frame, and expressed in that sensor frame."""
        # TODO: this wrench is expressed in the sensor frame. The native UR command returns the wrench expressed in the base frame.
        # should do this as well by using the adjoint matrix of the current sensor (flange) pose.

        if self._latest_wrench is None:
            warnings.warn("No wrench received yet. Returning None.")
            return None
        force = np.array(list(self._latest_wrench["force"].values()))
        torque = np.array(list(self._latest_wrench["torque"].values()))
        return np.concatenate([force, torque])

    def set_target_pose(self, tcp_pose: HomogeneousMatrixType) -> None:
        """sets a new target pose for the active controller. The method is asynchronous and will return immediately.
        If the admittance controller is active, the robot will try to reach this new setpoint while behaving as the mass-spring-damper system defined by the admittance parameters.
        If the motion controller is active, the result will be similar to calling the servoJ/L method of the UR due to the inner workings of the motion controller.

        Args:
            tcp_pose (HomogeneousMatrixType): the target pose of the TCP in the base frame.

        Returns:
            None
        """

        if self.active_controller not in (self.FZI_ADMITTANCE_CONTROLLER_NAME, self.FZI_MOTION_CONTROLLER_NAME):
            warnings.warn("Ignoring servo target as nor the admittance nor the motion controller is active.")
            return
        # normalize the SE3 to avoid errors in the spatialmath lib
        tcp_pose = sm.trnorm(tcp_pose)

        message = self._create_stamped_pose_message_from_pose(tcp_pose)
        self._admittance_target_pose_publisher.publish(Message(message))

    def move_to_pose(self, tcp_pose: HomogeneousMatrixType) -> None:
        """Moves the robot to the given TCP pose. The method is synchronous and will block until the robot has reached the target pose.
        under the hood, the method activates the FZI motion controller and publishes the target pose to the corresponding topic.
        The FZI motion controller will interpolate the target pose if required so that the velocity is somewhat constant, but it does not apply a strict velocity profile such as the
        UR's native MoveL function.

        Args:
            tcp_pose (HomogeneousMatrixType): the target pose of the TCP in the base frame.
        """

        if not self.active_controller == self.FZI_MOTION_CONTROLLER_NAME:
            print("switching to position control")
            self.switch_to_position_control()

        message = self._create_stamped_pose_message_from_pose(tcp_pose)
        self._motion_target_pose_publisher.publish(Message(message))

        waiting_time = 0.0
        while np.linalg.norm(self.get_tcp_pose()[:3, 3] - tcp_pose[:3, 3]) > 0.005:
            time.sleep(0.1)
            waiting_time += 0.1
            if waiting_time > 10:
                warnings.warn("Waiting for robot to reach target pose timed out.")
                break
        return

    def _create_stamped_pose_message_from_pose(self, pose: HomogeneousMatrixType) -> Message:
        """helper function to publish target poses to the FZI controllers over the webbridge."""
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
        """returns the name of the active FZI controller, or None if no FZI controller is active."""
        request = roslibpy.ServiceRequest({})
        attempts = 3
        for attempt in range(attempts):
            try:
                response = self.control_manager_list_controllers_service.call(request, callback=None, timeout=10)
            except roslibpy.core.ServiceException:
                warnings.warn(f"could not list controllers in the {attempt}/{attempts} th attempt")
                time.sleep(0.5)
                if attempt == attempts - 1:
                    raise ConnectionError("Could not query the active FZI controller.")

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
        """switches to the desired controller. If another FZI controller is active, it will be stopped simultaneously.
        uses the controller_manager service to accomplish this.

        Args:
            controller_to_start (str): the name of the controller to start. Must be one of the FZI_CONTROLLERS.

        Raises:
            ConnectionError: if the service call to the controller_manager service fails.
        """
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
        attempts = 3
        for attempt in range(attempts):
            try:
                self.control_manager_switch_service.call(request, callback=None, timeout=10)
            except roslibpy.core.ServiceException:
                warnings.warn(f"could not switch controllers in the {attempt}/{attempts} th attempt ")
                time.sleep(0.5)
                if attempt == attempts - 1:
                    raise ConnectionError("Could not query the active FZI controller.")

        self.active_controller = controller_to_start


if __name__ == "__main__":
    robot = FZIControlledRobot()
    print(robot._get_active_FZI_controller())
    robot.switch_to_admittance_control()
    # pose = SE3Container.from_euler_angles_and_translation([0, np.pi, 0], [0.2, -0.3, 0.1]).homogeneous_matrix
    # print(robot.get_tcp_pose())
    # print(robot.get_wrench())
    # robot.move_to_pose(pose)
    # print(robot.get_wrench())
    # robot.switch_to_admittance_control()
    # print(robot._get_active_FZI_controller())
    # # time.sleep(4)
    # pose[0, 3] += 0.1
    # robot.set_target_pose(pose)
    # time.sleep(10)
    # pose[1, 3] += 0.1
    # robot.move_to_pose(pose)
    # print(robot.get_tcp_pose())

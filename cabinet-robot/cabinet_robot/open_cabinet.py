import time
from datetime import datetime
from typing import List, Union

import numpy as np
import rerun
import spatialmath.base as sm
from airo_camera_toolkit.cameras.zed2i import Zed2i
from airo_robots.grippers.hardware.robotiq_2f85_tcp import Robotiq2F85
from airo_spatial_algebra import SE3Container
from airo_typing import HomogeneousMatrixType, TwistType
from cabinet_robot.camera_calibration_utils import get_camera_pose_in_robot_frame
from cabinet_robot.joint_estimation import FGJointEstimator
from cabinet_robot.robot import FZIControlledRobot
from cabinet_robot.visualisation import visualize_estimation

rerun.init(f"cabinet-opener-{datetime.now()}", spawn=True)


class CabinetOpener:
    """class to open a cabinet door with a robot by using the TCP poses to perform runtime joint estimation configuration with factor graphs"""

    def __init__(self, robot: FZIControlledRobot, gripper):
        self.robot = robot
        self.gripper = gripper
        self.gripper_poses = []
        self.n_steps = 10
        self.joint_configuration_step_delta = 0.01
        self.initial_gripper_pose = None
        self.estimation_results = None
        self.visualize = True
        self.camera = Zed2i(resolution=Zed2i.RESOLUTION_720, depth_mode=Zed2i.NEURAL_DEPTH_MODE)
        self.camera.runtime_params.texture_confidence_threshold = 100
        self.camera.runtime_params.confidence_threshold = 100

        self.FG_estimator = FGJointEstimator()

    def open_grasped_cabinet(self):

        self.initial_gripper_pose = self.robot.get_tcp_pose()

        # determine initial joint estimation
        # first try: as a revolute joint between the handle and the robot base
        # now: emulate the normals by assuming that the grasp orientation is such that -Z is away from the door.
        self.gripper_poses.append(self.initial_gripper_pose)

        # estimated_twist_translation = -SE3Container.from_homogeneous_matrix(self.initial_gripper_pose).translation
        estimated_twist_translation = np.array(
            [0, 0, -1.0]
        )  # TODO: this is a hack, should be determined from the door plane normal?
        estimated_twist_rotation = np.zeros(3)
        estimated_twist: TwistType = np.concatenate((estimated_twist_translation, estimated_twist_rotation))
        twist_in_base_pose = self.initial_gripper_pose
        q_joint = 0.0
        while not self._is_cabinet_open() and self._is_grasped_heuristic():
            check = input("check if the joint estimation is not crazy. Press a key to continue or CTRL+C to abort")
            print(check)
            # safety check - is robot controller still up and running?
            if not self.robot._get_active_FZI_controller() == self.robot.FZI_ADMITTANCE_CONTROLLER_NAME:
                raise Exception("Admittance controller is not running")

            # if there hase been a joint estimation (always except for the first iteration)
            # determine an appropriate step_size (both magnitude and direction)
            if self.estimation_results is not None:
                estimated_joint_states = self.estimation_results.aux_data["joint_states"]
                step_size_direction = np.sign(estimated_joint_states[-1] - estimated_joint_states[0])
                # TODO: should this only depend on the norm of the twist?
                # use fixed value for now.
                self.joint_configuration_step_delta = 0.005 * step_size_direction

            # start compiling the factor graph
            self.FG_estimator._build_graph()
            for i in range(self.n_steps):
                # take a small step in the joint configuration
                q_joint = q_joint + self.joint_configuration_step_delta

                # TODO: add safety check with distance between current pose and new setpoint pose. If distance is too large, stop
                new_setpoint_pose = self.get_gripper_pose_from_joint_q_and_twist(
                    q_joint, twist_in_base_pose, estimated_twist
                )
                if self.visualize:
                    rerun.log_point(
                        "world/robot_setpoint", new_setpoint_pose[:3, 3], color=(255, 255, 0), radius=0.005
                    )

                self.robot.set_target_pose(new_setpoint_pose)
                # wait for robot to reach the setpoint
                # TODO: check for changes in the TCP pose to determine if the robot has reached a stable pose
                time.sleep(3.0)

                # if force has become too high or gripper no longer has contact, stop
                if self._is_force_too_high() or not self._is_grasped_heuristic():
                    raise Exception("Cabinet could not be opened")
                # collect the gripper pose
                self.gripper_poses.append(self.robot.get_tcp_pose())

            if self.visualize:
                rerun.log_points(
                    "world/gripper_poses",
                    positions=np.array(self.gripper_poses)[:, :3, 3],
                    colors=np.zeros((len(self.gripper_poses), 3), dtype=np.uint8),
                    radii=0.01,
                )

            # make new estimate of the articulation
            # TODO: should all points be used? maybe after a certain number of points it is good enough to sample only a subset of the points?
            estimated_twist, twist_in_base_pose, q_joint = self.estimate_twist(self.gripper_poses)

    def _is_grasped_heuristic(self) -> bool:
        # TODO: check gripper and check if force is not zero
        return True

    def _is_force_too_high(self) -> bool:
        # TODO: check force
        return False

    def _is_cabinet_open(self) -> bool:
        # TODO: how to measure? buildup in force could indicate this but that is maybe not a perfect heuristic?
        # set a q_max? but this highly depends on the twist ofc. should the twist be normalized then?
        return False

    def estimate_twist(self, poses: List[HomogeneousMatrixType]) -> Union[TwistType, HomogeneousMatrixType, float]:
        estimation_results = FG_twist_estimation(poses, 0.005, 0.05)
        self.estimation_results = estimation_results
        twist = np.asarray(estimation_results.twist)
        twist_frame_in_base_pose = np.asarray(estimation_results.twist_frame_in_base_pose.as_matrix())

        if self.visualize:
            visualize_estimation(estimation_results)
        return twist, twist_frame_in_base_pose, estimation_results.current_joint_configuration

    def get_gripper_pose_from_joint_q_and_twist(
        self, q: float, twist_in_base_pose: HomogeneousMatrixType, twist: TwistType
    ) -> HomogeneousMatrixType:
        # spatialmath lib uses rotation first, so we need to roll the array to get the rotation first
        # as our convention is rotation last
        pose = twist_in_base_pose @ sm.trexp(q * twist)
        return pose

    def grasp_cabinet_handle(self, grasp_pose: HomogeneousMatrixType) -> None:
        self.robot.move_to_pose(grasp_pose)
        self.robot.switch_to_admittance_control()
        time.sleep(2.0)
        self.gripper.close()

    def log_pointcloud(self):
        rgb = self.camera.get_rgb_image()
        depth = self.camera.get_depth_image()
        rerun.log_image("world/camera/rgb", image=rgb)

        rerun.log_image("camera/depth", image=depth)
        rerun.log_image("camera/rgb", image=rgb)

        pointcloud = self.camera.get_colored_point_cloud()
        rerun.log_points("world/camera/pointcloud", positions=pointcloud[:, :3], colors=pointcloud[:, 3:])
        camera_pose_in_world = get_camera_pose_in_robot_frame()
        se3_container = SE3Container.from_homogeneous_matrix(camera_pose_in_world)
        rerun.log_rigid3(
            "world/camera", child_from_parent=(se3_container.translation, se3_container.orientation_as_quaternion)
        )
        rerun.log_pinhole(
            "world/camera/rgb",
            child_from_parent=self.camera.intrinsics_matrix,
            width=rgb.shape[1],
            height=rgb.shape[0],
        )


if __name__ == "__main__":
    robot_ip = "10.42.0.162"

    robot = FZIControlledRobot()
    gripper = Robotiq2F85(robot_ip)
    gripper.open()
    gripper.speed = gripper.gripper_specs.min_speed  # so that closing is not too fast and admittance can keep up
    cabinet_opener = CabinetOpener(robot, gripper)
    home_pose = SE3Container.from_euler_angles_and_translation(
        np.array([0, np.pi / 2, 0]), np.array([0.5, -0.2, 0.2])
    ).homogeneous_matrix
    robot.move_to_pose(home_pose)
    cabinet_opener.log_pointcloud()

    handle_pose = SE3Container.from_euler_angles_and_translation(
        np.array([0, np.pi / 2, 0]), np.array([0.678, -0.180, 0.193])
    ).homogeneous_matrix
    cabinet_opener.grasp_cabinet_handle(handle_pose)
    cabinet_opener.open_grasped_cabinet()

    robot.move_to_pose(home_pose)

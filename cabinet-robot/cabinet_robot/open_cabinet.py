import time
from typing import List

import numpy as np
import rerun
import spatialmath.base as sm
from airo_robots.grippers.hardware.robotiq_2f85_tcp import Robotiq2F85
from airo_spatial_algebra import SE3Container
from airo_typing import HomogeneousMatrixType, TwistType
from cabinet_robot.joint_estimation import FG_twist_estimation
from cabinet_robot.robot import FZIControlledRobot

rerun.init("cabinet-opener", spawn=True)


class CabinetOpener:
    """class to open a cabinet door with a robot by using the TCP poses to perform runtime joint estimation configuration with factor graphs"""

    def __init__(self, robot: FZIControlledRobot, gripper):
        self.robot = robot
        self.gripper = gripper
        self.gripper_poses = []
        self.n_steps = 10
        self.step_size = 0.02
        self.initial_gripper_pose = None

    def open_grasped_cabinet(self):

        self.initial_gripper_pose = self.robot.get_tcp_pose()
        # determine initial joint estimation
        # as a revolute joint between the handle and the robot base
        self.gripper_poses.append(self.initial_gripper_pose)
        estimated_twist_translation = -SE3Container.from_homogeneous_matrix(self.initial_gripper_pose).translation
        estimated_twist_rotation = np.zeros(3)
        estimated_twist: TwistType = np.concatenate((estimated_twist_translation, estimated_twist_rotation))

        while not self._is_cabinet_open() and self._is_grasped_heuristic():
            # determine current joint configuration (the q of the Twist)
            q_joint = self.get_joint_q_from_gripper_pose_and_twist(
                self.initial_gripper_pose, self.robot.get_tcp_pose(), estimated_twist
            )
            for i in range(self.n_steps):
                # take a small step in the joint configuration
                q_joint = q_joint + self.step_size
                new_setpoint_pose = self.get_gripper_pose_from_joint_q_and_twist(
                    q_joint, self.initial_gripper_pose, estimated_twist
                )
                self.robot.set_target_pose(new_setpoint_pose)
                # wait for robot to reach the setpoint
                time.sleep(
                    3.0
                )  # TODO: check for changes in the TCP pose to determine if the robot has reached a stable pose

                # if force has become too high or gripper no longer has contact, stop
                if self._is_force_too_high() or not self._is_grasped_heuristic():
                    raise Exception("Cabinet could not be opened")
                # collect the gripper pose
                self.gripper_poses.append(self.robot.get_tcp_pose())

            # make new estimate of the articulation
            rerun.log_points(
                "gripper_poses",
                positions=np.array(self.gripper_poses)[:, :3, 3],
                colors=np.zeros((len(self.gripper_poses), 3), dtype=np.uint8),
                radii=0.01,
            )
            estimated_twist = self.estimate_twist(self.gripper_poses)

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

    def estimate_twist(self, poses: List[HomogeneousMatrixType]) -> TwistType:
        twist_joint_params, aux_data = FG_twist_estimation(poses, 0.005, 0.05)
        twist = twist_joint_params.twist
        twist_frame = twist_joint_params.base_transform
        twist_in_poses_frame = sm.tr2adjoint(np.asarray(twist_frame.as_matrix())) @ np.asarray(twist)
        return twist_in_poses_frame

    def get_joint_q_from_gripper_pose_and_twist(
        self,
        initial_gripper_pose: HomogeneousMatrixType,
        current_gripper_pose: HomogeneousMatrixType,
        joint_twist: TwistType,
    ) -> float:
        current_twist = sm.trlog(np.linalg.inv(initial_gripper_pose) @ current_gripper_pose, twist=True)

        # if the joint twist component is zero, we cannot divide by it and we consider the current twist value to be numerical errors if it is not equal to zero.
        # so we set it to a very high value, to basically mask it out in the calculation of q
        joint_twist = np.copy(joint_twist)  # make copy before modifying
        joint_twist[np.where(joint_twist == 0)] = 1e6
        q = current_twist / joint_twist
        q = np.mean(q)  # average over numerical errors
        # TODO: check if the errors are not too high?
        return q

    def get_gripper_pose_from_joint_q_and_twist(
        self, q: float, initial_gripper_pose: HomogeneousMatrixType, twist: TwistType
    ) -> HomogeneousMatrixType:
        # spatialmath lib uses rotation first, so we need to roll the array to get the rotation first
        # as our convention is rotation last
        pose = initial_gripper_pose @ sm.trexp(q * twist)
        return pose

    def grasp_cabinet_handle(self, grasp_pose: HomogeneousMatrixType) -> None:
        self.robot.move_to_pose(grasp_pose)
        self.robot.switch_to_admittance_control()
        time.sleep(2.0)
        self.gripper.close()


if __name__ == "__main__":
    robot_ip = "10.42.0.162"

    robot = FZIControlledRobot()
    gripper = Robotiq2F85(robot_ip)
    gripper.open()
    gripper.speed = gripper.gripper_specs.min_speed  # so that closing is not too fast and admittance can keep up

    cabinet_opener = CabinetOpener(robot, gripper)
    home_pose = SE3Container.from_euler_angles_and_translation(
        np.array([0, np.pi / 2, 0]), np.array([0.4, -0.2, 0.2])
    ).homogeneous_matrix
    robot.move_to_pose(home_pose)

    handle_pose = SE3Container.from_euler_angles_and_translation(
        np.array([0, np.pi / 2, 0]), np.array([0.678, -0.180, 0.193])
    ).homogeneous_matrix
    cabinet_opener.grasp_cabinet_handle(handle_pose)
    cabinet_opener.open_grasped_cabinet()

    robot.move_to_pose(home_pose)

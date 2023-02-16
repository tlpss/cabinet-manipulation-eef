from typing import List

import numpy as np
import spatialmath.base as sm
from airo_spatial_algebra import SE3Container
from airo_typing import HomogeneousMatrixType, TwistType
from cabinet_robot.joint_estimation import FG_twist_estimation
from cabinet_robot.robot import FZIControlledRobot


class CabinetOpener:
    """class to open a cabinet door with a robot by using the TCP poses to perform runtime joint estimation configuration with factor graphs"""

    def __init__(self, robot: FZIControlledRobot):
        self.robot = robot
        self.gripper = None  # TODO: make gripper object
        self.gripper_poses = []
        self.n_steps = 10
        self.step_size = 0.02
        self.initial_gripper_pose = None

    def open_grasped_cabinet(self):

        self.initial_gripper_pose = self.robot.get_tcp_pose()
        # determine initial joint estimation
        # as a revolute joint between the handle and the robot base
        self.gripper_poses.append(self.initial_gripper_pose)
        estimated_twist_translation = SE3Container.from_homogeneous_matrix(self.initial_gripper_pose).translation
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
                self.robot.servo_to_pose(new_setpoint_pose)
                # if force has become too high or gripper no longer has contact, stop
                if self._is_force_too_high() or not self._is_grasped_heuristic():
                    raise Exception("Cabinet could not be opened")
                # collect the gripper pose
                self.gripper_poses.append(self.robot.get_tcp_pose())

            # make new estimate of the articulation
            # TODO: use the factor graphs to estimate the joint configuration
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
        twist_joint_params, aux_data = FG_twist_estimation(poses, 0.05, 0.5)
        twist = twist_joint_params.twist
        twist_frame = twist_joint_params.base_transform
        twist_in_poses_frame = sm.tr2adjoint(twist_frame) @ twist
        return twist_in_poses_frame

    def get_joint_q_from_gripper_pose_and_twist(
        self,
        initial_gripper_pose: HomogeneousMatrixType,
        current_gripper_pose: HomogeneousMatrixType,
        twist: TwistType,
    ) -> float:
        se3_rotation_first = sm.trlog(initial_gripper_pose ^ -1 @ current_gripper_pose)
        # spatialmath lib uses rotation first, so we need to roll the array to get the translation first
        # as is our convention and the one used in the articulation estimation
        se3_rotation_last = np.roll(se3_rotation_first, 3)
        q = se3_rotation_last / twist
        q = np.mean(q)  # average over numerical errors
        # TODO: check if the errors are not too high?
        return q

    def get_gripper_pose_from_joint_q_and_twist(
        self, q: float, initial_gripper_pose: HomogeneousMatrixType, twist: TwistType
    ) -> HomogeneousMatrixType:
        # spatialmath lib uses rotation first, so we need to roll the array to get the rotation first
        # as our convention is rotation last
        rotation_first_twist = np.roll(twist, 3)
        pose = initial_gripper_pose @ sm.trexp(q * rotation_first_twist)
        return pose

    def grasp_cabinet_handle(self, grasp_pose: HomogeneousMatrixType) -> None:
        self.robot.move_to_pose(grasp_pose)
        self.robot.switch_to_admittance_control()
        self.gripper.close()

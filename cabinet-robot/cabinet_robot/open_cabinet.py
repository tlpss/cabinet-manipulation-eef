import time
from concurrent.futures import Future, ThreadPoolExecutor
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
from cabinet_robot.joint_estimation import FGJointEstimator, EstimationResults
from cabinet_robot.robot import FZIControlledRobot
from loguru import logger
from cabinet_robot.manual_point_cloud import create_pointcloud_from_depth_map



class CabinetOpener:
    """class to open a cabinet door with a robot by using the TCP poses to perform runtime joint estimation configuration with factor graphs"""

    default_delta_step = 0.01

    def __init__(self, robot: FZIControlledRobot, gripper):
        self.robot = robot
        self.gripper = gripper
        self.gripper_poses = None
        self.n_steps = 15
        self.joint_configuration_step_delta = None
        self.initial_gripper_pose = None
        self.estimation_results = None
        self.camera = Zed2i(resolution=Zed2i.RESOLUTION_720, depth_mode=Zed2i.NEURAL_DEPTH_MODE)
        self.camera.runtime_params.texture_confidence_threshold = 100
        self.camera.runtime_params.confidence_threshold = 100

        self.max_FG_samples = 1 + 2 * self.n_steps # only 2 graph compilations
        self.FG_estimator = FGJointEstimator()
        self.thread_pool = ThreadPoolExecutor(max_workers=1)

        self.visualize = True
        self.check_estimations_manually = True
        
        if self.visualize:
            self._init_rerun()

    def open_grasped_cabinet(self):
        self.gripper_poses = []
        self.joint_configuration_step_delta = self.default_delta_step
        self.initial_gripper_pose = self.robot.get_tcp_pose()
        self.gripper_poses.append(self.initial_gripper_pose)

        # TODO: this is a hack, should be determined from the door plane normal? unless assumption that grasp pose will always be like this.
        estimated_twist_translation = np.array([0, 0, -1.0])
        estimated_twist_rotation = np.zeros(3)
        estimated_twist: TwistType = np.concatenate((estimated_twist_translation, estimated_twist_rotation))
        twist_in_base_pose = self.initial_gripper_pose
        q_joint = 0.0

        while not self._is_cabinet_open() and self._is_grasped_heuristic():
            if self.check_estimations_manually:
                check = input(
                    "check in rerun if the joint estimation is not crazy. Press a key to continue, D to signal the cabinet is open or CTRL+C to abort"
                ) #noqa
                if check == "d":
                    return
            # safety check - is robot controller still up and running?
            if not self.robot._get_active_FZI_controller() == self.robot.FZI_ADMITTANCE_CONTROLLER_NAME:
                raise Exception("Admittance controller is not running")

            # if there hase been a joint estimation (always except for the first iteration)
            # determine an appropriate step_size (both magnitude and direction)
            if self.estimation_results is not None:
                estimated_joint_states = self.estimation_results.aux_data["joint_states"]
                logger.debug(f" {estimated_joint_states=}")
                self.joint_configuration_step_delta = self._get_step_delta(estimated_joint_states,estimated_twist)

            logger.info(f" {self.joint_configuration_step_delta=}")

            # pre-compile the factor graph in a separate thread
            precompile_future = self._precompile_graph(num_samples=min(len(self.gripper_poses) + self.n_steps,self.max_FG_samples))
            for i in range(self.n_steps):
                # take a small step in the joint configuration
                q_joint = q_joint + self.joint_configuration_step_delta

                # TODO: add safety check with distance between current pose and new setpoint pose. If distance is too large, stop
                new_setpoint_pose = self.get_gripper_pose_from_joint_q_and_twist(
                    q_joint, twist_in_base_pose, estimated_twist
                )
        
                self.robot.set_target_pose(new_setpoint_pose)
                # wait for robot to reach the setpoint
                # TODO: check for changes in the TCP pose to determine if the robot has reached a stable pose
                time.sleep(2.0)

                # if force has become too high or gripper no longer has contact, stop
                if self._is_force_too_high() or not self._is_grasped_heuristic():
                    raise Exception("Cabinet could not be opened")
               
                # collect the gripper pose
                self.gripper_poses.append(self.robot.get_tcp_pose())

                if self.visualize:
                    rerun.log_point(
                    "world/robot_setpoint", new_setpoint_pose[:3, 3], color=(1, 1, 0, 0.7), radius=0.012
                    )
                    rerun.log_image("camera/rgb", self.camera.get_rgb_image())

                    rerun.log_points(
                        "world/gripper_poses",
                        positions=np.asarray(self.gripper_poses)[:,:3, 3],
                        colors=[150, 0, 0],
                        radii=0.01,
                    )

                            # log wrench 
                    wrench = self.robot.get_wrench()
                    # log each scalar
                    for i,label in zip(range(6),["Fx","Fy","Fz","Tx","Ty","Tz"]):
                        rerun.log_scalar(f"robot/wrench/{label}", wrench[i])


            # make new estimate of the articulation

            # wait for the precompiled graph to be ready
            logger.debug("Waiting for graph compilation to finish")
            precompile_future.result(60)
            logger.debug("graph is compiled")
            
            gripper_poses = self._subsample_gripper_poses(self.gripper_poses,self.max_FG_samples)
            estimated_twist, twist_in_base_pose, q_joint = self.estimate_twist(gripper_poses)
            logger.debug("twist estimation finished")

    def precompile_all_graphs(self):
        logger.info("Precompiling all graphs")
        n = self.max_FG_samples
        while n > self.n_steps:
            self._precompile_graph(num_samples=n).result(100)
            n = n - self.n_steps
        logger.info("Finished precompiling all graphs")

    @staticmethod
    def _subsample_gripper_poses(poses: List[np.ndarray],n_samples) -> List[np.ndarray]:
        if len(poses) < n_samples:
            return poses
        else:
            step = len(poses) / (n_samples-1)
            indices = [round(step*i) for i in range(n_samples-1)]
            sampled_poses = [poses[i] for i in indices]
            sampled_poses.append(poses[-1])
            return sampled_poses

    def _get_step_delta(self,estimated_joint_states:List[float],estimated_twist:np.ndarray) -> float:
        step_direction = np.sign(estimated_joint_states[-1] - estimated_joint_states[0])
        # determine step size to 'normalize' the linear velocity of the twist
        step_size = self.default_delta_step/ np.linalg.norm(estimated_twist[:3])
        return step_direction * step_size

    def _precompile_graph(self, num_samples: int) -> Future:
        logger.debug(f"compiling graph for {num_samples} samples")
        return self.thread_pool.submit(self.FG_estimator.get_compiled_graph, num_samples)

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
        estimation_results = self.FG_estimator.estimate_joint_twist(poses)
        self.estimation_results = estimation_results
        twist = np.asarray(estimation_results.twist)
        twist_frame_in_base_pose = np.asarray(estimation_results.twist_frame_in_base_pose.as_matrix())

        if self.visualize:
            self.visualize_estimation(estimation_results)
        logger.info(f"Estimated twist: {twist}")
        logger.info(f"Estimated twist frame in base pose: {twist_frame_in_base_pose}")
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
        depth_map = self.camera.get_depth_map()
        intrinsics = self.camera.intrinsics_matrix()
        diy_pointcloud = create_pointcloud_from_depth_map(depth_map,rgb, intrinsics)
        
        #pointcloud = self.camera.get_colored_point_cloud()

        #rerun.log_image("world/camera/rgb", image=rgb)
        rerun.log_image("camera/depth", image=depth)
        rerun.log_image("camera/rgb", image=rgb)
        #rerun.log_points("world/camera/pointcloud", positions=pointcloud[:, :3], colors=pointcloud[:, 3:])
        rerun.log_points("world/camera/diy_pointcloud", positions=diy_pointcloud[:, :3], colors=diy_pointcloud[:, 3:])
        
        camera_pose_in_world = get_camera_pose_in_robot_frame()
        se3_container = SE3Container.from_homogeneous_matrix(camera_pose_in_world)
        rerun.log_rigid3(
            "world/camera", parent_from_child=(se3_container.translation, se3_container.orientation_as_quaternion)
        )
        # rerun.log_pinhole(
        #     "world/camera/rgb",
        #     child_from_parent=self.camera.intrinsics_matrix(),
        #     width=rgb.shape[1],
        #     height=rgb.shape[0],
        # )



    def visualize_estimation(self,estimation: EstimationResults):
        q_values = np.asarray(estimation.aux_data["joint_states"])
        delta_step = self._get_step_delta(q_values, estimation.twist)
        future_q_values = np.linspace(q_values[-1] - 10 * delta_step, q_values[-1] + 20 * delta_step, 30)

        estimated_latent_part_poses = np.stack(
            [np.asarray(m.as_matrix()) for m in estimation.aux_data["latent_poses"]["second"]]
        )
        estimated_latent_body_poses = np.stack(
            [np.asarray(m.as_matrix()) for m in estimation.aux_data["latent_poses"]["first"]]
        )
        estimated_future_latent_poses = np.stack(
            [
                np.asarray(estimation.twist_frame_in_base_pose.as_matrix()) @ sm.trexp(np.asarray(estimation.twist) * q)
                for q in future_q_values
            ]
        )
        rerun.log_points("world/estimated_latent_part_poses", positions = estimated_latent_part_poses[:, :3, 3], colors=[0, 255, 100], radii=0.01)
        rerun.log_points("world/estimated_latent_body_poses", positions = estimated_latent_body_poses[:, :3, 3], colors=[0, 255, 200], radii=0.01)

        rerun.log_points("world/estimated_part_poses", positions=estimated_future_latent_poses[:, :3, 3], colors=[0, 255, 0], radii=0.01)

    def _init_rerun(self):
        rerun.disconnect()
        rerun.init(f"cabinet-opener-{datetime.now()}", spawn=True)


if __name__ == "__main__":
    robot_ip = "10.42.0.162"

    robot = FZIControlledRobot()
    gripper = Robotiq2F85(robot_ip)
    gripper.open()
    gripper.speed = gripper.gripper_specs.min_speed  # so that closing is not too fast and admittance can keep up
    gripper.force = gripper.gripper_specs.max_force # limit slip as much as possible
    cabinet_opener = CabinetOpener(robot, gripper)
    cabinet_opener.precompile_all_graphs()

    while(True):
    
        input("manually grasp the handle and press enter")
        robot.switch_to_admittance_control()
        robot.set_target_pose(robot.get_tcp_pose())
        input("press play on the polyscope to hand over control. Press enter when done")
        cabinet_opener._init_rerun()
        cabinet_opener.log_pointcloud()
        cabinet_opener.open_grasped_cabinet()


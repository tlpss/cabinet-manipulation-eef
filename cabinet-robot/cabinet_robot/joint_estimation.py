import dataclasses
from typing import List

import articulation_estimation.factor_graph as factor_graph
import articulation_estimation.helpers as helpers
import jax.numpy as jnp
import numpy as onp
from airo_typing import HomogeneousMatrixType
from articulation_estimation import baseline
from articulation_estimation.baseline.joints import TwistJointParameters
from articulation_estimation.factor_graph.helpers import JointFormulation
from articulation_estimation.sample_generator import JointConnection
from jaxlie import SE3 as jaxlie_SE3


@dataclasses.dataclass
class EstimationResults:
    twist: jnp.ndarray  # twist in the twist frame
    twist_frame_in_base_pose: jaxlie_SE3  # twist frame in the frame in which the part poses are expressed (base frame)
    current_joint_configuration: float
    aux_data: dict


def build_graph(num_samples: int, stddev_pos, stddev_ori):

    structure = {"first_second": JointConnection(from_id="first", to_id="second", via_id="first_second")}
    factor_graph_options = factor_graph.graph.GraphOptions(
        observe_transformation=False,
        observe_part_poses=True,
        observe_part_pose_betweens=False,
        observe_part_centers=False,
    )
    joint_formulation = {"first_second": JointFormulation.GeneralTwist}

    graph: factor_graph.graph.Graph = factor_graph.graph.Graph()
    variance_exp_factor = onp.concatenate(
        (
            onp.repeat(stddev_pos**2, repeats=3),
            onp.repeat(stddev_ori**2, repeats=3),
        )
    )
    graph.build_graph(
        num_samples,
        structure,
        factor_graph_options,
        joint_formulation,
        variance_exp_factor=variance_exp_factor,
    )
    return graph


def optimize_graph(
    graph: factor_graph.graph.Graph,
    poses_named,
    variance_pos=0.005,
    variance_ori=0.02,
    verbose=False,
    max_restarts=2,
    aux_data_in={},
    use_huber=False,
):
    # Copy dict!
    aux_data = aux_data_in.copy()
    # if not "best_assignment" in aux_data.keys():
    #         aux_data["best_assignment"] = None

    pose_variance = onp.concatenate(
        (
            onp.repeat(variance_pos, repeats=3),
            onp.repeat(variance_ori, repeats=3),
        )
    )

    graph.update_poses(poses_named, pose_variance, use_huber=use_huber)
    twist, base_transform, aux_data = graph.solve_graph(max_restarts=max_restarts, aux_data_in=aux_data)
    result = EstimationResults(
        twist=twist,
        twist_frame_in_base_pose=base_transform,
        current_joint_configuration=float(aux_data["joint_states"][-1]),
        aux_data=aux_data,
    )
    return result


def FG_twist_estimation(part_poses: List[HomogeneousMatrixType], stddev_pos, stddev_ori):
    graph = build_graph(num_samples=len(part_poses), stddev_pos=stddev_pos, stddev_ori=stddev_ori)

    part_poses = [jaxlie_SE3.from_matrix(pose) for pose in part_poses]
    # using the initial part pose as body poses
    # will make the twist in the part pose's initial frame
    # could also set to the origin
    body_poses = [part_poses[0]] * len(part_poses)

    poses = {"first": body_poses, "second": part_poses}
    estimation_results = optimize_graph(
        graph,
        poses,
        variance_pos=stddev_pos * stddev_pos,
        variance_ori=stddev_ori * stddev_ori,
        aux_data_in={"joint_states": None, "latent_poses": None},
    )

    estimation = TwistJointParameters(
        helpers.mean_pose(estimation_results.aux_data["latent_poses"]["first"])
        @ estimation_results.twist_frame_in_base_pose,
        estimation_results.twist,
    )

    estimation_results.twist_frame_in_base_pose = estimation.base_transform
    return estimation_results


def sturm_twist_estimation(part_poses, stddev_pos, stddev_ori):
    part_poses = [jaxlie_SE3.from_matrix(pose) for pose in part_poses]
    # using the initial part pose as body poses
    # will make the twist in the part pose's initial frame
    # could also set to the origin
    body_poses = [part_poses[0]] * len(part_poses)
    base_transform_sturm, twist_sturm, aux_data_sturm = baseline.sturm.fit_pose_trajectories(
        body_poses,
        [part_poses],
        variance_pos=stddev_pos * stddev_pos,
        variance_ori=stddev_ori * stddev_ori,
        original=False,
        aux_data_in={"joint_states": None, "latent_poses": None},
    )

    estimation = TwistJointParameters(
        helpers.mean_pose(aux_data_sturm["latent_poses"]["first"]) @ base_transform_sturm,
        twist_sturm,
    )
    raise NotImplementedError
    # TODO: same format as FG_twist_estimation
    return estimation

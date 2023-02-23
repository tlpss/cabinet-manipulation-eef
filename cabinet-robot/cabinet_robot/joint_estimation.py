from typing import List

import articulation_estimation.factor_graph as factor_graph
import articulation_estimation.helpers as helpers
import numpy as onp
from airo_typing import HomogeneousMatrixType
from articulation_estimation import baseline
from articulation_estimation.baseline.joints import TwistJointParameters
from articulation_estimation.factor_graph.helpers import JointFormulation
from articulation_estimation.sample_generator import JointConnection
from jaxlie import SE3 as jaxlie_SE3


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
    return twist, base_transform, aux_data


def FG_twist_estimation(part_poses: List[HomogeneousMatrixType], stddev_pos, stddev_ori):
    graph = build_graph(num_samples=len(part_poses), stddev_pos=stddev_pos, stddev_ori=stddev_ori)

    part_poses = [jaxlie_SE3.from_matrix(pose) for pose in part_poses]
    # using the initial part pose as body poses
    # will make the twist in the part pose's initial frame
    # could also set to the origin
    body_poses = [part_poses[0]] * len(part_poses)

    poses = {"first": body_poses, "second": part_poses}
    twist_pred, transform_pred, aux_data_fg_pred = optimize_graph(
        graph,
        poses,
        variance_pos=stddev_pos * stddev_pos,
        variance_ori=stddev_ori * stddev_ori,
        aux_data_in={"joint_states": None, "latent_poses": None},
    )

    estimation = TwistJointParameters(
        helpers.mean_pose(aux_data_fg_pred["latent_poses"]["first"]) @ transform_pred,
        twist_pred,
    )
    return estimation, aux_data_fg_pred


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
    return estimation, aux_data_sturm


if __name__ == "__main__":

    import jax
    import numpy as np

    jax.config.update("jax_disable_jit", True)

    original_part_pose = np.eye(4)
    part_poses = [np.copy(original_part_pose)]
    for _ in range(20):
        original_part_pose[0, 3] += 0.01
        part_poses.append(np.copy(original_part_pose))

    results, aux_data_dict = sturm_twist_estimation(part_poses, stddev_pos=0.005, stddev_ori=0.02)
    print(f"{results.twist=}")
    print(f"{results.base_transform=}")
    print(aux_data_dict["latent_poses"]["first"][0])
    print(aux_data_dict["latent_poses"]["second"][0])
    print(aux_data_dict["joint_states"][0])

    # test: latent_pose second[0] = base_transform @ exp(twist * joint_state[0])

    print(
        results.base_transform.as_matrix()
        @ jaxlie_SE3.exp(results.twist * aux_data_dict["joint_states"][0]).as_matrix()
    )
    print(
        results.base_transform.as_matrix()
        @ jaxlie_SE3.exp(results.twist * aux_data_dict["joint_states"][5]).as_matrix()
    )

    import spatialmath.base as sm

    twist_in_poses_frame = sm.tr2adjoint(np.asarray(results.base_transform.as_matrix())) @ np.asarray(results.twist)
    print(jaxlie_SE3.exp(twist_in_poses_frame * aux_data_dict["joint_states"][5]).as_matrix())

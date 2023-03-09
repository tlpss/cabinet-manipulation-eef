import dataclasses
from typing import List

import articulation_estimation.factor_graph as factor_graph
import articulation_estimation.helpers as helpers
import jax.numpy as jnp
import numpy as onp
import numpy as np
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


class FGJointEstimator:
    """Factor graph based joint estimation, wraps the cat-ind-fg codebase"""

    def __init__(self):
        self.compiled_graphs_cache = {}
        # larger noise for the part poses, as slip might occur between gripper and handle
        self.trans_noise_stddev = 0.001
        self.rot_noise_stddev = 0.05

        # very low noise, as the base is fixed. Avoid that the FG can have body latents that are not fixed.
        # This is a hack, as the FG should be be reformulated to have only latents for the part and not for the body.
        self.body_trans_noise_stddev = 1e-10
        self.body_rot_noise_stddev = 1e-10

        self.max_restarts = 20

    def _build_graph(self, num_samples: int):
        """build the JAX factor graph for the required number of data samples"""
        structure = {"first_second": JointConnection(from_id="first", to_id="second", via_id="first_second")}
        factor_graph_options = factor_graph.graph.GraphOptions(
            observe_transformation=False,
            observe_part_poses=True,
            observe_part_pose_betweens=False,
            observe_part_centers=False,
        )
        joint_formulation = {"first_second": JointFormulation.Revolute}

        graph: factor_graph.graph.Graph = factor_graph.graph.Graph()
        variance_exp_factor = onp.concatenate(
            (
                onp.repeat(self.trans_noise_stddev**2, repeats=3),
                onp.repeat(self.rot_noise_stddev**2, repeats=3),
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

    def get_compiled_graph(self, num_samples: int):
        """Get a compiled graph for the number of samples.
        Caches compiled graphs for later use.
        Compilation is triggered by optimizing with dummy poses.
        You can use this to compile the graph before you start the optimization loop.
        """

        if num_samples in self.compiled_graphs_cache.keys():
            print(f"reusing compiled graph for {num_samples} samples")
            return self.compiled_graphs_cache[num_samples]
        else:
            graph = self._build_graph(num_samples)
            # trigger JIT with dummy poses
            pose_dict = self._convert_part_poses_to_graph_format([np.eye(4)] * num_samples)
            self._optimize_graph(graph, pose_dict)
            # cache compiled graph for later use
            self.compiled_graphs_cache[num_samples] = graph
            return graph

    def _optimize_graph(
        self,
        graph: factor_graph.graph.Graph,
        poses_named,
        aux_data_in={},
        use_huber=False,
    ):
        """optimize the graph with the given poses and auxiliary data."""

        # Copy dict!
        aux_data = aux_data_in.copy()
        if "joint_states" not in aux_data.keys():
            aux_data["joint_states"] = None

        parts_pose_variance = onp.concatenate(
            (
                onp.repeat(self.trans_noise_stddev**2, repeats=3),
                onp.repeat(self.rot_noise_stddev**2, repeats=3),
            )
        )
        body_pose_variance = onp.concatenate(
            (
                onp.repeat(self.body_trans_noise_stddev**2, repeats=3),
                onp.repeat(self.body_rot_noise_stddev**2, repeats=3),
            )
        )
        pose_variance = {"second": parts_pose_variance, "first": body_pose_variance}

        graph.update_poses(poses_named, pose_variance, use_huber=use_huber)
        twist, base_transform, aux_data = graph.solve_graph(aux_data_in=aux_data,max_restarts = self.max_restarts)
        result = EstimationResults(
            twist=twist,
            twist_frame_in_base_pose=base_transform,
            current_joint_configuration=float(aux_data["joint_states"][-1]),
            aux_data=aux_data,
        )
        return result

    def _convert_part_poses_to_graph_format(self, part_poses: List[HomogeneousMatrixType]):
        """convenience function to convert a list of part poses to the format expected by the graph."""
        part_poses = [jaxlie_SE3.from_matrix(pose) for pose in part_poses]
        # using the initial part pose as body poses
        # will make the twist in the part pose's initial frame
        # could also set to the origin
        body_poses = [part_poses[0]] * len(part_poses)
        pose_dict = {"first": body_poses, "second": part_poses}
        return pose_dict

    def estimate_joint_twist(self, part_poses: List[HomogeneousMatrixType]):
        """Estimate the joint twist from the given part poses. This is the main function of this class."""
        graph = self.get_compiled_graph(num_samples=len(part_poses))
        pose_dict = self._convert_part_poses_to_graph_format(part_poses)

        estimation_results = self._optimize_graph(
            graph,
            pose_dict,
            aux_data_in={"joint_states": None, "latent_poses": None},
        )

        estimation = TwistJointParameters(
            helpers.mean_pose(estimation_results.aux_data["latent_poses"]["first"])
            @ estimation_results.twist_frame_in_base_pose,
            estimation_results.twist,
        )
        print(f"latent poses: {estimation_results.aux_data['latent_poses']['first']}")
        print( helpers.mean_pose(estimation_results.aux_data["latent_poses"]["first"]))
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

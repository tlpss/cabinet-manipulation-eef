import numpy as np
import rerun
from cabinet_robot.joint_estimation import EstimationResults
from spatialmath import base as sm


def log_points(name, positions: np.ndarray, color, radius: float):
    """
    positions: Nx3 array
    """
    rerun.log_points(
        name,
        positions=np.array(positions),
        colors=np.array(color),
        radii=radius,
    )


def visualize_estimation(estimation: EstimationResults):
    q_values = np.asarray(estimation.aux_data["joint_states"])
    future_q_values = np.linspace(q_values[-1], q_values[-1] + 3 * (q_values[-1] - q_values[0]), 30)

    estimated_latent_poses = np.stack(
        [np.asarray(m.as_matrix()) for m in estimation.aux_data["latent_poses"]["second"]]
    )
    estimated_future_latent_poses = np.stack(
        [
            np.asarray(estimation.twist_frame_in_base_pose.as_matrix()) @ sm.trexp(np.asarray(estimation.twist) * q)
            for q in future_q_values
        ]
    )

    log_points("world/estimated_latent_poses", estimated_latent_poses[:, :3, 3], [255, 0, 0], 0.01)
    log_points("world/estimated_future_latent_poses", estimated_future_latent_poses[:, :3, 3], [0, 255, 0], 0.01)


    
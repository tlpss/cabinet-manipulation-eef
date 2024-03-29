import pathlib
import pickle

import numpy as np
from airo_spatial_algebra import SE3Container
from airo_typing import HomogeneousMatrixType


def get_marker_pickle_path():
    return pathlib.Path(__file__).parents[1] / "scripts" / "marker.pickle"


def get_marker_pose_in_robot_frame() -> HomogeneousMatrixType:
    """manually measured for the current pickled marker_in_camera pose"""
    return SE3Container.from_euler_angles_and_translation(
        np.array([0, 0, 0]), np.array([0.275, -0.107, -0.010])
    ).homogeneous_matrix


def get_marker_pose_in_camera_frame() -> HomogeneousMatrixType:
    """load a pickled marker_in_camera pose. see static_camera_calibration.py"""
    # load pickle
    with open(get_marker_pickle_path(), "rb") as f:
        translation, rotation_matrix = pickle.load(f)
    return SE3Container.from_rotation_matrix_and_translation(rotation_matrix, translation).homogeneous_matrix


def get_camera_pose_in_robot_frame() -> HomogeneousMatrixType:
    return get_marker_pose_in_robot_frame() @ np.linalg.inv(get_marker_pose_in_camera_frame())


if __name__ == "__main__":
    print(get_marker_pose_in_robot_frame())
    print(get_marker_pose_in_camera_frame())
    print(np.linalg.inv(get_marker_pose_in_camera_frame()))
    print(get_camera_pose_in_robot_frame())

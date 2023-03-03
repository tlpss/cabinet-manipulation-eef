import pickle
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from airo_camera_toolkit.cameras.zed2i import Zed2i
from airo_camera_toolkit.reprojection import project_frame_to_image_plane
from airo_camera_toolkit.utils import ImageConverter


def get_aruco_marker_poses(
    frame: np.ndarray,
    cam_matrix: np.ndarray,
    aruco_marker_size: float,
    aruco_dictionary_name: str,
    visualize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    this_aruco_dictionary = cv2.aruco.Dictionary_get(aruco_dictionary_name)
    this_aruco_parameters = cv2.aruco.DetectorParameters_create()

    # Detect ArUco markers in the video frame
    (corners, marker_ids, rejected) = cv2.aruco.detectMarkers(
        frame, this_aruco_dictionary, parameters=this_aruco_parameters
    )

    # Check that at least one ArUco marker was detected
    if marker_ids is None:
        return frame, None, None, None

    # Refine the corners
    # cv2.aruco.refineDetectedMarkers(frame, board, corners, marker_ids, rejected)
    termination_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
        100,
        0.001,
    )  # max 100 iterations or 0.001m acc
    search_window_size = (
        5,
        5,
    )  # multiply by 2 + 1 to get real search window size opencv will use (5x5) = (11x11) window
    zero_zone = (-1, -1)  # none
    corners = cv2.cornerSubPix(frame[:, :, 0], corners[0], search_window_size, zero_zone, termination_criteria)

    # Get the rotation and translation vectors
    rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(corners, aruco_marker_size, cam_matrix, np.zeros(4))

    # The pose of the marker is with respect to the camera lens frame.
    # Imagine you are looking through the camera viewfinder,
    # the camera lens frame's:
    # x-axis points to the right
    # y-axis points straight down towards your toes
    # z-axis points straight ahead away from your eye, out of the camera
    translations = []
    rotation_matrices = []

    if marker_ids.size == rvecs.shape[0]:
        for i, marker_id in enumerate(marker_ids):
            # Store the rotation information
            rotation_matrix = cv2.Rodrigues(np.array(rvecs[i][0]))[0]

            translations.append(tvecs[i][0])
            rotation_matrices.append(rotation_matrix)

    else:
        print("[WARNING] detected markers does not equal amount of rotation vectors weirdly")
    # if visualize:
    #     frame = np.ascontiguousarray(frame)
    #     # Draw the axes on the marker
    #     for i in range(len(marker_ids)):
    #         cv2.drawFrameAxes(frame, cam_matrix, np.zeros(4), rvecs[i], tvecs[i], 0.05)
    return frame, translations, rotation_matrices, marker_ids


def draw_center_circle(image) -> np.ndarray:
    h, w, _ = image.shape
    center_u = w // 2
    center_v = h // 2
    center = (center_u, center_v)
    image = cv2.circle(image, center, 1, (255, 0, 255), thickness=2)
    return image


def draw_world_axes(image, world_to_camera, camera_matrix):
    project_points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
    )

    origin, x_pos, x_neg, y_pos, y_neg, z_pos = project_frame_to_image_plane(
        project_points, camera_matrix, world_to_camera
    ).astype(int)
    image = cv2.circle(image, origin, 10, (0, 255, 255), thickness=2)
    image = cv2.line(image, x_pos, origin, color=(0, 0, 255), thickness=2)
    image = cv2.line(image, x_neg, origin, color=(100, 100, 255), thickness=2)
    image = cv2.line(image, y_pos, origin, color=(0, 255, 0), thickness=2)
    image = cv2.line(image, y_neg, origin, color=(150, 255, 150), thickness=2)
    image = cv2.line(image, z_pos, origin, color=(255, 0, 0), thickness=2)
    return image


def save_calibration(rotation_matrix, translation):
    with open(Path(__file__).parent / "marker.pickle", "wb") as f:
        pickle.dump([translation, rotation_matrix], f)


if __name__ == "__main__":
    zed = Zed2i(resolution=Zed2i.RESOLUTION_2K)

    # Configure custom project-wide InputTransform based on camera, resolution, etc.
    _, h, w = zed.get_rgb_image().shape

    print("Press s to save Marker pose, q to quit.")
    while True:
        start_time = time.time()
        image = zed.get_rgb_image()
        image = ImageConverter(image).image_in_opencv_format

        intrinsics_matrix = zed.intrinsics_matrix
        _, translations, rotations, _ = get_aruco_marker_poses(
            image, intrinsics_matrix, 0.10, cv2.aruco.DICT_6X6_250, True
        )
        image = draw_center_circle(image)
        if rotations:
            print(rotations[0])

            aruco_in_camera_transform = np.eye(4)
            aruco_in_camera_transform[:3, :3] = rotations[0]
            aruco_in_camera_transform[:3, 3] = translations[0]
            world_to_camera = aruco_in_camera_transform
            image = draw_world_axes(image, world_to_camera, intrinsics_matrix)

        image = cv2.resize(image, (1280, 720))

        cv2.imshow("Camera feed", image)

        key = cv2.waitKey(10)
        if key == ord("s") and rotations is not None:
            print("Saving current marker pose to pickle.")
            save_calibration(rotations[0], translations[0])
            time.sleep(5)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break

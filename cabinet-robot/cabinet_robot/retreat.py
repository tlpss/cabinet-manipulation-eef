import time

import numpy as np
from airo_robots.grippers.hardware.robotiq_2f85_tcp import Robotiq2F85
from airo_spatial_algebra import SE3Container
from cabinet_robot.robot import FZIControlledRobot

if __name__ == "__main__":
    robot_ip = "10.42.0.162"

    robot = FZIControlledRobot()
    gripper = Robotiq2F85(robot_ip)
    gripper.open()
    time.sleep(5)
    rotation_matrix = np.array([
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ])
    home_pose = SE3Container.from_rotation_matrix_and_translation(rotation_matrix, np.array([0.4, -0.2, 0.2])).homogeneous_matrix
    robot.move_to_pose(home_pose)

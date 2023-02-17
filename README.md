# EEF cabinet manipulation

running this code requires a UR e-series robot and a robotiq gripper. If you have another position-controlled robot with a F/T sensor, you will have to find ROS drivers for it and change the launch and config files in the `ure_cartesian_controllers]` package.

To run the code on your robot setup:
- have your robot configured for ROS control, see [the driver setup guide](https://docs.ros.org/en/ros2_packages/rolling/api/ur_robot_driver/installation/robot_setup.html). Follow the network setup and robot preparation sections. Make sure you can ping the robot.
- create a polyscope installation where you configure the CoM and weight of your gripper, so that the measured wrench will be gravity-compensated by the UR controlbox.

## Local Development

### Local installation

#### python code

- clone this repo and pull the submodules `git submodule init && git submodule update`
- create the conda environment `conda env create -f environment.yaml`
- manually install the cat-ind-fg dependencies with `pip install -r cat-ind-fg/requirements.txt` and don't forget the apt installs in the cat-ind-fg readme
- initialize the pre-commit hooks `pre-commit install`


#### ROS docker image
building the docker image
- `docker build ROS/ -t ros-fzi-admittance`

using the docker image
- run `xhost + local:`
- start the docker container with the ROS stack in a separate terminal:
`docker run --net host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --gpus all --privileged -it ros-fzi-admittance`
- in the docker container terminal run `bash scripts/ure-fzi-start-script.sh` to start the ROS nodes. RVIZ should now spin up and the visualized robot should be in the same configuration as your real robot.
- run the external control program on the polyscope. As soon as this is started you are ready to control the robot using the `cabinet_robot/robot.py` module.
### Running formatting, linting and testing
The makefile contains commands to make this convenient. Run using `make <command>`.

### ROS development
This sections assumes basic knowledge about ROS2, ros2_control.

The docker container runs the following ROS packages
- ROS2 UR drivers to communicate with the UR robot
- FZI cartesian controllers
- ROS2 webbridge to allow the python modules outside of the docker container to communicate with JSON messages
-


- You can run rqt in a separate terminal on the docker container using `docker exec -it <container-name> bash`. This can be used to set parameters for the cartesian controllers, send messages manually or check which controllers are active at the moment.
- If the robot.py module is not behaving as it should, send messages manually to the target_pose topics and check that the controllers are activated. Read the log output of the ros nodes carefully to check for warnings/errors.  This can be done with rqt or using the ROS2 CLI

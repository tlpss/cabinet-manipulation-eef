# EEF cabinet manipulation


## Local Development

### Local installation

- clone this repo
- create the conda environment `conda env create -f environment.yaml`
- initialize the pre-commit hooks `pre-commit install`


### Running formatting, linting and testing
The makefile contains commands to make this convenient. Run using `make <command>`.


# dump commands
 ros2 launch ure_cartesian_controllers ure_fzi_controllers.launch.py ur_type:=ur3e robot_ip:="s" use_fake_hardware:=true launch_rviz:=true
 docker run --net host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --gpus all --privileged -it ros-fzi-admittance

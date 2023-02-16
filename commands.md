# dump commands
 ros2 launch ure_cartesian_controllers ure_fzi_controllers.launch.py ur_type:=ur3e robot_ip:=10.42.0.162 use_fake_hardware:=false launch_rviz:=true

ros2 launch ure_cartesian_controllers ure_fzi_controllers.launch.py ur_type:=ur3e robot_ip:=10.42.0.162 use_fake_hardware:=true launch_rviz:=true


docker run --net host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --gpus all --privileged -it ros-fzi-admittance:galactic

docker build ROS/ -t ros-fzi-admittance
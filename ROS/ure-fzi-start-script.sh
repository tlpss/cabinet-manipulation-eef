
source install/setup.bash
ros2 launch ure_cartesian_controllers ure_fzi_controllers.launch.py ur_type:=ur3e robot_ip:=10.42.0.162 use_fake_hardware:=false launch_rviz:=true

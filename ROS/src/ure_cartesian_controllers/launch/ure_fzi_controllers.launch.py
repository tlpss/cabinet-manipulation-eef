# launch file to start up the UR ros drivers, the fzi cartesian controllers
# a ros webserver and a static transform publi

# This file was hacked together by @tlpss
# by combining the ur_control.launch.py file from the ur ros2 drivers
# and the example launch file (https://github.com/fzi-forschungszentrum-informatik/cartesian_controllers/blob/ros2-devel/cartesian_controller_simulation/launch/simulation.launch.py) from the fzi controllers
# to get the fzi controllers working with the ur ros2 drivers

# as the topics in the fzi controllers are hardcoded and the ur ros2 drivers use a different naming scheme
# the topics need to be remapped at the moment the controller manager is launched
# which requires retaking the ur_bringup launch file and adding the remapping to the controller manager, which creates a lot of code duplication

# The original ur_control.launch.py file is licensed under the BSD 3-Clause License:
#
#  Copyright (c) 2021 PickNik, Inc.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the {copyright_holder} nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

#
# Author: Denis Stogl

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):

    # Initialize Arguments
    ur_type = LaunchConfiguration("ur_type")
    robot_ip = LaunchConfiguration("robot_ip")
    safety_limits = LaunchConfiguration("safety_limits")
    safety_pos_margin = LaunchConfiguration("safety_pos_margin")
    safety_k_position = LaunchConfiguration("safety_k_position")
    # General arguments
    LaunchConfiguration("controllers_file")
    description_package = LaunchConfiguration("description_package")
    description_file = LaunchConfiguration("description_file")
    urdf_description_package = "ure_cartesian_controllers"
    prefix = LaunchConfiguration("prefix")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    fake_sensor_commands = LaunchConfiguration("fake_sensor_commands")
    LaunchConfiguration("initial_joint_controller")
    activate_joint_controller = LaunchConfiguration("activate_joint_controller")
    launch_rviz = LaunchConfiguration("launch_rviz")
    headless_mode = LaunchConfiguration("headless_mode")
    launch_dashboard_client = LaunchConfiguration("launch_dashboard_client")
    use_tool_communication = LaunchConfiguration("use_tool_communication")
    tool_parity = LaunchConfiguration("tool_parity")
    tool_baud_rate = LaunchConfiguration("tool_baud_rate")
    tool_stop_bits = LaunchConfiguration("tool_stop_bits")
    tool_rx_idle_chars = LaunchConfiguration("tool_rx_idle_chars")
    tool_tx_idle_chars = LaunchConfiguration("tool_tx_idle_chars")
    tool_device_name = LaunchConfiguration("tool_device_name")
    tool_tcp_port = LaunchConfiguration("tool_tcp_port")
    tool_voltage = LaunchConfiguration("tool_voltage")

    joint_limit_params = PathJoinSubstitution(
        [FindPackageShare(description_package), "config", ur_type, "joint_limits.yaml"]
    )
    kinematics_params = PathJoinSubstitution(
        [FindPackageShare(description_package), "config", ur_type, "default_kinematics.yaml"]
    )
    physical_params = PathJoinSubstitution(
        [FindPackageShare(description_package), "config", ur_type, "physical_parameters.yaml"]
    )
    visual_params = PathJoinSubstitution(
        [FindPackageShare(description_package), "config", ur_type, "visual_parameters.yaml"]
    )
    script_filename = PathJoinSubstitution([FindPackageShare("ur_robot_driver"), "resources", "ros_control.urscript"])
    input_recipe_filename = PathJoinSubstitution(
        [FindPackageShare("ur_robot_driver"), "resources", "rtde_input_recipe.txt"]
    )
    output_recipe_filename = PathJoinSubstitution(
        [FindPackageShare("ur_robot_driver"), "resources", "rtde_output_recipe.txt"]
    )

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare(urdf_description_package), "urdf", description_file]),
            " ",
            "robot_ip:=",
            robot_ip,
            " ",
            "joint_limit_params:=",
            joint_limit_params,
            " ",
            "kinematics_params:=",
            kinematics_params,
            " ",
            "physical_params:=",
            physical_params,
            " ",
            "visual_params:=",
            visual_params,
            " ",
            "safety_limits:=",
            safety_limits,
            " ",
            "safety_pos_margin:=",
            safety_pos_margin,
            " ",
            "safety_k_position:=",
            safety_k_position,
            " ",
            "name:=",
            ur_type,
            " ",
            "script_filename:=",
            script_filename,
            " ",
            "input_recipe_filename:=",
            input_recipe_filename,
            " ",
            "output_recipe_filename:=",
            output_recipe_filename,
            " ",
            "prefix:=",
            prefix,
            " ",
            "use_fake_hardware:=",
            use_fake_hardware,
            " ",
            "fake_sensor_commands:=",
            fake_sensor_commands,
            " ",
            "headless_mode:=",
            headless_mode,
            " ",
            "use_tool_communication:=",
            use_tool_communication,
            " ",
            "tool_parity:=",
            tool_parity,
            " ",
            "tool_baud_rate:=",
            tool_baud_rate,
            " ",
            "tool_stop_bits:=",
            tool_stop_bits,
            " ",
            "tool_rx_idle_chars:=",
            tool_rx_idle_chars,
            " ",
            "tool_tx_idle_chars:=",
            tool_tx_idle_chars,
            " ",
            "tool_device_name:=",
            tool_device_name,
            " ",
            "tool_tcp_port:=",
            tool_tcp_port,
            " ",
            "tool_voltage:=",
            tool_voltage,
            " ",
        ]
    )
    robot_description = {"robot_description": robot_description_content}

    initial_controllers_config = PathJoinSubstitution(
        [FindPackageShare("ure_cartesian_controllers"), "config", "cartesian_controllers.yaml"]
    )

    rviz_config_file = PathJoinSubstitution([FindPackageShare(description_package), "rviz", "view_robot.rviz"])

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, initial_controllers_config],
        output={
            "stdout": "screen",
            "stderr": "screen",
        },
        # the cartesian controllers do not allow to specify the topic names
        # so remap their topics here
        remappings=[
            ("cartesian_compliance_controller/target_frame", "target_pose"),
            ("cartesian_compliance_controller/target_wrench", "target_wrench"),
            ("cartesian_compliance_controller/ft_sensor_wrench", "force_torque_sensor_broadcaster/wrench"),
            ("cartesian_motion_controller/target_frame", "target_pose"),
            ("cartesian_force_controller/target_frame", "target_pose"),
            ("cartesian_force_controller/target_force", "target_wrench"),
        ],
        condition=IfCondition(use_fake_hardware),
    )

    cartesian_compliance_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["cartesian_compliance_controller", "--stopped", "-c", "/controller_manager"],
    )

    cartesian_motion_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["cartesian_motion_controller", "--stopped", "-c", "/controller_manager"],
    )
    cartesian_force_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["cartesian_force_controller", "--stopped", "-c", "/controller_manager"],
    )

    ur_control_node = Node(
        package="ur_robot_driver",
        executable="ur_ros2_control_node",
        parameters=[robot_description, initial_controllers_config],
        output={
            "stdout": "screen",
            "stderr": "screen",
        },
        # the cartesian controllers do not allow to specify the topic names
        # so remap their topics here
        remappings=[
            ("cartesian_compliance_controller/target_frame", "target_pose"),
            ("cartesian_compliance_controller/target_wrench", "target_wrench"),
            ("cartesian_compliance_controller/ft_sensor_wrench", "force_torque_sensor_broadcaster/wrench"),
            ("cartesian_motion_controller/target_frame", "target_pose"),
            ("cartesian_force_controller/target_frame", "target_pose"),
            ("cartesian_force_controller/target_force", "target_wrench"),
        ],
        condition=UnlessCondition(use_fake_hardware),
    )

    dashboard_client_node = Node(
        package="ur_robot_driver",
        condition=IfCondition(launch_dashboard_client) and UnlessCondition(use_fake_hardware),
        executable="dashboard_client",
        name="dashboard_client",
        output="screen",
        emulate_tty=True,
        parameters=[{"robot_ip": robot_ip}],
    )

    tool_communication_node = Node(
        package="ur_robot_driver",
        condition=IfCondition(use_tool_communication),
        executable="tool_communication.py",
        name="ur_tool_comm",
        output="screen",
        parameters=[
            {
                "robot_ip": robot_ip,
                "tcp_port": tool_tcp_port,
                "device_name": tool_device_name,
            }
        ],
    )

    controller_stopper_node = Node(
        package="ur_robot_driver",
        executable="controller_stopper_node",
        name="controller_stopper",
        output="screen",
        emulate_tty=True,
        condition=UnlessCondition(use_fake_hardware),
        parameters=[
            {"headless_mode": headless_mode},
            {"joint_controller_active": activate_joint_controller},
            {
                "consistent_controllers": [
                    "io_and_status_controller",
                    "force_torque_sensor_broadcaster",
                    "joint_state_broadcaster",
                    "speed_scaling_state_broadcaster",
                ]
            },
        ],
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )

    rviz_node = Node(
        package="rviz2",
        condition=IfCondition(launch_rviz),
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )

    io_and_status_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["io_and_status_controller", "-c", "/controller_manager"],
    )

    speed_scaling_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "speed_scaling_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
    )

    force_torque_sensor_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "force_torque_sensor_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
    )

    ### launch webserver nodes with default arguments
    # https://github.com/RobotWebTools/rosbridge_suite/blob/ros2/rosbridge_server/launch/rosbridge_websocket_launch.xml
    webserver_node = Node(
        package="rosbridge_server",
        executable="rosbridge_websocket",
        output="screen",
    )

    rosapi_node = Node(
        package="rosapi",
        executable="rosapi_node",
    )

    # node to publish the TCP pose as a topic
    # as you have no access to the tf2 library outside of ROS2
    tf2_to_topic_publisher_node = Node(
        package="ure_cartesian_controllers",
        executable="tf2_to_topic_publisher",
        parameters=[{"from_frame": "base"}, {"to_frame": "tcp"}, {"topic_name": "ur_tcp_pose"}],
    )

    nodes_to_start = [
        control_node,
        ur_control_node,
        dashboard_client_node,
        tool_communication_node,
        controller_stopper_node,
        robot_state_publisher_node,
        rviz_node,
        joint_state_broadcaster_spawner,
        io_and_status_controller_spawner,
        speed_scaling_state_broadcaster_spawner,
        force_torque_sensor_broadcaster_spawner,
        cartesian_compliance_controller_spawner,
        cartesian_motion_controller_spawner,
        cartesian_force_controller_spawner,
        webserver_node,
        rosapi_node,
        tf2_to_topic_publisher_node,
    ]

    return nodes_to_start


def generate_launch_description():
    declared_arguments = []
    # UR specific arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "ur_type",
            description="Type/series of used UR robot.",
            choices=["ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e", "ur16e"],
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument("robot_ip", description="IP address by which the robot can be reached.")
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "safety_limits",
            default_value="true",
            description="Enables the safety limits controller if true.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "safety_pos_margin",
            default_value="0.15",
            description="The margin to lower and upper limits in the safety controller.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "safety_k_position",
            default_value="20",
            description="k-position factor in the safety controller.",
        )
    )
    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "controllers_file",
            default_value="ur_controllers.yaml",
            description="YAML file with the controllers configuration.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "description_package",
            default_value="ur_description",
            description="Description package with robot URDF/XACRO files. Usually the argument \
        is not set, it enables use of a custom description.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "description_file",
            default_value="ur.urdf.xacro",
            description="URDF/XACRO description file with the robot.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "prefix",
            default_value='""',
            description="Prefix of the joint names, useful for \
        multi-robot setup. If changed than also joint names in the controllers' configuration \
        have to be updated.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_fake_hardware",
            default_value="false",
            description="Start robot with fake hardware mirroring command to its states.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "fake_sensor_commands",
            default_value="false",
            description="Enable fake command interfaces for sensors used for simple simulations. \
            Used only if 'use_fake_hardware' parameter is true.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "headless_mode",
            default_value="false",
            description="Enable headless mode for robot control",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "initial_joint_controller",
            default_value="scaled_joint_trajectory_controller",
            description="Initially loaded robot controller.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "activate_joint_controller",
            default_value="true",
            description="Activate loaded joint controller.",
        )
    )
    declared_arguments.append(DeclareLaunchArgument("launch_rviz", default_value="true", description="Launch RViz?"))
    declared_arguments.append(
        DeclareLaunchArgument("launch_dashboard_client", default_value="true", description="Launch Dashboard Client?")
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_tool_communication",
            default_value="false",
            description="Only available for e series!",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "tool_parity",
            default_value="0",
            description="Parity configuration for serial communication. Only effective, if \
            use_tool_communication is set to True.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "tool_baud_rate",
            default_value="115200",
            description="Baud rate configuration for serial communication. Only effective, if \
            use_tool_communication is set to True.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "tool_stop_bits",
            default_value="1",
            description="Stop bits configuration for serial communication. Only effective, if \
            use_tool_communication is set to True.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "tool_rx_idle_chars",
            default_value="1.5",
            description="RX idle chars configuration for serial communication. Only effective, \
            if use_tool_communication is set to True.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "tool_tx_idle_chars",
            default_value="3.5",
            description="TX idle chars configuration for serial communication. Only effective, \
            if use_tool_communication is set to True.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "tool_device_name",
            default_value="/tmp/ttyUR",
            description="File descriptor that will be generated for the tool communication device. \
            The user has be be allowed to write to this location. \
            Only effective, if use_tool_communication is set to True.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "tool_tcp_port",
            default_value="54321",
            description="Remote port that will be used for bridging the tool's serial device. \
            Only effective, if use_tool_communication is set to True.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "tool_voltage",
            default_value="24",
            description="Tool voltage that will be setup.",
        )
    )

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])

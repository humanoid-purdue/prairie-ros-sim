import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    use_joy = LaunchConfiguration('use_joy', default='true')
    use_hardware = LaunchConfiguration('use_hardware', default='false')
    use_rviz = LaunchConfiguration('use_rviz', default='true')

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('gz_sim'),
                'launch',
                'gz_only_nemo6.launch.py'
            )
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    prairie_share = get_package_share_directory('prairie_control')
    default_rviz_config_path = os.path.join(prairie_share, 'rviz/robot_viewer.rviz')
    teleop_config_path = os.path.join(prairie_share, 'config/xbox_teleop.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'),
        DeclareLaunchArgument(
            'use_joy',
            default_value='true',
            description='Use joystick input if true; use keyboard input if false'),
        DeclareLaunchArgument(
            'use_hardware',
            default_value='false',
            description='Launch real motors, real IMU, and real controllers'),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Launch RViz'),
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            output='screen',
            condition=IfCondition(use_joy)),
        Node(
            package='prairie_control',
            executable='prairie_teleop',
            name='prairie_teleop',
            output='screen',
            parameters=[teleop_config_path],
            condition=IfCondition(use_joy)),
        Node(
            package='prairie_control',
            executable='prairie_keyboard_teleop',
            name='prairie_keyboard_teleop',
            output='screen',
            condition=UnlessCondition(use_joy)),
        Node(
            package='prairie_control',
            executable='prairie_supervisor',
            name='prairie_supervisor',
            output='screen',
            parameters=[{'allow_real_walk': False}]),
        Node(
            package='prairie_control',
            executable='prairie_command_mux',
            name='prairie_command_mux',
            output='screen'),
        Node(
            package='prairie_control',
            executable='gz_standing',
            name='gz_standing',
            output='screen'),
        Node(
            package='prairie_control',
            executable='gz_policy',
            name='gz_policy',
            output='screen'),
        Node(
            package='prairie_control',
            executable='gz_mirror',
            name='gz_mirror',
            output='screen'),
        gz_sim,
        Node(
            package='rviz2',
            executable='rviz2',
            name = 'rviz2',
            arguments = ['-d', default_rviz_config_path],
            condition=IfCondition(use_rviz)
        ),
        Node(
            package='motor_controller',
            executable='motor_controller',
            name='motor_controller',
            output='screen',
            condition=IfCondition(use_hardware)),
        Node(
            package='prairie_control',
            executable='real_imu',
            name='real_imu',
            output='screen',
            condition=IfCondition(use_hardware)),
        Node(
            package='prairie_control',
            executable='real_state_estimator',
            name='real_state_estimator',
            output='screen',
            condition=IfCondition(use_hardware)),
        Node(
            package='prairie_control',
            executable='real_standing',
            name='real_standing',
            output='screen',
            condition=IfCondition(use_hardware)),
    ])

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, SetEnvironmentVariable,
                            IncludeLaunchDescription, SetLaunchConfiguration)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    empty_gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('gz_sim'),
                'launch',
                'gz_only_nemo6.launch.py'
            )
        )
    )

    default_rviz_config_path = os.path.join(get_package_share_directory('prairie_control'), 'rviz/robot_viewer.rviz')

    urdf_file_name = 'urdf/nemo6.urdf'
    urdf = os.path.join(
        get_package_share_directory('prairie_control'),
        urdf_file_name)


    with open(urdf, 'r') as infp:
        robot_desc = infp.read()

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time, 'robot_description': robot_desc}],
            arguments=[urdf]),
        Node(
            package='motor_controller',
            executable='motor_controller',
            name='motor_controller',
            output='screen'),
        Node(
            package='rviz2',
            executable='rviz2',
            name = 'rviz2',
            arguments = ['-d' + default_rviz_config_path]
        ),
        Node(
            package='prairie_control',
            executable='master',
            name='master',
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
            empty_gz,
    ])
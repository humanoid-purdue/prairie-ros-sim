from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription([
        Node(
            package='motor_controller',
            executable='motor_controller',
            name='motor_controller',
            output='screen'),
        Node(
            package='prairie_control',
            executable='master_test',
            name='master_test',
            output='screen'),
        Node(
            package='prairie_control',
            executable='real_state_estimator',
            name='real_state_estimator',
            output='screen'),
    ])
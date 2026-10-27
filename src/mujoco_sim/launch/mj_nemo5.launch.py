from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    sim_node = Node(
        package="mujoco_sim",
        executable="sim_node",
        name="mujoco_sim_node",
        output="screen"
    )
    return LaunchDescription([
        sim_node
    ])
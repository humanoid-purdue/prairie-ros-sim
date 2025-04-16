from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    teleop_node = Node(
        package='keyboard_teleop',
        executable='keyboard_teleop_hold',
        name='keyboard_teleop_hold',
        output='screen'
    )
    
    return LaunchDescription([teleop_node])
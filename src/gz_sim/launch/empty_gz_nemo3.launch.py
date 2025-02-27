import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, SetEnvironmentVariable,
                            IncludeLaunchDescription, SetLaunchConfiguration,
                            RegisterEventHandler)
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessExit
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():

    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    # Xacro to URDF
    robot_description_content = ParameterValue(
        Command([
            PathJoinSubstitution([FindExecutable(name='xacro')]),
            ' ',
            PathJoinSubstitution(
                [FindPackageShare('gz_sim'),
                 'urdf', 'nemo3.xacro.urdf']
            ),
        ]), value_type=str
    )

    # Config file for ros2_control
    robot_controllers_config = PathJoinSubstitution(
        [
            FindPackageShare('gz_sim'),
            'config',
            'nemo3_robot_control.yaml',
        ]
    )

    # Node: ros2_control related nodes
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
    )
    joint_trajectory_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_trajectory_controller'],
            # Not sure why following not working
            #[
            #'joint_trajectory_controller',
            #'--param-file',
            #robot_controllers_config,
            #],
    )

    # Node: robot_state_publisher
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'use_sim_time': use_sim_time, 'robot_description': robot_description_content}],
        output='both'
    )

    # SDF for Gazebo Sim:
    sdf = os.path.join(get_package_share_directory('gz_sim'), 'sdf/walk_plane.sdf')

    # Node: ros_gz_sim (Gazebo Sim)
    #pkg_gz_sim = get_package_share_directory('gz_sim')
    #pkg_gz_sim = pkg_gz_sim[:pkg_gz_sim.rindex("/")]
    #os.environ['GZ_SIM_RESOURCE_PATH'] = pkg_gz_sim
    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-topic', 'robot_description',
                   '-name', 'purdue_nemo',
                    '-x', ['0.0'],
                    '-y', ['0.0'],
                    '-z', ['0.75'],  # TBD
                    '-R', ['0.0'],
                    '-P', ['0.0'],
                    '-Y', ['0.0']],
    )

    # Node: gz_bridge
    gz_bridge_config = os.path.join(
        get_package_share_directory('gz_sim'), 'config', 'config_gzbridge.yaml')
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{
            'config_file': gz_bridge_config,
        }],
        output='screen'
    )

    # Node: gz_state_observer (for RL training)
    node_gz_state_observer = Node(
        package='gz_sim',
        executable='gz_state_observer',
        name='gz_state_observer',
        output='screen'
    )

    #default_rviz_config_path = os.path.join(get_package_share_directory('gz_sim'), 'rviz/robot_viewer.rviz')
    #rviz_arg = DeclareLaunchArgument(name='rvizconfig', default_value=default_rviz_config_path,
    #                                 description='Absolute path to rviz config file')

    return LaunchDescription([
        # Launch gazebo environment
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [PathJoinSubstitution([FindPackageShare('ros_gz_sim'),
                                       'launch',
                                       'gz_sim.launch.py'])]),
            launch_arguments=[('gz_args', [' -r -v 1 ', sdf]), # -r to run immediately, otherwise, the controllers will not work
                              ('on_exit_shutdown', 'True')]
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=gz_spawn_entity,
                on_exit=[joint_state_broadcaster_spawner],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster_spawner,
                on_exit=[joint_trajectory_controller_spawner],
            )
        ),
        gz_bridge,
        node_robot_state_publisher,
        gz_spawn_entity,
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='both',
            parameters=[{'use_sim_time': use_sim_time, 'robot_description': robot_description_content}]
            ),
        Node(
            package='gz_sim',
            executable='gz_state_observer',
            name='gz_state_observer',
            output='screen'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
    ])

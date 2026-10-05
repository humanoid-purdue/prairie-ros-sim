# prairie-ros-sim


### Install requirements
```
sudo apt-get install gz-harmonic
sudo apt-get install ros-${ROS_DISTRO}-ros-gz
sudo apt install ros-${ROS_DISTRO}-ros2-control ros-${ROS_DISTRO}-ros2-controllers
sudo apt install ros-${ROS_DISTRO}-gz-ros2-control
sudo apt install ros-${ROS_DISTRO}-gz-ros2-control-demos
```

### Running the project
IF YOU USE WINDOWS:
In the compose.yaml, uncomment:
- /tmp/.X11-unix:/tmp/.X11-unix:rw
AND
- DISPLAY=${DISPLAY}

From the repository root:
docker compose -f docker/compose.yaml build        # first time: ~10 min
docker compose -f docker/compose.yaml run --rm dev

Inside the container (repo is at /ws):
colcon build --symlink-install && source install/setup.bash
ros2 launch prairie_control master_gzlink.launch.py use_rviz:=false use_joy:=false

### Launch nemo3 testFor the plot can you remov
```
ros2 launch gz_sim empty_gz_nemo3.launch.py

```

### Debug
```
ros2 control list_hardware_components # check if the urdf config for ros2_control plugin is loaded correctly

ros2 control list_controllers

ros2 param list /controller_manager
ros2 param get /controller_manager joint_trajectory_controller.joints
```

### Example joint trajectory message
NOTE: All the joints must be sent (cannot send only subset of the joints)
```
ros2 topic pub /joint_trajectory_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory '{
  "header": {
    "stamp": {"sec": 0, "nanosec": 0},
    "frame_id": ""
  },
  "joint_names": [
    "l_hip_pitch_joint", "l_hip_roll_joint", "l_hip_yaw_joint", "l_knee_joint", "l_foot_pitch_joint", "l_foot_roll_joint",
    "r_hip_pitch_joint", "r_hip_roll_joint", "r_hip_yaw_joint", "r_knee_joint", "r_foot_pitch_joint", "r_foot_roll_joint"
  ],
  "points": [
    {
      "positions": [-1.5, 0.1, -0.2, 0.5, -0.3, 0.0, -1.5, 0.1, -0.2, 0.5, -0.3, 0.0],
      "velocities": [],
      "accelerations": [],
      "effort": [],
      "time_from_start": {"sec": 0.5, "nanosec": 0}
    },
    {
      "positions": [0.1, -0.2, 0.3, -0.5, 0.4, -0.1, 0.1, -0.2, 0.3, -0.5, 0.4, -0.1],
      "velocities": [],
      "accelerations": [],
      "effort": [],
      "time_from_start": {"sec": 1, "nanosec": 0}
    }
  ]
}'
```

### master_gzlink controls

See [docs/master_gzlink_controls.md](docs/master_gzlink_controls.md) for the high-level control scheme, Xbox mapping, launch arguments, and topics to inspect.

# master_gzlink Controls

`master_gzlink.launch.py` is the Gazebo-first entry point for walking Nemo with a user controller. By default it launches the simulation control stack and leaves real hardware nodes off.

## Launch Modes

Simulation-only, with joystick and RViz:

```bash
ros2 launch prairie_control master_gzlink.launch.py
```

Useful launch arguments:

```bash
ros2 launch prairie_control master_gzlink.launch.py use_hardware:=false
ros2 launch prairie_control master_gzlink.launch.py use_joy:=false
ros2 launch prairie_control master_gzlink.launch.py use_rviz:=false
```

Real hardware nodes are only started when explicitly enabled:

```bash
ros2 launch prairie_control master_gzlink.launch.py use_hardware:=true
```

## Control Pipeline

The launch file splits user input, state supervision, and command routing into separate nodes:

```text
joy_node
  -> /joy
  -> prairie_teleop
  -> /prairie/user_command
  -> prairie_supervisor
  -> /prairie/state
  -> prairie_command_mux
  -> /joint_trajectories       # Gazebo
  -> /real_joint_trajectories  # Real motors, only useful with hardware nodes
```

`prairie_supervisor` owns the accepted high-level mode. `prairie_command_mux` forwards the matching controller output to Gazebo or to the real motor command topic.

## Xbox Mapping

The default mapping lives in `src/prairie_control/config/xbox_teleop.yaml`.

### Gazebo Controls

| Control | Effect |
| --- | --- |
| `A` | Simulation stand mode. Forwards `/gz_standing_jtp` to `/joint_trajectories`. |
| `B` | Simulation walk mode. Forwards `/gz_policy_jtp` to `/joint_trajectories`. |
| Left stick Y | Forward/back command, scaled by `0.4`. |
| Left stick X | Lateral command, scaled by `-0.3`. |
| Right stick X | Yaw command, scaled by `-0.8`. |
| Right stick Y | Unused. |

### Real Robot Controls

These only matter when `use_hardware:=true`.

| Control | Requested real mode |
| --- | --- |
| `X` | Disable motors. Publishes zero gains and zero torque. |
| `Y` | Home. Interpolates from current joint positions to the home pose. |
| `LB` | Stand. Uses the real standing controller output. |
| `RB` | Walk. Uses the real policy controller output. |

The supervisor enforces the safe real-mode sequence:

```text
DISABLED -> HOME -> STAND -> WALK
```

`X` can always return to `DISABLED`. `LB` can return from `WALK` to `STAND`. Unsafe out-of-order transitions are refused with a throttled warning.

## Topics To Inspect

Useful topics while debugging:

```bash
ros2 topic echo /prairie/user_command
ros2 topic echo /prairie/state
ros2 topic hz /joint_trajectories
ros2 topic hz /gz_policy_jtp
ros2 topic hz /gz_standing_jtp
```

For Gazebo-only runs, `/real_joint_trajectories` may still be published by the mux, but no motor controller should be running unless `use_hardware:=true`.

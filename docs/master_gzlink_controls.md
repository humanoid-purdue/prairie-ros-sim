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

Translational and yaw commands use an exponential filter for both joystick and
keyboard control. Keyboard scales and the filter time constant are configured
in `src/prairie_control/config/keyboard_teleop.yaml`:

| Parameter | Default | Effect |
| --- | ---: | --- |
| `vx_scale` | `0.4` | Forward/backward speed in m/s. |
| `vy_scale` | `0.3` | Left/right speed in m/s. |
| `yaw_rate_scale` | `0.8` | Yaw speed in rad/s. |
| `command_filter_tau` | `0.35` | Exponential-filter time constant in seconds. |

The default `command_filter_tau` reaches about 95% of a new command after one
second, giving acceleration and deceleration a noticeable lag. Set it to `0.0`
to disable the filter. Mode selection and emergency stop are immediate;
emergency stop also clears the filter state.

Setting `use_joy:=false` starts keyboard control in place of `joy_node` and
`prairie_teleop`. It opens a small pygame input window; keep that window focused
while driving. If it loses focus, all movement commands are reset to zero.

Real hardware nodes are only started when explicitly enabled:

```bash
ros2 launch prairie_control master_gzlink.launch.py use_hardware:=true
```

This launch is intended for Gazebo-link control. It does not start the
`real_policy` node, and the supervisor refuses real `WALK` commands by default.

## Control Pipeline

The launch file splits user input, state supervision, and command routing into separate nodes:

```text
joy_node -> /joy -> prairie_teleop             # use_joy:=true
                         or
prairie_keyboard_teleop                        # use_joy:=false
  -> /prairie/user_command
  -> prairie_supervisor
  -> /prairie/state
  -> prairie_command_mux
  -> /joint_trajectories       # Gazebo
  -> /real_joint_trajectories  # Real motors, only useful with hardware nodes
```

`prairie_supervisor` owns the accepted high-level mode. `prairie_command_mux` forwards the matching controller output to Gazebo or to the real motor command topic.

## Keyboard Mapping

Start keyboard control with:

```bash
ros2 launch prairie_control master_gzlink.launch.py use_joy:=false
```

Movement commands remain active while each key is held. Releasing a movement
key stops that component of motion. Opposite keys held together cancel each
other. The pygame input window must have keyboard focus.

| Key | Effect |
| --- | --- |
| `W` / `S` | Translate forward / backward at `0.4 m/s`. |
| `A` / `D` | Translate left / right at `0.3 m/s`. |
| `Q` / `E` | Yaw left / right at `0.8 rad/s`. |
| `1` | Simulation stand mode. |
| `2` | Simulation walk mode. |
| `3` | Disable real motors. Only relevant with `use_hardware:=true`. |
| `4` | Home the real robot. Only relevant with `use_hardware:=true`. |
| `5` | Mirror Gazebo on the real robot. Only relevant with `use_hardware:=true`. |
| `0` | Emergency stop: switch simulation to stand, disable real motors, and zero motion. |

The top number row and numeric keypad are both supported. The supervisor still
enforces the real-robot transition sequence documented below.

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
| `LB` | Mirror Gazebo. Sends `/gz_mirror_jtp` to `/real_joint_trajectories`. |
| `RB` | Unused. |

The supervisor enforces the safe real-mode sequence:

```text
DISABLED -> HOME -> MIRROR
```

`X` can always return to `DISABLED`. Unsafe out-of-order transitions are
refused with a throttled warning. If `/gz_mirror_jtp` is missing while real
mirror mode is selected, the mux publishes disabled motor commands instead of a
stale mirror command.

## Topics To Inspect

Useful topics while debugging:

```bash
ros2 topic echo /prairie/user_command
ros2 topic echo /prairie/state
ros2 topic hz /joint_trajectories
ros2 topic hz /gz_policy_jtp
ros2 topic hz /gz_standing_jtp
ros2 topic hz /gz_mirror_jtp
```

For Gazebo-only runs, `/real_joint_trajectories` may still be published by the mux, but no motor controller should be running unless `use_hardware:=true`.

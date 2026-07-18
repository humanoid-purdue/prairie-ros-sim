import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

DOMAIN_SIM = 0
DOMAIN_REAL = 1

MODE_DISABLED = 0
MODE_HOME = 1
MODE_STAND = 2
MODE_WALK = 3
MODE_MIRROR = 4
MODE_ESTOP = 5

JOINT_LIST_COMPLETE = [
    "l_hip_pitch_joint", "l_hip_roll_joint", "l_hip_yaw_joint",
    "l_knee_joint", "l_foot_pitch_joint", "l_foot_roll_joint",
    "r_hip_pitch_joint", "r_hip_roll_joint", "r_hip_yaw_joint",
    "r_knee_joint", "r_foot_pitch_joint", "r_foot_roll_joint",
    "l_shoulder_pitch_joint", "l_shoulder_roll_joint", "l_elbow_joint",
    "r_shoulder_pitch_joint", "r_shoulder_roll_joint", "r_elbow_joint",
]

HOME_POSE = [
    -0.698132, 0.0, 0.0, 1.22173, -0.523599, 0.0,
    -0.698132, 0.0, 0.0, 1.22173, -0.523599, 0.0,
    0.0, 0.05, 0.0, 0.0, -0.05, 0.0,
]

DEFAULT_REAL_KP = [
    35.0, 25.0, 25.0, 35.0, 35.0, 25.0,
    35.0, 25.0, 25.0, 35.0, 35.0, 25.0,
    15.0, 15.0, 15.0, 15.0, 15.0, 15.0,
]

WALK_REAL_KP = [
    35.0, 25.0, 25.0, 35.0, 35.0, 25.0,
    35.0, 25.0, 25.0, 35.0, 35.0, 25.0,
    15.0, 15.0, 15.0, 15.0, 15.0, 15.0,
]

DEFAULT_REAL_KD = [
    2.0, 1.0, 1.0, 2.0, 2.0, 1.0,
    2.0, 1.0, 1.0, 2.0, 2.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
]


@dataclass
class TeleopConfig:
    axis_left_x: int = 0
    axis_left_y: int = 1
    axis_right_x: int = 3
    button_a: int = 0
    button_b: int = 1
    button_x: int = 2
    button_y: int = 3
    button_lb: int = 4
    button_rb: int = -1
    vx_scale: float = 0.4
    vy_scale: float = -0.3
    yaw_rate_scale: float = -0.8
    deadzone: float = 0.1


@dataclass
class KeyboardConfig:
    vx_scale: float = 0.4
    vy_scale: float = 0.3
    yaw_rate_scale: float = 0.8


@dataclass
class CommandIntent:
    domain: int = DOMAIN_SIM
    mode: int = MODE_STAND
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0


@dataclass
class ControlState:
    sim_mode: int = MODE_STAND
    real_mode: int = MODE_DISABLED
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    real_start_standing: bool = False


class ExponentialCommandFilter:
    """Low-pass the motion fields of a command while preserving its mode."""

    def __init__(self, time_constant: float = 0.35):
        self.time_constant = float(time_constant)
        if self.time_constant < 0.0:
            raise ValueError("command filter time constant cannot be negative")
        self.reset()

    def reset(self):
        self.vx = 0.0
        self.vy = 0.0
        self.yaw_rate = 0.0

    def update(self, command: CommandIntent, dt: float) -> CommandIntent:
        if command.mode == MODE_ESTOP:
            self.reset()
        elif self.time_constant == 0.0:
            self.vx = command.vx
            self.vy = command.vy
            self.yaw_rate = command.yaw_rate
        else:
            alpha = 1.0 - math.exp(-max(0.0, float(dt)) / self.time_constant)
            self.vx += alpha * (command.vx - self.vx)
            self.vy += alpha * (command.vy - self.vy)
            self.yaw_rate += alpha * (command.yaw_rate - self.yaw_rate)

        return CommandIntent(
            domain=command.domain,
            mode=command.mode,
            vx=self.vx,
            vy=self.vy,
            yaw_rate=self.yaw_rate,
        )


def _axis(axes: Sequence[float], index: int, deadzone: float) -> float:
    if index < 0 or index >= len(axes):
        return 0.0
    value = float(axes[index])
    if abs(value) < deadzone:
        return 0.0
    return max(-1.0, min(1.0, value))


def _button(buttons: Sequence[int], index: int) -> bool:
    return 0 <= index < len(buttons) and bool(buttons[index])


class JoyMapper:
    def __init__(self, config: Optional[TeleopConfig] = None):
        self.config = config or TeleopConfig()
        self.previous_buttons: List[int] = []
        self.last_domain = DOMAIN_SIM
        self.last_mode = MODE_STAND

    def update(self, axes: Sequence[float], buttons: Sequence[int]) -> CommandIntent:
        config = self.config
        bindings = [
            (config.button_a, (DOMAIN_SIM, MODE_STAND)),
            (config.button_b, (DOMAIN_SIM, MODE_WALK)),
            (config.button_x, (DOMAIN_REAL, MODE_DISABLED)),
            (config.button_y, (DOMAIN_REAL, MODE_HOME)),
            (config.button_lb, (DOMAIN_REAL, MODE_MIRROR)),
        ]

        for button_index, command in bindings:
            if self._rising_edge(buttons, button_index):
                self.last_domain, self.last_mode = command

        self.previous_buttons = list(buttons)
        return CommandIntent(
            domain=self.last_domain,
            mode=self.last_mode,
            vx=config.vx_scale * _axis(axes, config.axis_left_y, config.deadzone),
            vy=config.vy_scale * _axis(axes, config.axis_left_x, config.deadzone),
            yaw_rate=(
                config.yaw_rate_scale *
                _axis(axes, config.axis_right_x, config.deadzone)
            ),
        )

    def _rising_edge(self, buttons: Sequence[int], index: int) -> bool:
        return _button(buttons, index) and not _button(self.previous_buttons, index)


class KeyboardMapper:
    """Map held movement keys and number-key mode selections to commands."""

    MODE_KEYS = {
        "0": (DOMAIN_SIM, MODE_ESTOP),
        "1": (DOMAIN_SIM, MODE_STAND),
        "2": (DOMAIN_SIM, MODE_WALK),
        "3": (DOMAIN_REAL, MODE_DISABLED),
        "4": (DOMAIN_REAL, MODE_HOME),
        "5": (DOMAIN_REAL, MODE_MIRROR),
    }
    MOVEMENT_KEYS = frozenset("wasdqe")

    def __init__(self, config: Optional[KeyboardConfig] = None):
        self.config = config or KeyboardConfig()
        self.pressed_keys = set()
        self.last_domain = DOMAIN_SIM
        self.last_mode = MODE_STAND

    def press(self, key: str) -> CommandIntent:
        key = key.lower()
        if key in self.MOVEMENT_KEYS:
            self.pressed_keys.add(key)
        if key in self.MODE_KEYS:
            self.last_domain, self.last_mode = self.MODE_KEYS[key]
        return self.command()

    def release(self, key: str) -> CommandIntent:
        self.pressed_keys.discard(key.lower())
        return self.command()

    def command(self) -> CommandIntent:
        config = self.config
        return CommandIntent(
            domain=self.last_domain,
            mode=self.last_mode,
            vx=config.vx_scale * self._direction("w", "s"),
            vy=config.vy_scale * self._direction("a", "d"),
            yaw_rate=config.yaw_rate_scale * self._direction("q", "e"),
        )

    def clear_movement(self) -> CommandIntent:
        self.pressed_keys.clear()
        return self.command()

    def _direction(self, positive_key: str, negative_key: str) -> float:
        return float(
            (positive_key in self.pressed_keys) -
            (negative_key in self.pressed_keys)
        )


class SupervisorCore:
    def __init__(self, allow_real_walk: bool = False):
        self.state = ControlState()
        self._last_mode_command: Optional[Tuple[int, int]] = None
        self.allow_real_walk = allow_real_walk

    def update(self, command: CommandIntent) -> Tuple[ControlState, bool, Optional[str]]:
        self.state.real_start_standing = False
        command_key = (command.domain, command.mode)

        if command.mode == MODE_ESTOP:
            self.state.sim_mode = MODE_STAND
            self.state.real_mode = MODE_DISABLED
            self.state.vx = 0.0
            self.state.vy = 0.0
            self.state.yaw_rate = 0.0
            self._last_mode_command = command_key
            return self.state, True, None

        self.state.vx = command.vx
        self.state.vy = command.vy
        self.state.yaw_rate = command.yaw_rate

        if command_key == self._last_mode_command:
            return self.state, True, None

        if command.domain == DOMAIN_SIM:
            state, accepted, reason = self._apply_sim(command.mode)
        elif command.domain == DOMAIN_REAL:
            state, accepted, reason = self._apply_real(command.mode)
        else:
            state, accepted, reason = (
                self.state,
                False,
                f"unknown command domain {command.domain}",
            )

        if accepted:
            self._last_mode_command = command_key
        return state, accepted, reason

    def _apply_sim(self, mode: int) -> Tuple[ControlState, bool, Optional[str]]:
        if mode in (MODE_STAND, MODE_WALK, MODE_MIRROR):
            self.state.sim_mode = mode
            return self.state, True, None
        return self.state, False, f"refusing unsupported sim mode {mode}"

    def _apply_real(self, mode: int) -> Tuple[ControlState, bool, Optional[str]]:
        current = self.state.real_mode
        if mode == MODE_DISABLED:
            self.state.real_mode = MODE_DISABLED
            return self.state, True, None
        if mode == current:
            return self.state, True, None
        if mode == MODE_HOME and current == MODE_DISABLED:
            self.state.real_mode = MODE_HOME
            return self.state, True, None
        if mode == MODE_STAND and current in (MODE_HOME, MODE_WALK, MODE_MIRROR):
            self.state.real_mode = MODE_STAND
            self.state.real_start_standing = True
            return self.state, True, None
        if mode == MODE_MIRROR and current in (MODE_HOME, MODE_STAND):
            self.state.real_mode = MODE_MIRROR
            return self.state, True, None
        if mode == MODE_WALK and not self.allow_real_walk:
            return (
                self.state,
                False,
                "refusing real WALK because real policy mode is disabled",
            )
        if mode == MODE_WALK and current == MODE_STAND:
            self.state.real_mode = MODE_WALK
            return self.state, True, None
        return (
            self.state,
            False,
            f"refusing real transition {mode_name(current)} -> {mode_name(mode)}",
        )


def mode_name(mode: int) -> str:
    names = {
        MODE_DISABLED: "DISABLED",
        MODE_HOME: "HOME",
        MODE_STAND: "STAND",
        MODE_WALK: "WALK",
        MODE_MIRROR: "MIRROR",
        MODE_ESTOP: "ESTOP",
    }
    return names.get(mode, f"UNKNOWN({mode})")


def _normalized(values: Sequence[float], length: int, fill: float = 0.0) -> List[float]:
    result = list(values[:length])
    if len(result) < length:
        result.extend([fill] * (length - len(result)))
    return result


def motor_payload_from_trajectory(
    joint_names: Sequence[str],
    positions: Sequence[float],
    velocities: Sequence[float],
    torques: Sequence[float],
    disable: bool = False,
    kp: Optional[Sequence[float]] = None,
) -> dict:
    count = len(JOINT_LIST_COMPLETE)
    payload = {
        "joint_names": list(joint_names) if joint_names else list(JOINT_LIST_COMPLETE),
        "positions": _normalized(positions, count),
        "velocities": _normalized(velocities, count),
        "torques": _normalized(torques, count),
    }
    if disable:
        payload["kp"] = [0.0] * count
        payload["kd"] = [0.0] * count
        payload["torques"] = [0.0] * count
    else:
        payload["kp"] = _normalized(kp if kp is not None else DEFAULT_REAL_KP, count)
        payload["kd"] = list(DEFAULT_REAL_KD)
    return payload

from prairie_control.control_modes import (
    DOMAIN_REAL,
    DOMAIN_SIM,
    MODE_DISABLED,
    MODE_ESTOP,
    MODE_HOME,
    MODE_MIRROR,
    MODE_STAND,
    MODE_WALK,
    CommandIntent,
    DEFAULT_REAL_KD,
    JoyMapper,
    KeyboardConfig,
    KeyboardMapper,
    SupervisorCore,
    TeleopConfig,
    WALK_REAL_KP,
    motor_payload_from_trajectory,
)


def assert_close(actual, expected):
    assert abs(actual - expected) < 1e-9


def test_joy_mapper_maps_buttons_and_axes():
    mapper = JoyMapper(TeleopConfig(deadzone=0.1))

    command = mapper.update([0.2, 0.5, 0.0, -0.25], [0, 1, 0, 0, 0, 0])

    assert command.domain == DOMAIN_SIM
    assert command.mode == MODE_WALK
    assert_close(command.vx, 0.2)
    assert_close(command.vy, -0.06)
    assert_close(command.yaw_rate, 0.2)

    command = mapper.update([0.0, 0.0, 0.0, 0.0], [0, 0, 1, 0, 0, 0])

    assert command.domain == DOMAIN_REAL
    assert command.mode == MODE_DISABLED
    assert command.vx == 0.0
    assert command.vy == 0.0
    assert command.yaw_rate == 0.0

    command = mapper.update([0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 1, 0])

    assert command.domain == DOMAIN_REAL
    assert command.mode == MODE_MIRROR

    command = mapper.update([0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 1])

    assert command.domain == DOMAIN_REAL
    assert command.mode == MODE_MIRROR


def test_keyboard_mapper_maps_held_movement_keys():
    mapper = KeyboardMapper(
        KeyboardConfig(vx_scale=0.4, vy_scale=0.3, yaw_rate_scale=0.8)
    )

    command = mapper.press("w")
    command = mapper.press("A")
    command = mapper.press("q")

    assert_close(command.vx, 0.4)
    assert_close(command.vy, 0.3)
    assert_close(command.yaw_rate, 0.8)

    command = mapper.press("s")
    assert command.vx == 0.0

    command = mapper.release("s")
    assert_close(command.vx, 0.4)

    mapper.release("w")
    mapper.release("a")
    mapper.release("q")
    command = mapper.press("d")
    command = mapper.press("e")

    assert command.vx == 0.0
    assert_close(command.vy, -0.3)
    assert_close(command.yaw_rate, -0.8)


def test_keyboard_mapper_maps_number_keys_to_modes():
    mapper = KeyboardMapper()

    expected_modes = {
        "1": (DOMAIN_SIM, MODE_STAND),
        "2": (DOMAIN_SIM, MODE_WALK),
        "3": (DOMAIN_REAL, MODE_DISABLED),
        "4": (DOMAIN_REAL, MODE_HOME),
        "5": (DOMAIN_REAL, MODE_MIRROR),
    }
    for key, expected in expected_modes.items():
        command = mapper.press(key)
        assert (command.domain, command.mode) == expected

    command = mapper.press("0")
    assert command.mode == MODE_ESTOP


def test_repeated_emergency_stop_keeps_motion_zero():
    supervisor = SupervisorCore()

    supervisor.update(
        CommandIntent(domain=DOMAIN_SIM, mode=MODE_ESTOP, vx=0.4)
    )
    state, accepted, _ = supervisor.update(
        CommandIntent(domain=DOMAIN_SIM, mode=MODE_ESTOP, yaw_rate=0.8)
    )

    assert accepted
    assert state.sim_mode == MODE_STAND
    assert state.real_mode == MODE_DISABLED
    assert state.vx == 0.0
    assert state.vy == 0.0
    assert state.yaw_rate == 0.0


def test_supervisor_accepts_real_mirror_sequence_and_refuses_real_walk():
    supervisor = SupervisorCore()

    state, accepted, reason = supervisor.update(
        CommandIntent(domain=DOMAIN_REAL, mode=MODE_WALK)
    )
    assert not accepted
    assert "real policy mode is disabled" in reason
    assert state.real_mode == MODE_DISABLED

    state, accepted, reason = supervisor.update(
        CommandIntent(domain=DOMAIN_REAL, mode=MODE_MIRROR)
    )
    assert not accepted
    assert "refusing real transition" in reason
    assert state.real_mode == MODE_DISABLED

    state, accepted, _ = supervisor.update(
        CommandIntent(domain=DOMAIN_REAL, mode=MODE_HOME)
    )
    assert accepted
    assert state.real_mode == MODE_HOME

    state, accepted, _ = supervisor.update(
        CommandIntent(domain=DOMAIN_REAL, mode=MODE_MIRROR)
    )
    assert accepted
    assert state.real_mode == MODE_MIRROR

    state, accepted, reason = supervisor.update(
        CommandIntent(domain=DOMAIN_REAL, mode=MODE_WALK)
    )
    assert not accepted
    assert "real policy mode is disabled" in reason
    assert state.real_mode == MODE_MIRROR


def test_supervisor_can_allow_real_walk_when_explicitly_enabled():
    supervisor = SupervisorCore(allow_real_walk=True)

    supervisor.update(CommandIntent(domain=DOMAIN_REAL, mode=MODE_HOME))
    state, accepted, _ = supervisor.update(
        CommandIntent(domain=DOMAIN_REAL, mode=MODE_STAND)
    )
    assert accepted
    assert state.real_mode == MODE_STAND
    assert state.real_start_standing

    state, accepted, _ = supervisor.update(
        CommandIntent(domain=DOMAIN_REAL, mode=MODE_WALK)
    )
    assert accepted
    assert state.real_mode == MODE_WALK


def test_supervisor_accepts_sim_stand_walk_switches():
    supervisor = SupervisorCore()

    state, accepted, _ = supervisor.update(
        CommandIntent(domain=DOMAIN_SIM, mode=MODE_WALK)
    )
    assert accepted
    assert state.sim_mode == MODE_WALK

    state, accepted, _ = supervisor.update(
        CommandIntent(domain=DOMAIN_SIM, mode=MODE_STAND)
    )
    assert accepted
    assert state.sim_mode == MODE_STAND


def test_motor_payload_disabled_and_walk_gains():
    disabled = motor_payload_from_trajectory(
        ["j0"],
        [1.0],
        [2.0],
        [3.0],
        disable=True,
    )
    assert disabled["positions"][0] == 1.0
    assert all(kp == 0.0 for kp in disabled["kp"])
    assert all(kd == 0.0 for kd in disabled["kd"])
    assert all(torque == 0.0 for torque in disabled["torques"])

    walking = motor_payload_from_trajectory(
        ["j0"],
        [1.0],
        [],
        [],
        kp=WALK_REAL_KP,
    )
    assert walking["kp"] == WALK_REAL_KP
    assert walking["kd"] == DEFAULT_REAL_KD
    assert walking["positions"][0] == 1.0
    assert walking["velocities"][0] == 0.0

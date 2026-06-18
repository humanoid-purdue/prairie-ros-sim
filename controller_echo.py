#!/usr/bin/env python3
"""Echo joystick/controller input without using ROS."""

import argparse
import sys
import time


AXIS_NAMES = {
    0: "left_x",
    1: "left_y",
    2: "right_x",
    3: "right_y",
    4: "lt_rt_or_trigger",
    5: "rt_or_trigger",
}

BUTTON_NAMES = {
    0: "A",
    1: "B",
    2: "X",
    3: "Y",
    4: "LB",
    5: "RB",
    6: "back",
    7: "start",
    8: "guide",
    9: "left_stick",
    10: "right_stick",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Detect a game controller and echo raw pygame events. "
            "This script is intentionally independent of ROS."
        )
    )
    parser.add_argument(
        "--id",
        type=int,
        default=0,
        help="Joystick index to open, default: 0.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List detected controllers and exit.",
    )
    parser.add_argument(
        "--deadzone",
        type=float,
        default=0.05,
        help="Ignore axis motion smaller than this value, default: 0.05.",
    )
    parser.add_argument(
        "--snapshot-rate",
        type=float,
        default=1.0,
        help="Print a full state snapshot at this rate in Hz, default: 1.0.",
    )
    return parser.parse_args()


def load_pygame():
    try:
        import pygame
    except ImportError:
        print(
            "pygame is not installed in this Python environment.\n"
            "Try: python3 -m pip install pygame",
            file=sys.stderr,
        )
        return None
    return pygame


def button_name(index):
    return BUTTON_NAMES.get(index, f"button_{index}")


def axis_name(index):
    return AXIS_NAMES.get(index, f"axis_{index}")


def list_controllers(pygame):
    count = pygame.joystick.get_count()
    if count == 0:
        print("No controllers detected.")
        return
    print(f"Detected {count} controller(s):")
    for i in range(count):
        joystick = pygame.joystick.Joystick(i)
        joystick.init()
        print(
            f"  [{i}] {joystick.get_name()} "
            f"axes={joystick.get_numaxes()} "
            f"buttons={joystick.get_numbuttons()} "
            f"hats={joystick.get_numhats()}"
        )


def format_axes(joystick):
    values = []
    for i in range(joystick.get_numaxes()):
        values.append(f"{axis_name(i)}={joystick.get_axis(i):+.3f}")
    return " ".join(values)


def format_buttons(joystick):
    pressed = []
    for i in range(joystick.get_numbuttons()):
        if joystick.get_button(i):
            pressed.append(button_name(i))
    return ", ".join(pressed) if pressed else "none"


def format_hats(joystick):
    values = []
    for i in range(joystick.get_numhats()):
        values.append(f"hat_{i}={joystick.get_hat(i)}")
    return " ".join(values) if values else "none"


def print_snapshot(joystick):
    print(
        f"STATE axes[{format_axes(joystick)}] "
        f"buttons[{format_buttons(joystick)}] "
        f"hats[{format_hats(joystick)}]",
        flush=True,
    )


def echo_events(pygame, joystick, deadzone, snapshot_rate):
    joystick_id = joystick.get_instance_id()
    snapshot_period = 1.0 / snapshot_rate if snapshot_rate > 0.0 else None
    next_snapshot = time.monotonic()

    print("Echoing controller input. Press Ctrl+C to exit.")
    print_snapshot(joystick)

    while True:
        for event in pygame.event.get():
            if getattr(event, "instance_id", joystick_id) != joystick_id:
                continue

            if event.type == pygame.JOYAXISMOTION:
                if abs(event.value) >= deadzone:
                    print(
                        f"AXIS {axis_name(event.axis)} "
                        f"index={event.axis} value={event.value:+.3f}",
                        flush=True,
                    )
            elif event.type == pygame.JOYBUTTONDOWN:
                print(
                    f"BUTTON {button_name(event.button)} "
                    f"index={event.button} down",
                    flush=True,
                )
            elif event.type == pygame.JOYBUTTONUP:
                print(
                    f"BUTTON {button_name(event.button)} "
                    f"index={event.button} up",
                    flush=True,
                )
            elif event.type == pygame.JOYHATMOTION:
                print(f"HAT index={event.hat} value={event.value}", flush=True)
            elif event.type == pygame.JOYDEVICEADDED:
                print(f"DEVICE added index={event.device_index}", flush=True)
            elif event.type == pygame.JOYDEVICEREMOVED:
                print(f"DEVICE removed instance_id={event.instance_id}", flush=True)

        if snapshot_period is not None and time.monotonic() >= next_snapshot:
            print_snapshot(joystick)
            next_snapshot = time.monotonic() + snapshot_period

        time.sleep(0.01)


def main():
    args = parse_args()
    pygame = load_pygame()
    if pygame is None:
        return 1

    pygame.init()
    pygame.joystick.init()

    if args.list:
        list_controllers(pygame)
        return 0

    count = pygame.joystick.get_count()
    if count == 0:
        print("No controllers detected. Check USB/Bluetooth pairing.", file=sys.stderr)
        return 2
    if args.id < 0 or args.id >= count:
        print(
            f"Controller id {args.id} is out of range. "
            f"Detected ids are 0..{count - 1}.",
            file=sys.stderr,
        )
        return 2

    joystick = pygame.joystick.Joystick(args.id)
    joystick.init()
    print(
        f"Opened controller [{args.id}] {joystick.get_name()} "
        f"axes={joystick.get_numaxes()} "
        f"buttons={joystick.get_numbuttons()} "
        f"hats={joystick.get_numhats()}"
    )

    try:
        echo_events(pygame, joystick, args.deadzone, args.snapshot_rate)
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        joystick.quit()
        pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())

import pygame
import rclpy
from gz_sim_interfaces.msg import PrairieCommand
from rclpy.node import Node

from .control_modes import KeyboardConfig, KeyboardMapper


class PrairieKeyboardTeleop(Node):
    KEY_BINDINGS = {
        pygame.K_w: "w",
        pygame.K_a: "a",
        pygame.K_s: "s",
        pygame.K_d: "d",
        pygame.K_q: "q",
        pygame.K_e: "e",
        pygame.K_0: "0",
        pygame.K_1: "1",
        pygame.K_2: "2",
        pygame.K_3: "3",
        pygame.K_4: "4",
        pygame.K_5: "5",
        pygame.K_KP0: "0",
        pygame.K_KP1: "1",
        pygame.K_KP2: "2",
        pygame.K_KP3: "3",
        pygame.K_KP4: "4",
        pygame.K_KP5: "5",
    }

    def __init__(self):
        super().__init__("prairie_keyboard_teleop")
        self.declare_parameter("vx_scale", KeyboardConfig.vx_scale)
        self.declare_parameter("vy_scale", KeyboardConfig.vy_scale)
        self.declare_parameter("yaw_rate_scale", KeyboardConfig.yaw_rate_scale)

        config = KeyboardConfig(
            vx_scale=self.get_parameter("vx_scale").value,
            vy_scale=self.get_parameter("vy_scale").value,
            yaw_rate_scale=self.get_parameter("yaw_rate_scale").value,
        )
        self.mapper = KeyboardMapper(config)
        self.publisher = self.create_publisher(
            PrairieCommand,
            "/prairie/user_command",
            10,
        )
        self.initialize_input_window()
        self.timer = self.create_timer(0.02, self.timer_callback)
        self.get_logger().info(
            "Keyboard control active in the pygame window: WASD translates, "
            "Q/E yaws, and number keys 0-5 select modes"
        )

    def initialize_input_window(self):
        pygame.display.init()
        pygame.font.init()
        self.window = pygame.display.set_mode((620, 150))
        pygame.display.set_caption("Prairie keyboard control")
        self.window.fill((25, 28, 34))
        font = pygame.font.Font(None, 28)
        lines = (
            "Keep this window focused while driving",
            "W/S: forward/back   A/D: left/right   Q/E: yaw",
            "1: sim stand   2: sim walk   0: emergency stop",
        )
        for index, line in enumerate(lines):
            label = font.render(line, True, (230, 235, 240))
            self.window.blit(label, (20, 18 + index * 38))
        pygame.display.flip()

    def timer_callback(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                self.update_key(event.key, pressed=True)
            elif event.type == pygame.KEYUP:
                self.update_key(event.key, pressed=False)
            elif event.type == pygame.WINDOWFOCUSLOST:
                self.mapper.clear_movement()
            elif event.type == pygame.QUIT:
                self.mapper.press("0")
                self.get_logger().warn(
                    "Keyboard window close requested; emergency stop active"
                )

        command = self.mapper.command()

        msg = PrairieCommand()
        msg.domain = command.domain
        msg.mode = command.mode
        msg.vx = command.vx
        msg.vy = command.vy
        msg.yaw_rate = command.yaw_rate
        self.publisher.publish(msg)

    def update_key(self, key_code, pressed):
        key = self.KEY_BINDINGS.get(key_code)
        if key is None:
            return
        if pressed:
            self.mapper.press(key)
        else:
            self.mapper.release(key)

    def close(self):
        self.mapper.clear_movement()
        pygame.quit()


def main(args=None):
    rclpy.init(args=args)
    node = PrairieKeyboardTeleop()
    try:
        rclpy.spin(node)
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

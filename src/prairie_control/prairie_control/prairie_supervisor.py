import time

import rclpy
from rclpy.node import Node
from gz_sim_interfaces.msg import MasterState, PrairieCommand, PrairieState

from .control_modes import (
    MODE_HOME,
    MODE_MIRROR,
    MODE_STAND,
    MODE_WALK,
    CommandIntent,
    SupervisorCore,
)


class PrairieSupervisor(Node):
    def __init__(self):
        super().__init__("prairie_supervisor")
        self.declare_parameter("allow_real_walk", False)
        self.core = SupervisorCore(
            allow_real_walk=self.get_parameter("allow_real_walk").value
        )
        self.last_warn_time = 0.0
        self.state_pub = self.create_publisher(PrairieState, "/prairie/state", 10)
        self.master_pub = self.create_publisher(MasterState, "/master_state", 10)
        self.command_sub = self.create_subscription(
            PrairieCommand,
            "/prairie/user_command",
            self.command_callback,
            10,
        )
        self.timer = self.create_timer(0.02, self.publish_state)

    def command_callback(self, msg: PrairieCommand):
        command = CommandIntent(
            domain=msg.domain,
            mode=msg.mode,
            vx=msg.vx,
            vy=msg.vy,
            yaw_rate=msg.yaw_rate,
        )
        _, accepted, reason = self.core.update(command)
        if not accepted and reason:
            self.warn_throttled(reason)
        self.publish_state()

    def publish_state(self):
        state = self.core.state

        prairie_state = PrairieState()
        prairie_state.sim_mode = state.sim_mode
        prairie_state.real_mode = state.real_mode
        prairie_state.vx = state.vx
        prairie_state.vy = state.vy
        prairie_state.yaw_rate = state.yaw_rate
        prairie_state.real_start_standing = state.real_start_standing
        self.state_pub.publish(prairie_state)

        master_state = MasterState()
        master_state.lx = self.safe_divide(state.vy, -0.3)
        master_state.ly = self.safe_divide(state.vx, 0.4)
        master_state.rx = self.safe_divide(state.yaw_rate, -0.8)
        master_state.ry = 0.0
        master_state.state1 = 1 if state.sim_mode == MODE_WALK else 0
        master_state.state2 = self.real_mode_to_legacy(state.real_mode)
        master_state.start_standing = state.real_start_standing
        self.master_pub.publish(master_state)

    def warn_throttled(self, message: str):
        now = time.time()
        if now - self.last_warn_time > 2.0:
            self.get_logger().warn(message)
            self.last_warn_time = now

    @staticmethod
    def safe_divide(value: float, scale: float) -> float:
        if scale == 0.0:
            return 0.0
        return max(-1.0, min(1.0, value / scale))

    @staticmethod
    def real_mode_to_legacy(mode: int) -> int:
        if mode == MODE_HOME:
            return 1
        if mode == MODE_STAND:
            return 2
        if mode == MODE_WALK:
            return 3
        if mode == MODE_MIRROR:
            return 0
        return 0


def main(args=None):
    rclpy.init(args=args)
    node = PrairieSupervisor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

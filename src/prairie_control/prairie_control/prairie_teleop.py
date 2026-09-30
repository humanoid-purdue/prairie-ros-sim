import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from gz_sim_interfaces.msg import PrairieCommand

from .control_modes import JoyMapper, TeleopConfig


class PrairieTeleop(Node):
    def __init__(self):
        super().__init__("prairie_teleop")
        self.declare_parameter("axis_left_x", TeleopConfig.axis_left_x)
        self.declare_parameter("axis_left_y", TeleopConfig.axis_left_y)
        self.declare_parameter("axis_right_x", TeleopConfig.axis_right_x)
        self.declare_parameter("button_a", TeleopConfig.button_a)
        self.declare_parameter("button_b", TeleopConfig.button_b)
        self.declare_parameter("button_x", TeleopConfig.button_x)
        self.declare_parameter("button_y", TeleopConfig.button_y)
        self.declare_parameter("button_lb", TeleopConfig.button_lb)
        self.declare_parameter("button_rb", TeleopConfig.button_rb)
        self.declare_parameter("vx_scale", TeleopConfig.vx_scale)
        self.declare_parameter("vy_scale", TeleopConfig.vy_scale)
        self.declare_parameter("yaw_rate_scale", TeleopConfig.yaw_rate_scale)
        self.declare_parameter("deadzone", TeleopConfig.deadzone)

        config = TeleopConfig(
            axis_left_x=self.get_parameter("axis_left_x").value,
            axis_left_y=self.get_parameter("axis_left_y").value,
            axis_right_x=self.get_parameter("axis_right_x").value,
            button_a=self.get_parameter("button_a").value,
            button_b=self.get_parameter("button_b").value,
            button_x=self.get_parameter("button_x").value,
            button_y=self.get_parameter("button_y").value,
            button_lb=self.get_parameter("button_lb").value,
            button_rb=self.get_parameter("button_rb").value,
            vx_scale=self.get_parameter("vx_scale").value,
            vy_scale=self.get_parameter("vy_scale").value,
            yaw_rate_scale=self.get_parameter("yaw_rate_scale").value,
            deadzone=self.get_parameter("deadzone").value,
        )
        self.mapper = JoyMapper(config)
        self.command = self.mapper.update([], [])

        self.publisher = self.create_publisher(
            PrairieCommand,
            "/prairie/user_command",
            10,
        )
        self.subscription = self.create_subscription(Joy, "/joy", self.joy_callback, 10)
        self.timer = self.create_timer(0.02, self.timer_callback)

    def joy_callback(self, msg: Joy):
        self.command = self.mapper.update(msg.axes, msg.buttons)

    def timer_callback(self):
        msg = PrairieCommand()
        msg.domain = self.command.domain
        msg.mode = self.command.mode
        msg.vx = self.command.vx
        msg.vy = self.command.vy
        msg.yaw_rate = self.command.yaw_rate
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PrairieTeleop()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

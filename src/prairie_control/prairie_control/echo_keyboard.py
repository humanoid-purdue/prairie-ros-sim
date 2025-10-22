import rclpy
from rclpy.node import Node
from gz_sim_interfaces.msg import KeyboardCmd

class EchoKeyboard(Node):
    def __init__(self):
        super().__init__('echo_keyboard')
        self.subscription = self.create_subscription(
            KeyboardCmd,
            'keyboard_cmd',
            self.cmd_callback,
            10
        )
        self.get_logger().info('EchoKeyboard: subscribed to topic "keyboard_cmd"')

    def cmd_callback(self, msg: KeyboardCmd):
        # Print the full message fields
        print("Fwd vel: ", msg.x, 
              " Turn left vel: ", msg.angz,
              " Strafe left vel: ", msg.y,
              " State: ", msg.state)

def main():
    rclpy.init()
    node = EchoKeyboard()
    try:
        rclpy.spin(node)
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
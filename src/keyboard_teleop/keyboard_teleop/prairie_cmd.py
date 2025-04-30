import os
import signal

import rclpy
from rclpy.node import Node
from pynput.keyboard import Key, Listener
from gz_sim_interfaces.msg import KeyboardCmd

class PrairieCmd(Node):
    def __init__(self):
        super().__init__('prairie_cmd')
        self.pub = self.create_publisher(KeyboardCmd, 'keyboard_cmd', 10)
        self.get_logger().info("PrairieCmd node started. Press number keys to set state, WASD/QE to move.")

        # start listener thread
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

        self.for_vel = 1.0
        self.back_vel = -1.0
        self.turn_left_vel = 1.0
        self.turn_right_vel = -1.0
        self.strafe_left_vel = 1.0
        self.strafe_right_vel = -1.0

        self.state = 0

        self.msg = KeyboardCmd()
        self.msg.state = self.state

    def on_release(self, key):
        send = True
        try:
            k = key.char
            if k == 'w':
                self.msg.x = 0.0
            if k == 's':
                self.msg.x = 0.0
            if k == 'a':
                self.msg.angz = 0.0
            if k == 'd':
                self.msg.angz = 0.0
            if k == 'q':
                self.msg.y = 0.0
            if k == 'e':
                self.msg.y = 0.0
        except AttributeError:
            # non‐printable key (e.g. arrows) – ignore
            send = False
        if send:
            self.pub.publish(self.msg)

    def on_press(self, key):
        send = True
        try:
            k = key.char
            # forward / backward
            if k == 'w':
                self.msg.x = self.for_vel
            elif k == 's':
                self.msg.x = self.back_vel

            # turn left / turn right
            if k == 'a':
                self.msg.angz = self.turn_left_vel
            elif k == 'd':
                self.msg.angz = self.turn_right_vel

            # strafe left / strafe right
            if k == 'q':
                self.msg.y = self.strafe_left_vel
            elif k == 'e':
                self.msg.y = self.strafe_right_vel

            # number keys set the state field
            if k.isdigit():
                self.state = int(k)

        except AttributeError:
            # non‐printable key (e.g. arrows) – ignore
            send = False

        if send:
            self.msg.state = self.state
            self.pub.publish(self.msg)

def main():
    rclpy.init()
    node = PrairieCmd()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
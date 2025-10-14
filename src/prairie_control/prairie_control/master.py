import rclpy
from rclpy.node import Node
import numpy as np
import os
import sys
import time
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile
from builtin_interfaces.msg import Duration, Time
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from gz_sim_interfaces.msg import StateObservationReduced, MasterState
from geometry_msgs.msg import Twist

helper_path = os.path.join(
            get_package_share_directory('prairie_control'),
            "helpers")

sys.path.append(helper_path)
import xbox

class master(Node):
    def __init__(self):
        super().__init__('master')
        qos_profile = QoSProfile(depth=10)

        # two jtp publishers

        self.state = 0

        self.gz_pub = self.create_publisher(JointTrajectory, 'joint_trajectories', qos_profile)
        self.master_pub = self.create_publisher(MasterState, 'master_state', qos_profile)

        # Timer callback to publish the two JTPs

        self.timer = self.create_timer(0.002, self.timer_callback)

        # Subscribers to multiple JTPs

        self.gz_standing = self.create_subscription(
            JointTrajectory, '/gz_standing_jtp',
            self.gz_standing_callback,
            qos_profile
        )

        self.gz_policy = self.create_subscription(
            JointTrajectory, '/gz_policy_jtp',
            self.gz_policy_callback,
            qos_profile
        )

        self.gz_policy_jtp = None

        self.gz_stand_jtp = None

        self.ctrl = xbox.XboxController()

    def gz_standing_callback(self, msg):
        self.gz_stand_jtp = msg

    def gz_policy_callback(self, msg):
        self.gz_policy_jtp = msg

    def timer_callback(self):
        self.update_xbox()
        if self.gz_stand_jtp is not None and self.state == 0:
            self.gz_pub.publish(self.gz_stand_jtp)
        elif self.gz_policy_jtp is not None and self.state == 1:
            self.gz_pub.publish(self.gz_policy_jtp)

    def update_xbox(self):
        self.ctrl.update()
        lx = self.ctrl.left_stick[0]
        ly = self.ctrl.left_stick[1]
        rx = self.ctrl.right_stick[0]
        ry = self.ctrl.right_stick[1]

        self.state = self.ctrl.state

        master_state = MasterState()
        master_state.lx = lx
        master_state.ly = ly
        master_state.rx = rx
        master_state.ry = ry
        master_state.state = self.ctrl.state
        self.master_pub.publish(master_state)
        return



def main(args=None):
    rclpy.init(args=args)

    node = master()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
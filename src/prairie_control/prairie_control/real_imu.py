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
from gz_sim_interfaces.msg import StateObservationReduced
from geometry_msgs.msg import Twist

helper_path = os.path.join(
            get_package_share_directory('prairie_control'),
            "helpers")

sys.path.append(helper_path)
import imu_wrapper

class real_imu(Node):
    def __init__(self):
        super().__init__('real_imu')
        qos_profile = QoSProfile(depth=10)

        # one publiser for StateObservationReduced
        self.state_pub = self.create_publisher(StateObservationReduced, '/imu_observation', qos_profile)

        self.imu = imu_wrapper.imu_wrapper()

        self.timer = self.create_timer(0.005, self.timer_callback)

    def timer_callback(self):
        imu_msg = StateObservationReduced()

        accel, gyro = self.imu.read_accel_gyro()

        imu_msg.lin_acc = accel.tolist()
        imu_msg.ang_vel = gyro.tolist()

        #self.get_logger().info(f"IMU Accel: {imu_msg.lin_acc}, Gyro: {imu_msg.ang_vel}")

        self.state_pub.publish(imu_msg)

def main(args=None):
    rclpy.init(args=args)

    node = real_imu()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
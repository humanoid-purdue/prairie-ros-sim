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
from sensor_msgs.msg import Imu
from gz_sim_interfaces.msg import StateObservationReduced
from geometry_msgs.msg import Twist, Vector3

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
        self.state_pub = self.create_publisher(Imu, '/imu_observation', qos_profile)

        self.imu = imu_wrapper.imu_wrapper()

        self.timer = self.create_timer(0.005, self.timer_callback)

    def timer_callback(self):
        imu_msg = Imu()

        accel, gyro = self.imu.read_accel_gyro()
        lin_acc = Vector3()
        lin_acc.x = accel[0]
        lin_acc.y = accel[1]
        lin_acc.z = accel[2]
        imu_msg.linear_acceleration = lin_acc
        ang_vel = Vector3()
        ang_vel.x = gyro[0]
        ang_vel.y = gyro[1]
        ang_vel.z = gyro[2]
        imu_msg.angular_velocity = ang_vel

        self.get_logger().info(f"IMU Accel: {lin_acc}, Gyro: {ang_vel}")

        self.state_pub.publish(imu_msg)

def main(args=None):
    rclpy.init(args=args)

    node = real_imu()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
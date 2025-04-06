from math import sin, cos, pi
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from tf2_ros import TransformBroadcaster, TransformStamped
import numpy as np
import time
from ament_index_python.packages import get_package_share_directory
import os, sys

JOINT_LIST_COMPLETE = ["l_hip_pitch_joint", "l_hip_roll_joint", "l_hip_yaw_joint", "l_knee_joint", "l_foot_pitch_joint", "l_foot_roll_joint",
                       "r_hip_pitch_joint", "r_hip_roll_joint", "r_hip_yaw_joint", "r_knee_joint", "r_foot_pitch_joint", "r_foot_roll_joint"]

class default_pd(Node):

    def __init__(self):
        rclpy.init()
        super().__init__('state_publisher')

        qos_profile = QoSProfile(depth=10)
        self.joint_pub = self.create_publisher(JointTrajectory, 'joint_trajectories', qos_profile)
        self.broadcaster = TransformBroadcaster(self, qos=qos_profile)
        self.nodeName = self.get_name()
        self.get_logger().info("{0} started".format(self.nodeName))

        loop_rate = self.create_rate(30)
        joint_traj = JointTrajectory()
        jtp = JointTrajectoryPoint()
        duration = Duration()
        try:
            while rclpy.ok():
                rclpy.spin_once(self)

                # update joint_state
                now = self.get_clock().now()
                joint_traj.header.stamp = now.to_msg()
                joint_traj.joint_names = JOINT_LIST_COMPLETE
                
                jtp.time_from_start = Duration()
                jtp.positions = [0.] * len(JOINT_LIST_COMPLETE)
                jtp.velocities = [0.] * len(JOINT_LIST_COMPLETE)
                jtp.effort = [0.] * len(JOINT_LIST_COMPLETE)
                joint_traj.points = [jtp]
                self.joint_pub.publish(joint_traj)

                loop_rate.sleep()

        except KeyboardInterrupt:
            pass

def main():
    node = default_pd()

if __name__ == '__main__':
    main()
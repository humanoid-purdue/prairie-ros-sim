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
import utils
from utils import JOINT_LIST_COMPLETE

class gz_mirror(Node):
    def __init__(self):
        super().__init__('gz_mirror')
        qos_profile = QoSProfile(depth=10)

        self.state_subscriber = self.create_subscription(
            StateObservationReduced,
            '/gz_state_observation',
            self.state_callback,
            qos_profile
        )
        self.obs = {}

        self.joint_pub = self.create_publisher(JointTrajectory, 'gz_mirror_jtp', qos_profile)

        self.timer = self.create_timer(0.002, self.timer_callback)

    def state_callback(self, msg):
        self.obs = utils.fill_obs_dict(msg)
        return
    
    def timer_callback(self):
        if self.obs != {}:
            pos_t = self.obs['joint_position'].copy()
            pos_v = self.obs['joint_velocity'].copy()

            joint_traj = JointTrajectory()
            jtp = JointTrajectoryPoint()
            jtp2 = JointTrajectoryPoint()

            
            now = self.get_clock().now()
            joint_traj.header.stamp = now.to_msg()
            joint_traj.joint_names = utils.JOINT_LIST_COMPLETE
                    
            jtp.time_from_start = Duration()
            jtp.time_from_start.sec = 0
            jtp.time_from_start.nanosec = 0

            jtp2.time_from_start = Duration()
            jtp2.time_from_start.sec = 0
            jtp2.time_from_start.nanosec = 0

            jtp.positions = pos_t.tolist()
            jtp.velocities = pos_v.tolist()
            jtp.effort = [0.] * 18

            jtp2.positions = pos_t.tolist()
            jtp2.velocities = pos_v.tolist()
            jtp2.effort = [0.] * 18
            jtp2.time_from_start.sec = 100
            joint_traj.points = [jtp, jtp2]

            self.joint_pub.publish(joint_traj)

def main(args=None):
    rclpy.init(args=args)

    node = gz_mirror()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
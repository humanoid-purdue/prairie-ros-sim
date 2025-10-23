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
import utils

ACC_ALPHA = 0.4
ANG_ALPHA = 0.4
POS_ALPHA = 0.1
VEL_ALPHA = 0.2

class real_state_est(Node):
    def __init__(self):
        super().__init__('real_state_est')
        qos_profile = QoSProfile(depth=10)

        # Subscribe to real_imu and the joint state publisher
        self.imu_subscriber = self.create_subscription(
            StateObservationReduced,
            '/imu_observation',
            self.imu_callback,
            qos_profile
        )

        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            qos_profile
        )

        # publish obs
        self.obs_publisher = self.create_publisher(StateObservationReduced, '/real_state_observation', qos_profile)

        self.lin_acc = np.array([0.0, 0.0, 9.81])
        self.ang_vel = np.zeros(3)

        self.joint_pos = np.zeros(len(utils.JOINT_LIST_COMPLETE))
        self.joint_vel = np.zeros(len(utils.JOINT_LIST_COMPLETE))

        self.timer = self.create_timer(0.005, self.timer_callback)


    def imu_callback(self, msg):
        lin_acc = np.array(msg.lin_acc)
        ang_vel = np.array(msg.ang_vel)
        self.lin_acc = self.lin_acc * ACC_ALPHA + lin_acc * (1 - ACC_ALPHA)
        self.ang_vel = self.ang_vel * ANG_ALPHA + ang_vel * (1 - ANG_ALPHA)

    def joint_state_callback(self, msg):
        joint_vel_dict = dict(zip(msg.name, msg.velocity))
        joint_pos_dict = dict(zip(msg.name, msg.position))

        for c in range(len(utils.JOINT_LIST_COMPLETE)):
            name = utils.JOINT_LIST_COMPLETE[c]
            if name not in joint_vel_dict:
                joint_vel_dict[name] = 0.0
            if name not in joint_pos_dict:
                joint_pos_dict[name] = 0.0
            self.joint_pos[c] = self.joint_pos[c] * POS_ALPHA + joint_pos_dict[name] * (1 - POS_ALPHA)
            self.joint_vel[c] = self.joint_vel[c] * VEL_ALPHA + joint_vel_dict[name] * (1 - VEL_ALPHA)
            
    def timer_callback(self):
        obs_msg = StateObservationReduced()

        obs_msg.joint_name = utils.JOINT_LIST_COMPLETE
        obs_msg.lin_acc = self.lin_acc.tolist()
        obs_msg.ang_vel = self.ang_vel.tolist()
        obs_msg.joint_pos = self.joint_pos.tolist()
        obs_msg.joint_vel = self.joint_vel.tolist()

        # self.get_logger().info(f"Publishing real_state_observation: lin_acc={obs_msg.lin_acc}, ang_vel={obs_msg.ang_vel}")

        self.obs_publisher.publish(obs_msg)

def main(args=None):
    rclpy.init(args=args)

    node = real_state_est()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
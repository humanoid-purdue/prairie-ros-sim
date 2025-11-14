import rclpy
from rclpy.node import Node
import numpy as np
import os
import sys
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from gz_sim_interfaces.msg import StateObservationReduced, MasterState

helper_path = os.path.join(get_package_share_directory('prairie_control'), "helpers")
sys.path.append(helper_path)
import utils
from stabilizer import Stabilizer

JOINT_LIST_COMPLETE = ["l_hip_pitch_joint", "l_hip_roll_joint", "l_hip_yaw_joint", "l_knee_joint", "l_foot_pitch_joint",
                       "l_foot_roll_joint",
                       "r_hip_pitch_joint", "r_hip_roll_joint", "r_hip_yaw_joint", "r_knee_joint", "r_foot_pitch_joint",
                       "r_foot_roll_joint",
                       "l_shoulder_pitch_joint", "l_shoulder_roll_joint", "l_elbow_joint",
                       "r_shoulder_pitch_joint", "r_shoulder_roll_joint", "r_elbow_joint"]
BASE_COM = np.array([-0.04, 0, 0.4])
COM_OFFSET = np.array([0.0, 0.0, 0.0])
STANDING_COM = BASE_COM + COM_OFFSET
STAND_TIME = 3

class real_standing(Node):
    def __init__(self):
        super().__init__('real_standing')
        qos_profile = QoSProfile(depth=10)

        # Read IMU data and publish JTP

        self.state_subscriber = self.create_subscription(
            StateObservationReduced,
            '/real_state_observation',
            self.state_callback,
            qos_profile
        )

        self.master_subscriber = self.create_subscription(
            MasterState,
            '/master_state',
            self.master_callback,
            qos_profile
        )

        self.start_time = None
        self.start_com = None
        self.start_standing = False

        self.num_joints = len(JOINT_LIST_COMPLETE)
        self.ids = {name: index for index, name in enumerate(JOINT_LIST_COMPLETE)}
        self.stabilizer = Stabilizer()

        self.joint_pub = self.create_publisher(JointTrajectory, 'gz_standing_jtp', qos_profile)
        self.timer = self.create_timer(0.002, self.timer_callback)
        self.obs = {}

    def state_callback(self, msg):
        self.obs = utils.fill_obs_dict(msg)
        return

    def master_callback(self, msg):
        self.start_standing = msg.start_standing
        return

    def timer_callback(self):
        joint_traj = JointTrajectory()
        jtp = JointTrajectoryPoint()
        jtp2 = JointTrajectoryPoint()

        now = self.get_clock().now()
        joint_traj.header.stamp = now.to_msg()
        joint_traj.joint_names = JOINT_LIST_COMPLETE

        jtp.time_from_start = Duration()
        jtp.time_from_start.sec = 0
        jtp.time_from_start.nanosec = 0

        jtp2.time_from_start = Duration()
        jtp2.time_from_start.sec = 0
        jtp2.time_from_start.nanosec = 0

        tau_delta = np.zeros(self.num_joints)
        pos_t = np.zeros(self.num_joints)

        if self.obs != {}:
            pos_t = self.obs['joint_position']
            self.stabilizer.update_simulation(self.obs['joint_position'], self.obs['joint_velocity'])
            t = self.obs['time']
            if self.start_standing and not self.start_time:
                self.start_time = t
                self.start_com = self.stabilizer.get_relative_com()
            desired_com = STANDING_COM
            if self.start_standing and t < self.start_time + STAND_TIME:
                desired_com = self.start_com + ((t - self.start_time) / STAND_TIME) * (STANDING_COM - self.start_com)
            tau_delta = self.stabilizer.calculate_joint_torques(desired_com)


        jtp.positions = pos_t.tolist()
        jtp.velocities = [0.] * self.num_joints
        jtp.effort = tau_delta.tolist()

        jtp2.positions = pos_t.tolist()
        jtp2.velocities = [0.] * self.num_joints
        jtp2.effort = tau_delta.tolist()
        jtp2.time_from_start.sec = 100
        joint_traj.points = [jtp, jtp2]

        self.joint_pub.publish(joint_traj)


def main(args=None):
    rclpy.init(args=args)

    node = real_standing()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
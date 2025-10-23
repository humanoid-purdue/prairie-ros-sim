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
import xbox, utils

class master(Node):
    def __init__(self):
        super().__init__('master')
        qos_profile = QoSProfile(depth=10)

        # two jtp publishers

        self.state1 = 0
        self.state2 = 0

        self.gz_pub = self.create_publisher(JointTrajectory, 'joint_trajectories', qos_profile)
        self.joint_pub = self.create_publisher(JointTrajectory, '/real_joint_trajectories', qos_profile)
        
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

        self.gz_mirror = self.create_subscription(
            JointTrajectory, '/gz_mirror_jtp',
            self.gz_mirror_callback,
            qos_profile
        )

        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            qos_profile
        )

        self.joint_pos = np.zeros([18])

        self.gz_policy_jtp = None

        self.gz_stand_jtp = None

        self.gz_mirror_jtp = None

        try:
            self.ctrl = xbox.XboxController()
        except RuntimeError:
            self.ctrl = None

        self.start_time_gradual_hold = time.time()

        self.start_pos_gradual_hold = np.zeros([18])

        self.home_pose = np.array([-0.698132,
                        0,
                        0,
                        1.22173,
                        -0.523599,
                        0,
                        -0.698132,
                        0,
                        0,
                        1.22173,
                        -0.523599, 0,
                        0, 0.05, 0,
                        0, -0.05, 0])

    def gz_standing_callback(self, msg):
        self.gz_stand_jtp = msg

    def gz_policy_callback(self, msg):
        self.gz_policy_jtp = msg

    def gz_mirror_callback(self, msg):
        self.gz_mirror_jtp = msg

    def timer_callback(self):
        self.update_xbox()
        if self.gz_stand_jtp is not None and self.state1 == 0:
            self.gz_pub.publish(self.gz_stand_jtp)
        elif self.gz_policy_jtp is not None and self.state1 == 1:
            self.gz_pub.publish(self.gz_policy_jtp)

        real_jtp = self.default_real_pd()
        self.joint_pub.publish(real_jtp)

    def update_xbox(self):
        if not self.ctrl: return
        self.ctrl.update()
        if self.ctrl.state2 == 1 and self.state2 != 1:
            self.start_stand_pd()

        lx = self.ctrl.left_stick[0]
        ly = self.ctrl.left_stick[1]
        rx = self.ctrl.right_stick[0]
        ry = self.ctrl.right_stick[1]

        self.state1 = self.ctrl.state1
        self.state2 = self.ctrl.state2

        master_state = MasterState()
        master_state.lx = lx
        master_state.ly = ly
        master_state.rx = rx
        master_state.ry = ry
        master_state.state1 = self.ctrl.state1
        master_state.state2 = self.ctrl.state2
        self.master_pub.publish(master_state)
        return
    
    def default_real_pd(self):
        print(self.state2)
        if self.state2 == 1:
            pos_t = self.stand_pd()
            jtp = self.pos_t2traj(pos_t)
        elif self.state2 == 2 and self.gz_mirror_jtp is not None:
            jtp = self.gz_mirror_jtp
        else:
            pos_t = np.zeros([18])
            jtp = self.pos_t2traj(pos_t)
        return jtp

    def pos_t2traj(self, pos_t):
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
        jtp.velocities = [0.] * 18
        jtp.effort = [0.] * 18

        jtp2.positions = pos_t.tolist()
        jtp2.velocities = [0.] * 18
        jtp2.effort = [0.] * 18
        jtp2.time_from_start.sec = 100
        joint_traj.points = [jtp, jtp2]
        return joint_traj

    def stand_pd(self):
        time_coeff = min((time.time() - self.start_time_gradual_hold) / 2.0 , 1.0)
        pos_t = self.home_pose * time_coeff + (1 - time_coeff) * self.start_pos_gradual_hold
        return pos_t
    
    def start_stand_pd(self):
        self.start_pos_gradual_hold = self.joint_pos.copy()
        self.start_time_gradual_hold = time.time()

    def joint_state_callback(self, msg):
        for i in range(12):
            self.joint_pos[i] = msg.position[i]

    
def main(args=None):
    rclpy.init(args=args)

    node = master()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
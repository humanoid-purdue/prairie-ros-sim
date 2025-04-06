import rclpy
from rclpy.node import Node
import numpy as np
import os
import sys
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile
from builtin_interfaces.msg import Duration, Time
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from gz_sim_interfaces.msg import StateObservationReduced

JOINT_LIST_COMPLETE = ["l_hip_pitch_joint", "l_hip_roll_joint", "l_hip_yaw_joint", "l_knee_joint", "l_foot_pitch_joint", "l_foot_roll_joint",
                       "r_hip_pitch_joint", "r_hip_roll_joint", "r_hip_yaw_joint", "r_knee_joint", "r_foot_pitch_joint", "r_foot_roll_joint"]

class home_pd(Node):
    def __init__(self):
        super().__init__('home_pd')
        qos_profile = QoSProfile(depth=10)
        self.state_subscriber = self.create_subscription(
            StateObservationReduced,
            '/state_observation',
            self.state_callback,
            qos_profile
        )
        self.joint_pub = self.create_publisher(JointTrajectory, 'joint_trajectories', qos_profile)
        self.timer = self.create_timer(0.0001, self.timer_callback)
        self.lin_vel = np.zeros([3])
        self.joint_pos = np.zeros([12])
        self.joint_vel = np.zeros([12])
        self.ang_vel = np.zeros([3])
        self.grav_vec = np.zeros([3])
        self.lin_acc = np.zeros([3])
        self.state_time = 0.0

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
                                        -0.523599,
                                        0])
        
        self.ankle_l_id = 4
        self.ankle_r_id = 10

    def state_callback(self, obs):
        self.joint_pos = np.array(obs.joint_pos)
        self.joint_vel = np.array(obs.joint_vel)
        self.ang_vel = np.array(obs.ang_vel)
        self.grav_vec = np.array(obs.grav_vec)
        self.lin_vel = np.array(obs.lin_vel)
        self.state_time = obs.time
        self.lin_acc = np.array(obs.lin_acc)
        return

    def timer_callback(self):

        joint_traj = JointTrajectory()
        jtp = JointTrajectoryPoint()

        time_coeff = min(self.state_time / 0.5 , 1.0)
        now = self.get_clock().now()
        joint_traj.header.stamp = now.to_msg()
        joint_traj.joint_names = JOINT_LIST_COMPLETE
                
        jtp.time_from_start = Duration()

        jtp_ankle_pd = self.lin_acc[0] * -1
        pos_delta = np.zeros([12])
        pos_delta[self.ankle_l_id] = jtp_ankle_pd
        pos_delta[self.ankle_r_id] = jtp_ankle_pd
        
        pos2 = self.home_pose * time_coeff
        pos2 = pos2

        jtp.positions = pos2

        efforts = [0.] * len(JOINT_LIST_COMPLETE)


        jtp.velocities = [0.] * len(JOINT_LIST_COMPLETE)
        jtp.effort = pos_delta.tolist()
        joint_traj.points = [jtp]
        self.joint_pub.publish(joint_traj)

        return
    
def main(args=None):
    rclpy.init(args=args)

    hpd = home_pd()

    rclpy.spin(hpd)
    hpd.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
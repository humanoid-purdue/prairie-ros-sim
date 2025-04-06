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
helper_path = os.path.join(
            get_package_share_directory('prairie_control'),
            "helpers")

sys.path.append(helper_path)
import policy_network

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
        self.wpn = policy_network.walk_policy(t = 0.0)

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

        
        now = self.get_clock().now()
        joint_traj.header.stamp = now.to_msg()
        joint_traj.joint_names = JOINT_LIST_COMPLETE
                
        jtp.time_from_start = Duration()

        tau_delta = np.zeros([12])
        vel_t = np.zeros([12])

        if self.state_time < 3.0:
            pos_t, tau_delta = self.stand_pd()
        else:
            pos_t, vel_t = self.walk_gz()


        jtp.positions = pos_t.tolist()



        jtp.velocities = vel_t.tolist()
        jtp.effort = tau_delta.tolist()
        joint_traj.points = [jtp]
        self.joint_pub.publish(joint_traj)

        return
    

    def stand_pd(self):
        time_coeff = min(self.state_time / 0.5 , 1.0)
        jtp_ankle_pd = self.lin_acc[0] * -2
        tau_delta = np.zeros([12])
        tau_delta[self.ankle_l_id] = jtp_ankle_pd
        tau_delta[self.ankle_r_id] = jtp_ankle_pd
        pos_t = self.home_pose * time_coeff
        return pos_t, tau_delta
    
    def walk_gz(self):
        if self.state_time < 3.0:
            self.wpn.reinit(t = self.state_time)
            pos = np.zeros([12])
            vel = np.zeros([12])
        else:
            vel_target = np.array([0.3, 0.0])
            angvel_target = np.array([0.0])
            pos, vel = self.wpn.apply_net(self.joint_pos,
                               self.joint_vel,
                               self.ang_vel,
                               self.grav_vec,
                               self.lin_vel,
                               vel_target,
                               angvel_target,
                               0,
                               self.state_time)
            pos = np.array(pos)
            vel = np.array(vel)
        return pos, vel

def main(args=None):
    rclpy.init(args=args)

    hpd = home_pd()

    rclpy.spin(hpd)
    hpd.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
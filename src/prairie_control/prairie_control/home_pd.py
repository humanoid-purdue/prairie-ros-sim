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
from geometry_msgs.msg import Twist
import time

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
        self.keyboard_subscriber = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.keyboard_callback,
            qos_profile
        )
        self.joint_pub = self.create_publisher(JointTrajectory, 'joint_trajectories', qos_profile)
        self.timer = self.create_timer(0.002, self.timer_callback)
        self.lin_vel = np.zeros([3])
        self.joint_pos = np.zeros([12])
        self.joint_vel = np.zeros([12])
        self.ang_vel = np.zeros([3])
        self.grav_vec = np.zeros([3])
        self.lin_acc = np.zeros([3])
        self.cmd_vel = np.zeros([2])
        self.cmd_angvel = np.zeros([1])
        self.halt = 1
        self.prev_halt = 1
        self.true_halt = 1
        self.halt_start_t = time.time() - 0.5
        self.state_time = 0.0
        self.prev_time = 0.0

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
        
        self.p_gains = np.array([50, 40, 40, 50, 50, 40, 50, 40, 40, 50, 50, 40.])
        self.k_gains = np.array([4., 2., 2., 4., 4., 2., 4., 2., 2., 4., 4., 2.])

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

        tau_delta = np.zeros([12])
        vel_t = np.zeros([12])

        if self.state_time - self.prev_time > 0.02:
            self.prev_time = self.state_time
            if self.state_time < 3.0:
                pos_t, tau_delta = self.stand_pd()
            else:
                pos_t, vel_t = self.walk_gz()


            jtp.positions = pos_t.tolist()
            jtp.velocities = vel_t.tolist()
            jtp.effort = tau_delta.tolist()

            jtp2.positions = pos_t.tolist()
            jtp2.velocities = vel_t.tolist()
            jtp2.effort = tau_delta.tolist()
            jtp2.time_from_start.sec = 99999999
            joint_traj.points = [jtp, jtp2]
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
    
    def keyboard_callback(self, msg):
        self.cmd_vel[0] = msg.linear.x * 0.4
        self.cmd_vel[1] = msg.linear.y * 0.0
        self.cmd_angvel[0] = msg.angular.z * 0.7
        if (msg.linear.x == 0.0 and msg.linear.y == 0.0 and msg.angular.z == 0.0):
            self.halt = 1
        else:
            self.halt = 0
        return
    
    def walk_gz(self):
        self.update_halt()
        if self.state_time < 3.0:
            self.wpn.reinit(t = self.state_time)
            pos = np.zeros([12])
            vel = np.zeros([12])
        else:
            if (self.cmd_angvel[0] == 0.0 and self.cmd_vel[0] != 0.0):
                self.cmd_angvel[0] = max(min(-self.ang_vel[2] * 1.1, 0.7), -0.7)
                
            pos, vel = self.wpn.apply_net(self.joint_pos,
                               self.joint_vel,
                               self.ang_vel,
                               self.grav_vec,
                               self.lin_acc,
                               self.cmd_vel,
                               self.cmd_angvel,
                               self.halt,
                               self.state_time)
            pos = np.array(pos)
            vel = np.array(vel)
            
        return pos, vel
    
    def update_halt(self):
        if self.halt != self.prev_halt:
            self.halt_start_t = time.time()
        if (self.halt == 1):
            if (time.time() - self.halt_start_t > 0.5):
                self.true_halt = 1
            else:
                self.true_halt = 0
        else:
            self.true_halt = 0
        self.prev_halt = self.halt
        return

def main(args=None):
    rclpy.init(args=args)

    hpd = home_pd()

    rclpy.spin(hpd)
    hpd.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
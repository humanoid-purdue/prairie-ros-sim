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
from gz_sim_interfaces.msg import KeyboardCmd
from geometry_msgs.msg import Twist

JOINT_LIST_COMPLETE = ["l_hip_pitch_joint", "l_hip_roll_joint", "l_hip_yaw_joint", "l_knee_joint", "l_foot_pitch_joint", "l_foot_roll_joint",
                       "r_hip_pitch_joint", "r_hip_roll_joint", "r_hip_yaw_joint", "r_knee_joint", "r_foot_pitch_joint", "r_foot_roll_joint"]

data_path = os.path.join(
            get_package_share_directory('prairie_control'),
            "data")

# STATES: 1 = Start home pd, 2 = Gradual home pd, 3 = Start PD to playback, 4 = PD playback

class static_pd(Node):
    def __init__(self):
        super().__init__('static_pd')
        qos_profile = QoSProfile(depth=10)
        self.keyboard_subscriber = self.create_subscription(
            KeyboardCmd,
            '/keyboard_cmd',
            self.keyboard_callback,
            qos_profile
        )

        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            qos_profile
        )
        self.state_subscriber = self.create_subscription(
            StateObservationReduced,
            '/state_observation',
            self.state_callback,
            qos_profile
        )
        self.nemo_traj = np.genfromtxt(os.path.join(data_path, 'joint_traj.csv'), delimiter = ',')
        self.joint_pos = np.zeros([12])
        self.joint_pub = self.create_publisher(JointTrajectory, '/real_joint_trajectories', qos_profile)
        self.timer = self.create_timer(0.002, self.timer_callback)
        self.start_time_gradual_hold = time.time()
        self.start_time_pd_playback = time.time()
        self.start_pos_gradual_hold = np.zeros([12])
        self.state = 0

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
                                        -0.523599, 0])
        
        self.gz_joint_pos = self.home_pose.copy()
        self.gz_joint_vel = np.zeros([12])
        
    def state_callback(self, obs):
        self.gz_joint_pos = np.array(obs.joint_pos)
        self.gz_joint_vel = np.array(obs.joint_vel)
        return
    
    def gz_link(self):
        pos_error = self.gz_joint_pos - self.joint_pos
        max_delta = 0.1
        pos_error = np.clip(pos_error, -max_delta, max_delta)
        pos2 = pos_error + self.joint_pos
        return pos2, self.gz_joint_vel
        

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
        pos_t = np.zeros([12])

        if self.state == 1:
            self.start_stand_pd()
        elif self.state == 2:
            pos_t = self.stand_pd()
        elif self.state == 3:
            self.start_playback()
        elif self.state == 4:
            pos_t, vel_t = self.execute_playback()
        elif self.state == 5:
            pos_t, vel_t = self.gz_link()

        print(pos_t)

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
        time_coeff = min((time.time() - self.start_time_gradual_hold) / 2.0 , 1.0)
        pos_t = self.home_pose * time_coeff + (1 - time_coeff) * self.start_pos_gradual_hold
        return pos_t
    
    def start_stand_pd(self):
        self.start_pos_gradual_hold = self.joint_pos.copy()
        self.start_time_gradual_hold = time.time()
        self.state = 2

    def start_playback(self):
        self.start_time_pd_playback = time.time()
        self.state = 4

    def execute_playback(self):
        time_delta = time.time() - self.start_time_pd_playback
        index = round(time_delta / 0.001)
        if (index < 0):
            index = 0
        if (index > 59999):
            index = 59999
            self.state = 0
        pos_t = self.nemo_traj[index, 1:12]
        vel_t = self.nemo_traj[index, 12:23]
        return pos_t, vel_t
    
    def joint_state_callback(self, msg):
        for i in range(12):
            self.joint_pos[i] = msg.position[i]

    def keyboard_callback(self, msg):
        print("keyboard callback")
        if msg.state == 1 and self.state != 2:
            self.state = 1
        if msg.state == 0:
            self.state = 0
        if msg.state == 2 and self.state != 3:
            self.state = 3
        if msg.state == 3:
            self.state = 5
        return
    
def main(args=None):
    rclpy.init(args=args)

    hpd = static_pd()

    rclpy.spin(hpd)
    hpd.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
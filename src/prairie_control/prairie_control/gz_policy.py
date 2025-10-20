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
import policy_network, utils
from utils import JOINT_LIST_COMPLETE

class gz_policy(Node):
    def __init__(self):
        super().__init__('gz_policy')
        self.prev_time = time.time()
        qos_profile = QoSProfile(depth=10)

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

        self.state_subscriber = self.create_subscription(
            StateObservationReduced,
            '/gz_state_observation',
            self.state_callback,
            qos_profile
        )

        self.state = 0

        self.cmd = np.array([0.0, 0.0, 0.0])

        self.master_subscriber = self.create_subscription(
            MasterState,
            '/master_state',
            self.master_callback,
            qos_profile
        )

        self.joint_pub = self.create_publisher(JointTrajectory, 'gz_policy_jtp', qos_profile)
        self.timer = self.create_timer(0.02, self.timer_callback)
        self.obs = {}

        self.wpn = policy_network.walk_policy(t = 0.0)

    def state_callback(self, msg):
        self.obs = utils.fill_obs_dict(msg)
        return 
    
    def master_callback(self, msg):
        self.state = msg.state1
        vel = np.array([msg.ly, msg.lx]) * np.array([0.4, -0.3])
        angvel = np.array([msg.rx]) * -0.8
        self.cmd = np.hstack((vel, angvel))
        return
    
    def timer_callback(self):
        if self.obs == {}:
            return
        if self.state == 0:
            self.wpn.reinit(t = self.obs['time'])
        action = self.wpn.apply_net(
            self.obs["joint_position"], 
            self.obs["joint_velocity"], 
            self.obs["angular_velocity"],  
            self.obs["linear_acceleration"], 
            self.cmd, 
            self.obs["time"]
        )

        np_action = np.array(action)
        self.prev_time = time.time()


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

        jtp.positions = np_action.tolist()
        jtp.velocities = [0.] * 18
        jtp.effort = [0.] * 18

        jtp2.positions = np_action.tolist()
        jtp2.velocities = [0.] * 18
        jtp2.effort = [0.] * 18
        jtp2.time_from_start.sec = 100
        joint_traj.points = [jtp, jtp2]
        #print(tau_delta)
        self.joint_pub.publish(joint_traj)



def main(args=None):
    rclpy.init(args=args)

    node = gz_policy()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
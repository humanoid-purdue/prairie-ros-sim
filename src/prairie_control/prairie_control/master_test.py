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
from gz_sim_interfaces.msg import MotorCmd
from sensor_msgs.msg import JointState

helper_path = os.path.join(
            get_package_share_directory('prairie_control'),
            "helpers")

sys.path.append(helper_path)
import utils

TIME_TO_HOME = 2.0 #Seconds
START_TIME = 5.0 #Seconds

class master(Node):
    def __init__(self):
        super().__init__('master_test')
        qos_profile = QoSProfile(depth=10)

        self.joint_pub = self.create_publisher(MotorCmd, '/real_joint_trajectories', qos_profile)

        self.timer = self.create_timer(0.002, self.timer_callback)

        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            qos_profile
        )

        self.joint_pos = None

        self.start_pos = None

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

    def joint_state_callback(self, msg):
        if not self.joint_pos: self.joint_pos = np.zeros([18])
        for i in range(12):
            self.joint_pos[i] = msg.position[i]

    def timer_callback(self):
        pos_t = None
        if time.time() > START_TIME and self.joint_pos is not None:
            if self.start_pos is None:
                self.start_pos = self.joint_pos.copy()
                print("master_test: STARTING MOVEMENT...")
            time_coeff = min((time.time() - START_TIME) / TIME_TO_HOME, 1.0)
            pos_t = self.start_pos + time_coeff * (self.home_pose - self.start_pos)
        mcmd = self.pos_t2mcmd(pos_t)
        self.joint_pub.publish(mcmd)

    def pos_t2mcmd(self, pos_t=None):
        mcmd = MotorCmd()
        mcmd.joint_names = utils.JOINT_LIST_COMPLETE
        mcmd.velocities = [0.] * 18
        mcmd.torques = [0.] * 18
        if pos_t is not None:
            mcmd.positions = pos_t.tolist()
            mcmd.kp = [35., 25., 25., 35., 35., 25.,
                       35., 25., 25., 35., 35., 25.,
                       15., 15., 15.,
                       15., 15., 15.]
            mcmd.kd = [2., 1., 1., 2., 2., 1.,
                       2., 1., 1., 2., 2., 1.,
                       1., 1., 1.,
                       1., 1., 1., ]
        else:
            mcmd.positions = [0.] * 18
            mcmd.kp = [0.] * 18
            mcmd.kd = [0.] * 18
        return mcmd
    
def main(args=None):
    rclpy.init(args=args)

    node = master()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
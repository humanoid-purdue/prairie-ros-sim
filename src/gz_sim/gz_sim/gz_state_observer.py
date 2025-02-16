from math import sin, cos, pi
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Quaternion, Wrench
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Imu
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry
#from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from tf2_ros import TransformBroadcaster, TransformStamped

#from hrc_msgs.msg import StateVector
from gz_sim_interfaces.msg import StateObservationReduced
import numpy as np
#import time
#from trajectory_msgs.msg import JointTrajectory
#import os, sys
#from ament_index_python.packages import get_package_share_directory
from helpers import helpers

#JOINT_LIST = helpers.makeJointList()[0]
N_JOINTS = 12

class GZSateObserver(Node):

    def __init__(self):
        super().__init__('gz_state_estimator')

        qos_profile = QoSProfile(depth=10)

        self.sim_time = 0
        self.prev_time = 0
        self.prev_jvel = None
        self.prev_jacc = None
        self.orientation = None
        #self.prev_pos = None
        #self.efforts = None

        # joint states
        self.jpos_filt = helpers.SignalFilter(N_JOINTS, 1000, 20)
        self.jvel_filt = helpers.SignalFilter(N_JOINTS, 1000, 20)

        self.angvel_filt = helpers.SignalFilter(3, 1000, 10)  # calculated from IMU data (of pelvis, i.e. expected CoM)
        #self.vel_filt = helpers.SignalFilter(3, 1000, 20) # calculated from odometry data (displacement / dt)

        self.ang_vel = np.array([0., 0., 0.]).tolist()
        #self.odom_pos = [0., 0., 0.743]
        #self.odom_rot = np.array([0., 0., 0., 1.]).tolist()
        #self.left_force = np.zeros([3])
        #self.right_force = np.zeros([3])

        self.obs_pub = self.create_publisher(StateObservationReduced, 'state_observation', qos_profile)

        #self.sv_fwd = helpers.SVFwdKinematics()

        self.subscription_1 = self.create_subscription(
            JointState,
            '/joint_states_gz',
            self.joint_state_callback,
            10)
        self.subscription_3 = self.create_subscription(
            Imu,
            '/pelvis_imu',
            self.imu_callback,
            10)
        self.subscription_4 = self.create_subscription(
            Clock,
            '/sim_clock',
            self.clock_callback, 10
        )
        # Might be removed since in the real robot we don't have odometry
        #self.subscription_2 = self.create_subscription(
        #    Odometry,
        #    '/robot_odometry',
        #    self.odometry_callback,
        #    10)
        #self.subscription_5 = self.create_subscription(
        #    JointTrajectory,
        #    '/joint_trajectories',
        #    self.effort_callback, 10
        #)
        #self.subscription_6 = self.create_subscription(
        #    Wrench,
        #    '/left_foot_force',
        #    self.left_foot_callback, 10
        #)
        #self.subscription_7 = self.create_subscription(
        #    Wrench,
        #    '/right_foot_force',
        #    self.right_foot_callback, 10
        #)

    #def right_foot_callback(self, msg):
    #    force = msg.force
    #    force = np.array([force.x, force.y, force.z])
    #    self.right_force = force

    #def left_foot_callback(self, msg):
    #    force = msg.force
    #    force = np.array([force.x, force.y, force.z])
    #    self.left_force = force

    #def effort_callback(self, msg):
    #    point = msg.points[0]
    #    self.efforts = dict(zip(msg.joint_names, point.effort))

    def clock_callback(self, msg):
        secs = msg.clock.sec
        nsecs = msg.clock.nanosec
        self.sim_time = secs + nsecs * (10 ** -9)


    def imu_callback(self, msg):
        orien_quat_list = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.orientation = orien_quat_list
        self.ang_vel = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]

    #def odometry_callback(self, msg):
    #    pose = msg.pose.pose
    #    self.odom_pos = [pose.position.x, pose.position.y, pose.position.z]
    #    self.odom_rot = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]


    def joint_state_callback(self, joint_msg: JointState):
        """
            Every time receive a joint state message, 
            update the state vector and publish it for planner node to generate the next trajectory/action
        """
        sim_time = self.sim_time
        pub_obs_msg = StateObservationReduced()
        pub_obs_msg.joint_name = joint_msg.name
        dt = sim_time - self.prev_time

        #self.get_logger().info("L{} R{}".format(self.left_force, self.right_force))

        #sv.pos = self.odom_pos
        #sv.orien_quat = self.odom_rot

        if self.prev_jvel is None:
            self.prev_jvel = np.array(joint_msg.velocity)
        if self.prev_jacc is None:
            self.prev_jacc = np.zeros([len(joint_msg.name)])
        #if self.prev_pos is None:
        #    self.prev_pos = self.odom_pos

        if dt == 0:
            jacc = self.prev_jacc
            #vel = np.zeros([3])
        else:
            jacc = (np.array(joint_msg.velocity) - self.prev_jvel) / dt
            #vel = (np.array(self.odom_pos) - np.array(self.prev_pos)) / dt
            self.prev_jvel = np.array(joint_msg.velocity)
            self.prev_jacc = jacc
        pub_obs_msg.joint_acc = jacc.tolist()

        #self.vel_filt.update(vel)

        self.jpos_filt.update(np.array(joint_msg.position))
        self.jvel_filt.update(np.array(joint_msg.velocity))
        self.angvel_filt.update(np.array(self.ang_vel))
        if dt != 0 and self.sim_time > 0.1:
            pub_obs_msg.joint_pos = joint_msg.position
            pub_obs_msg.joint_vel = self.jvel_filt.get().tolist()
            pub_obs_msg.ang_vel = self.angvel_filt.get().tolist()
            #sv.vel = self.vel_filt.get().tolist()
        elif self.sim_time < 0.05:
            pub_obs_msg.joint_pos = joint_msg.position
            pub_obs_msg.joint_vel = np.zeros([len(joint_msg.velocity)]).tolist()
            pub_obs_msg.ang_vel = np.zeros([3]).tolist()
            #sv.vel = np.zeros([3]).tolist()
        else:
            pub_obs_msg.joint_pos = joint_msg.position
            pub_obs_msg.joint_vel = joint_msg.velocity
            pub_obs_msg.ang_vel = self.ang_vel
            #sv.vel = vel.tolist()


        # Calculate gravity vector in body (robot) frame
        if self.orientation is not None:
            rot_matrix = helpers.quaternion_rotation_matrix(self.orientation)
            gravity = np.array([0, 0, -9.81])
            self.grav_vec = rot_matrix.dot(gravity)
        else:
            self.grav_vec = np.array([0, 0, -9.81])
        pub_obs_msg.grav_vec = self.grav_vec.tolist()
            

        #sv = self.sv_fwd.update(sv)

        #new_efforts = np.zeros([len(joint_msg.name)])
        #if self.efforts is not None:
        #    for c in range(len(joint_msg.name)):
        #        jn = joint_msg.name[c]
        #        if jn == self.efforts.keys():
        #            new_efforts[c] = self.efforts[jn]
        #sv.efforts = new_efforts.tolist()
        pub_obs_msg.time = sim_time
        self.prev_time = sim_time
        #self.prev_pos = self.odom_pos
        self.obs_pub.publish(pub_obs_msg)

def main():
    rclpy.init(args=None)
    node = GZSateObserver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
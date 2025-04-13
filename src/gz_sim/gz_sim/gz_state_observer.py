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

#JOINT_LIST = helpers.makeJointList()[0]
N_JOINTS = 12
from helpers import helpers

from scipy.spatial.transform import Rotation as R

def update_velocity(acceleration, orientation_quat, dt, current_velocity):
    """
    Update the local linear velocity of an IMU in the sensor's frame.
    
    Parameters:
      acceleration (np.array): 3-element array representing the raw accelerometer
                               reading in the sensor frame (m/s²).
      orientation_quat (np.array): 4-element array for the sensor's orientation as a 
                                   quaternion [x, y, z, w]. This quaternion rotates vectors 
                                   from the sensor frame to the world frame.
      dt (float): Time interval between measurements (s).
      current_velocity (np.array): 3-element array representing the current velocity in 
                                   the sensor frame (m/s).
    
    Returns:
      np.array: Updated velocity in the sensor frame (m/s).
    """
    # Create a rotation object from the quaternion.
    r = R.from_quat(orientation_quat)
    
    # Define the gravitational acceleration in the world frame.
    # Here we assume a right-handed coordinate system with z pointing upward.
    gravity_world = np.array([0, 0, -9.81])
    
    # Rotate the gravity vector from world frame to sensor frame.
    # Since 'r' rotates from sensor to world, the inverse rotates from world to sensor.
    gravity_sensor = r.inv().apply(gravity_world)
    
    # Remove the gravitational component from the raw acceleration.
    net_accel = acceleration - gravity_sensor
    
    # Update the velocity by integrating the net acceleration over the time interval.
    new_velocity = current_velocity + net_accel * dt
    return new_velocity

def global_to_local_velocity(global_velocity, quaternion):
    """
    Transform a global velocity vector into a local velocity vector 
    using the inverse of the provided orientation quaternion.
    
    Parameters:
        global_velocity (array-like): The velocity vector in the global frame (3 elements).
        quaternion (array-like): The orientation quaternion in [w, x, y, z] format.
        
    Returns:
        np.ndarray: The velocity vector in the local frame.
    """
    
    # Create a Rotation object from the quaternion
    rotation = R.from_quat(quaternion)
    
    # To transform from global to local, apply the inverse rotation.
    local_velocity = rotation.inv().apply(global_velocity)
    
    return local_velocity

def quaternion_rotation_matrix(Q):
    """
    Convert a quaternion into a three-dimensional rotation matrix.

    :param Q: A 4-element list representing the quaternion in the form [q1, q2, q3, q0]
    :type Q: list or np.ndarray
    :return: A 3x3 rotation matrix corresponding to the quaternion
    :rtype: np.ndarray
    """

    q0 = Q[3]
    q1 = Q[0]
    q2 = Q[1]
    q3 = Q[2]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix


class GZSateObserver(Node):

    def __init__(self):
        super().__init__('gz_state_estimator')

        qos_profile = QoSProfile(depth=10)

        self.sim_time = 0
        self.prev_time = 0
        self.orientation = None
        #self.prev_pos = None
        #self.efforts = None

        self.odom_pos = [0., 0., 0.60833]
        self.odom_pos_prev = [0., 0., 0.60833]
        self.odom_rot = np.array([0., 0., 0., 1.])
        self.linvel = np.zeros([3])

        self.odom_prev_time = 0

        self.lin_acc = np.zeros([3])

        # joint states
        self.jpos_filt = helpers.SignalFilter(N_JOINTS, 1000, 20)
        self.jvel_filt = helpers.SignalFilter(N_JOINTS, 1000, 20)

        self.angvel_filt = helpers.SignalFilter(3, 1000, 10)  # calculated from IMU data (of pelvis, i.e. expected CoM)
        self.vel_filt = helpers.SignalFilter(3, 1000, 20)
        #self.vel_filt = helpers.SignalFilter(3, 1000, 20) # calculated from odometry data (displacement / dt)

        self.ang_vel = np.array([0., 0., 0.]).tolist()

        self.obs_pub = self.create_publisher(StateObservationReduced, 'state_observation', qos_profile)


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
            '/clock',
            self.clock_callback, 10
        )
        self.subscription_2 = self.create_subscription(
            Odometry,
            '/robot_odometry',
            self.odometry_callback,
            10
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

    def odometry_callback(self, msg):
        pose = msg.pose.pose
        self.odom_pos = [pose.position.x, pose.position.y, pose.position.z]
        self.odom_rot = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]


    def clock_callback(self, msg):
        secs = msg.clock.sec
        nsecs = msg.clock.nanosec
        self.sim_time = secs + nsecs * (10 ** -9)


    def imu_callback(self, msg):
        orien_quat_list = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.orientation = orien_quat_list
        self.ang_vel = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        self.lin_acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])


    def joint_state_callback(self, joint_msg: JointState):
        """
            Every time receive a joint state message, 
            update the state vector and publish it for planner node to generate the next trajectory/action
        """
        sim_time = self.sim_time
        pub_obs_msg = StateObservationReduced()
        pub_obs_msg.joint_name = joint_msg.name
        dt = sim_time - self.prev_time
        global_vel = None
        #if dt != 0:
        op = np.array(self.odom_pos)
        op_prev = np.array(self.odom_pos_prev)
        if (np.linalg.norm(op - op_prev) > 0.00001):
            if (sim_time - self.odom_prev_time) != 0:
                global_vel = (op - op_prev) / (sim_time - self.odom_prev_time)
                self.odom_prev_time = sim_time
                self.odom_pos_prev = self.odom_pos
        if self.orientation is not None:
            self.linvel = update_velocity(self.lin_acc, self.orientation, dt, self.linvel)
        if self.orientation is not None and global_vel is not None:
            self.linvel = global_to_local_velocity(global_vel, self.orientation)
        #self.vel_filt.update(self.linvel)
        self.jpos_filt.update(np.array(joint_msg.position))
        self.jvel_filt.update(np.array(joint_msg.velocity))
        self.angvel_filt.update(np.array(self.ang_vel))
        #self.linvel = self.vel_filt.get()
        if dt != 0 and self.sim_time > 0.1:
            pub_obs_msg.joint_pos = joint_msg.position
            pub_obs_msg.joint_vel = joint_msg.velocity #self.jvel_filt.get().tolist()
            pub_obs_msg.ang_vel = self.ang_vel #self.angvel_filt.get().tolist()
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
            rot_matrix = quaternion_rotation_matrix(self.orientation)
            gravity = np.array([0, 0, -1.0])
            self.grav_vec = np.linalg.inv(rot_matrix) @ gravity
            #inv_pelvis_rot = math.quat_inv(self.orientation)
            #self.grav_vec = math.rotate(np.array([0, 0, -1]), inv_pelvis_rot)
        else:
            self.grav_vec = np.array([0, 0, -1.0])
        pub_obs_msg.grav_vec = self.grav_vec.tolist()
        pub_obs_msg.lin_vel = self.linvel.tolist()
        pub_obs_msg.lin_acc = self.lin_acc.tolist()

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
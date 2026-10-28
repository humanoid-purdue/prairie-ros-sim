import os

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from gz_sim.gz_state_observer import quaternion_rotation_matrix
from prairie_control.control_modes import (
    DEFAULT_REAL_KD,
    DEFAULT_REAL_KP,
    HOME_POSE,
    JOINT_LIST_COMPLETE,
)
from rclpy.node import Node
from rclpy.time import Time
from trajectory_msgs.msg import JointTrajectory

from gz_sim_interfaces.msg import StateObservationReduced
from mujoco_sim.sim import MujocoSim

SCENE = os.path.join(get_package_share_directory("prairie_control"), "urdf", "nemo5.xml")
VIEWER_TIMESTEP = 1. / 60.


class SimNode(Node):
    """Run MuJoCo sim with subscribed JointTrajectory data and publish sensor data to construct state observation"""

    def __init__(self):
        super().__init__("mujoco_sim_node")

        self.joint_len = len(JOINT_LIST_COMPLETE)
        self.target_pos = np.zeros(self.joint_len)
        self.target_vel = np.zeros(self.joint_len)
        self.kP = np.array(DEFAULT_REAL_KP)
        self.kD = np.array(DEFAULT_REAL_KD)
        self.headless = False

        self.mujoco_sim = MujocoSim(SCENE, self.kP, self.kD, HOME_POSE, self.headless)

        self.create_subscription(JointTrajectory, "joint_trajectories", self._on_new_jt, 10)

        # self.joint_state_publisher = self.create_publisher(JointState, "joint_states_mj", 10)
        # self.pelvis_imu_publisher = self.create_publisher(Imu, "pelvis_imu", 10)
        # self.robot_odo_publisher = self.create_publisher(Odometry, "robot_odometry", 10)
        # self.clock_publisher = self.create_publisher(Clock, "clock", 10)

        self.obs_publisher = self.create_publisher(StateObservationReduced, "mj_state_observation", 10)

        self.create_timer(self.mujoco_sim.model.opt.timestep, self._apply_physics)
        if not self.headless:
            self.create_timer(VIEWER_TIMESTEP, self.mujoco_sim.update_viewer)

    def _on_new_jt(self, msg: JointTrajectory):
        """update desired targets for sim"""
        point = msg.points[0]
        self.target_pos = self._normalize(point.positions)
        self.target_vel = self._normalize(point.velocities)

    def _apply_physics(self):
        self.mujoco_sim.step_sim(self.target_pos, self.target_vel)

    def _publish_obs(self):
        msg = StateObservationReduced()
        msg.sim_time = Time(seconds=self.mujoco_sim.data.time).to_msg()
        msg.joint_name = JOINT_LIST_COMPLETE
        msg.joint_pos = self.normalize(self.mujoco_sim.data.qpos[7:])
        msg.joint_vel = self.normalize(self.mujoco_sim.data.qvel[6:])

        msg.ang_vel = self.normalize(self.mujoco_sim.get_pelvis_gyro())
        quat = self.mujoco_sim.get_quat()
        quat_inv = np.concat([-quat[:3], quat[3]])
        msg.grav_vec = quaternion_rotation_matrix(quat_inv) @ np.array([0., 0., -1.])
        msg.lin_vel = self.normalize(self.mujoco_sim.get_pelvis_local_linvel())
        msg.lin_acc = self.normalize(self.mujoco_sim.get_pelvis_local_linaccel())

        self.obs_publisher.publish(msg)
    # def _publish_joint_state(self):
    #     msg = JointState()
    #     msg.header.stamp = Time(seconds=self.mujoco_sim.data.time).to_msg()
    #     msg.header.frame_id = "?"

    #     msg.name = JOINT_LIST_COMPLETE
    #     msg.position = self.mujoco_sim.data.qpos[7:]
    #     msg.velocity = self.mujoco_sim.data.qvel[6:]
    #     msg.effort = self.mujoco_sim.data.ctrl[:]

    #     self.joint_state_publisher.publish(msg)

    # def _publish_pelvis_imu(self):
    #     stamp = Time(seconds=self.mujoco_sim.data.time).to_msg()
    #     gyro = self.mujoco_sim.get_pelvis_gyro()
    #     accel = self.mujoco_sim.get_pelvis_linaccel()

    #     imu = Imu()
    #     imu.header.stamp = stamp
    #     imu.header.frame_id = "imu" # check with Henry; not sure if this is right
    #     imu.angular_velocity = Vector3(x=gyro[0], y=gyro[1], z=gyro[2])
    #     imu.linear_acceleration = Vector3(x=accel[0], y=accel[1], z=accel[2])

    #     self.pelvis_imu_publisher.publish(imu)

    # def _publish_clock(self):
    #     msg = Clock()
    #     msg.clock = Time(seconds=self.mujoco_sim.data.time).to_msg()
    #     self.clock_publisher.publish(msg)

    # def _publish_robot_odometry(self):
    #     msg = Odometry()
    #     msg.header.stamp = self.get_clock().now().to_msg()
    #     msg.header.frame_id = "global" # check with Henry; not sure if this is right
    #     msg.pose = PoseWithCovariance()
    #     msg.pose.pose = Pose()
    #     msg.twist = TwistWithCovariance()
    #     msg.twist.twist = Twist()

    #     pos = self.mujoco_sim.get_imu_position()
    #     quat = self.muojco_sim.get_imu_orientation()
    #     linvel = self.mujoco_sim.get_imu_linvel()
    #     angvel = self.mujoco_sim.get_imu_angvel()

    #     msg.pose.pose.position = Point(x=pos[0], y=pos[1], z=pos[2])
    #     msg.pose.pose.orientation = Quaternion(x=quat[1], y=quat[2], z=quat[3], w=quat[0])
    #     msg.twist.twist.linear = Vector3(x=linvel[0], y=linvel[1], z=linvel[2])
    #     msg.twist.twist.angular = Vector3(x=angvel[0], y=angvel[1], z=angvel[2])

    #     self.robot_odo_publisher.publish(msg)

 
    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        arr_len = len(arr)
        if arr_len == self.joint_len:
            return arr
        if arr_len < self.joint_len:
            return np.concat([arr, np.zeros(self.joint_len - arr_len)])
        return arr[: self.joint_len]

def main():
    rclpy.init()
    node = SimNode()
    rclpy.spin(node)
    rclpy.shutdown()
import os

import numpy as np
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, JointState
from trajectory_msgs.msg import JointTrajectory

from mujoco_sim.mujoco_sim import MujocoSim
from prairie_control.prairie_control.control_modes import JOINT_LIST_COMPLETE

SCENE = os.path.join(get_package_share_directory("prairie_control"), "urdf", "nemo5.xml")
PHYSICS_TIMESTEP = 0.01


class SimNode(Node):
    """Run MuJoCo sim with subscribed JointTrajectory data and publish sensor data to construct state observation"""

    def __init__(self):
        super().__init__("mujoco_sim")

        self.joint_len = len(JOINT_LIST_COMPLETE)
        self.target_pos = np.zeros(self.joint_len)
        self.target_vel = np.zeros(self.joint_len)
        self.kP = 0
        self.kD = 0

        self.mujoco_sim = MujocoSim(SCENE, self.kP, self.kD)

        self.create_subscription(JointTrajectory, "joint_trajectories", self._on_new_jt, 10)

        self.joint_state_publisher = self.create_publisher(JointState, "joint_states_mj", 10)
        self.pelvis_imu_publisher = self.create_publisher(Imu, "pelvis_imu", 10)
        self.robot_odo_publisher = self.create_publisher(Odometry, "robot_odometry", 10)
        self.clock_publisher = self.create_publisher(Clock, "clock", 10)

        self.create_timer(PHYSICS_TIMESTEP, self._apply_physics)

    def _on_new_jt(self, msg: JointTrajectory):
        """update desired targets for sim"""
        point = msg.points[0]
        self.target_pos = self._normalize(point.positions)
        self.target_vel = self._normalize(point.velocities)

    def _apply_physics(self):
        self.mujoco_sim.step_sim(self.target_pos, self.target_vel)

    def _publish_pelvis_imu(self):
        stamp = self.get_clock().now().to_msg()
        gyro = self.mujoco_sim.get_pelvis_gyro()
        accel = self.mujoco_sim.get_pelvis_linaccel()

        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = "imu"
        imu.angular_velocity = Vector3(gyro[0], gyro[1], gyro[2])
        imu.linear_acceleration = Vector3(accel[0], accel[1], accel[2])

        self.pelvis_imu_publisher.publish(imu)

    def _publish_clock(self):
        msg = Clock()
        msg.clock = Time(seconds=self.mujoco_sim.data.time).to_msg()
        self.clock_publisher.publish(msg)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        arr_len = len(arr)
        if arr_len == self.joint_len:
            return arr
        if arr_len < self.joint_len:
            return np.concat([arr, np.zeros(self.joint_len - arr_len)])
        return arr[: self.joint_len]

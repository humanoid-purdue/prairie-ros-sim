"""Multiplexer that subscribes to JointTrajectory topics for each MODE and
publishes one for sim and one for real depending on current state mode"""

import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from gz_sim_interfaces.msg import MotorCmd, PrairieState

from .control_modes import (
    HOME_POSE,
    JOINT_LIST_COMPLETE,
    MODE_DISABLED,
    MODE_HOME,
    MODE_MIRROR,
    MODE_STAND,
    MODE_WALK,
    WALK_REAL_KP,
    motor_payload_from_trajectory,
)


class PrairieCommandMux(Node):
    """
    Multiplexer that subscribes to JointTrajectory topics for each MODE and
    publishes one for sim and one for real depending on current state mode
    """

    def __init__(self):
        """
        Instantiates state, joint position, home metadata,
        gazebo and real joint trajectories and publishers, and state & mode trajectory subscribers
        """
        super().__init__("prairie_command_mux")
        self.state = PrairieState()
        self.state.sim_mode = MODE_STAND
        self.state.real_mode = MODE_DISABLED
        self.previous_real_mode = MODE_DISABLED
        self.joint_pos = np.zeros(len(JOINT_LIST_COMPLETE))
        self.home_start_pos = self.joint_pos.copy()
        self.home_start_time = time.time()

        self.gz_stand_trajectory = None
        self.gz_policy_trajectory = None
        self.gz_mirror_trajectory = None
        self.real_standing_trajectory = None
        self.real_policy_trajectory = None
        self.last_warn_times = {}

        self.gz_pub = self.create_publisher(JointTrajectory, "/joint_trajectories", 10)
        self.real_pub = self.create_publisher(MotorCmd, "/real_joint_trajectories", 10)

        self.create_subscription(
            PrairieState, "/prairie/state", self.state_callback, 10
        )
        self.create_subscription(
            JointTrajectory,
            "/gz_standing_jtp",
            self.gz_standing_callback,
            10,
        )
        self.create_subscription(
            JointTrajectory,
            "/gz_policy_jtp",
            self.gz_policy_callback,
            10,
        )
        self.create_subscription(
            JointTrajectory,
            "/gz_mirror_jtp",
            self.gz_mirror_callback,
            10,
        )
        self.create_subscription(
            JointTrajectory,
            "/real_standing_jtp",
            self.real_standing_callback,
            10,
        )
        self.create_subscription(
            JointTrajectory,
            "/real_policy_jtp",
            self.real_policy_callback,
            10,
        )
        self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, 10
        )

        self.timer = self.create_timer(0.002, self.timer_callback)

    def state_callback(self, msg: PrairieState):
        """
        Updates prairie state and previous_real)mode when given new PrairieState,
        and starts tracking time if the CommandMux enters HOME_MODE

        :param msg: ROS 2 message holding sim/real mode data and movement command
        """
        if self.previous_real_mode != MODE_HOME and msg.real_mode == MODE_HOME:
            self.home_start_pos = self.joint_pos.copy()
            self.home_start_time = time.time()
        self.previous_real_mode = msg.real_mode
        self.state = msg

    def gz_standing_callback(self, msg: JointTrajectory):
        """
        Mutator for self.gz_stand_trajectory

        :param msg: JointTrajectory to set gz_stand_trajectory to
        """
        self.gz_stand_trajectory = msg

    def gz_policy_callback(self, msg: JointTrajectory):
        """
        Mutator for self.gz_policy_trajectory

        :param msg: JointTrajectory to set gz_policy_trajectory to
        """
        self.gz_policy_trajectory = msg

    def gz_mirror_callback(self, msg: JointTrajectory):
        """
        Mutator for self.gz_mirror_trajectory

        :param msg: JointTrajectory to set gz_mirror_trajectory to
        """
        self.gz_mirror_trajectory = msg

    def real_standing_callback(self, msg: JointTrajectory):
        """
        Mutator for self.real_standing_trajectory

        :param msg: JointTrajectory to set real_standing_trajectory to
        """
        self.real_standing_trajectory = msg

    def real_policy_callback(self, msg: JointTrajectory):
        """
        Mutator for self.real_policy_trajectory

        :param msg: JointTrajectory to set real_policy_trajectory to
        """
        self.real_policy_trajectory = msg

    def joint_state_callback(self, msg: JointState):
        """
        Update self.joint_pos with normalized JointState positions

        :param msg: JointState containing joint names, positions, velocities, and efforts
        """
        limit = min(len(msg.position), len(JOINT_LIST_COMPLETE))
        for i in range(limit):
            self.joint_pos[i] = msg.position[i]

    def timer_callback(self):
        """
        The CommandMux's update function
        """
        sim_traj = self.select_sim_trajectory()
        if sim_traj is not None:
            self.gz_pub.publish(sim_traj)
        self.real_pub.publish(self.select_real_command())

    def select_sim_trajectory(self) -> JointTrajectory | None:
        """
        Sim trajectory multiplexer
        Tries to output JointTrajectory associated with current sim mode
        If the JointTrajectory data is invalid, send warning and return None

        :return: selected JointTrajectory
        """
        if self.state.sim_mode == MODE_WALK:
            if self.gz_policy_trajectory is not None:
                return self.gz_policy_trajectory
            self.warn_throttled(
                "gz_policy_missing",
                "missing /gz_policy_jtp; falling back to standing",
            )
        elif self.state.sim_mode == MODE_MIRROR:
            if self.gz_mirror_trajectory is not None:
                return self.gz_mirror_trajectory
            self.warn_throttled(
                "gz_mirror_missing",
                "missing /gz_mirror_jtp; falling back to standing",
            )

        if self.gz_stand_trajectory is not None:
            return self.gz_stand_trajectory
        self.warn_throttled(
            "gz_stand_missing",
            "missing /gz_standing_jtp; no Gazebo command published",
        )
        return None

    def select_real_command(self) -> MotorCmd:
        """
        Select a MotorCmd for the current real mode.
        Warn when a required trajectory is missing and return a disabled command.
        Disabled or unsupported modes also return a disabled command.

        :return: selected MotorCmd
        """
        if self.state.real_mode == MODE_HOME:
            return self.trajectory_to_motor_command(self.home_trajectory())
        if self.state.real_mode == MODE_STAND:
            if self.real_standing_trajectory is not None:
                return self.trajectory_to_motor_command(
                    self.real_standing_trajectory,
                    kp=[0.0] * len(JOINT_LIST_COMPLETE),
                )
            self.warn_throttled(
                "real_stand_missing",
                "missing /real_standing_jtp; disabling real command",
            )
            return self.disabled_mcmd()
        if self.state.real_mode == MODE_MIRROR:
            if self.gz_mirror_trajectory is not None:
                return self.trajectory_to_motor_command(self.gz_mirror_trajectory)
            self.warn_throttled(
                "real_mirror_missing",
                "missing /gz_mirror_jtp; disabling real command",
            )
            return self.disabled_mcmd()
        if self.state.real_mode == MODE_WALK:
            if self.real_policy_trajectory is not None:
                return self.trajectory_to_motor_command(self.real_policy_trajectory, kp=WALK_REAL_KP)
            self.warn_throttled(
                "real_policy_missing",
                "missing /real_policy_jtp; disabling real command",
            )
            return self.disabled_mcmd()
        return self.disabled_mcmd()

    def home_trajectory(self) -> JointTrajectory:
        """
        Interpolates target JointTrajectory between ending home and starting home position based on time elapsed since entering home

        :return: JointTrajectory representing position targets of each actuator
        """
        elapsed = time.time() - self.home_start_time
        coeff = min(elapsed / 2.0, 1.0)
        positions = np.array(HOME_POSE) * coeff + (1.0 - coeff) * self.home_start_pos
        return self.pos_to_trajectory(positions)

    def pos_to_trajectory(self, positions) -> JointTrajectory:
        """
        Create a constant target with zero velocity and effort.

        Identical targets at 0 and 100 seconds describe a hold trajectory.
        Hardware conversion uses only the first point and ignores this timing.

        :param positions: array of joint positions
        :return: JointTrajectory representing trajectory to hold mechanism in place at set position
        """
        joint_traj = JointTrajectory()
        joint_traj.header.stamp = self.get_clock().now().to_msg()
        joint_traj.joint_names = list(JOINT_LIST_COMPLETE)

        point = JointTrajectoryPoint()
        point.time_from_start = Duration(sec=0, nanosec=0)
        point.positions = positions.tolist()
        point.velocities = [0.0] * len(JOINT_LIST_COMPLETE)
        point.effort = [0.0] * len(JOINT_LIST_COMPLETE)

        hold_point = JointTrajectoryPoint()
        hold_point.time_from_start = Duration(sec=100, nanosec=0)
        hold_point.positions = list(point.positions)
        hold_point.velocities = list(point.velocities)
        hold_point.effort = list(point.effort)
        joint_traj.points = [point, hold_point]
        return joint_traj

    def disabled_mcmd(self) -> MotorCmd:
        """
        Return static motor command
        :return: motor command with no motion
        """
        return self.trajectory_to_motor_command(
            self.pos_to_trajectory(np.zeros(len(JOINT_LIST_COMPLETE))),
            disable=True,
        )

    def trajectory_to_motor_command(
        self, joint_traj: JointTrajectory, disable=False, kp=None
    ) -> MotorCmd:
        """
        Convert the first trajectory point into an instantaneous MotorCmd.

        Current publishers stream a target at time zero followed by an identical
        hold point at 100 seconds. MotorCmd has no trajectory timing, so later
        points and time_from_start are ignored. This does not execute arbitrary
        multipoint trajectories; new targets arrive through subsequent messages.

        :param joint_traj: JointTrajectory containing the current target
        :param disable: whether to disable motor actuation
        :param kp: optional per-joint proportional gains
        :return: MotorCmd for the first point
        """
        point = joint_traj.points[0] if joint_traj.points else JointTrajectoryPoint()
        payload = motor_payload_from_trajectory(
            joint_traj.joint_names,
            point.positions,
            point.velocities,
            point.effort,
            disable=disable,
            kp=kp,
        )

        msg = MotorCmd()
        msg.joint_names = payload["joint_names"]
        msg.positions = payload["positions"]
        msg.velocities = payload["velocities"]
        msg.torques = payload["torques"]
        msg.kp = payload["kp"]
        msg.kd = payload["kd"]
        return msg

    def warn_throttled(self, key: str, message: str):
        """
        Logs warnings at a max rate of 2 seconds per warning for every unique key

        :param key: string identifying the type of issue
        :param message: warning message to try and log
        """
        now = time.time()
        if now - self.last_warn_times.get(key, 0.0) > 2.0:
            self.get_logger().warn(message)
            self.last_warn_times[key] = now


def main(args=None):
    """
    Starts PrairieCommandMux node's event loop
    """
    rclpy.init(args=args)
    node = PrairieCommandMux()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

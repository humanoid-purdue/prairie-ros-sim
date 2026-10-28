import os
import sys
import time

import mujoco
import numpy as np
from ament_index_python.packages import get_package_share_directory

# helpers/ is installed as data files, not as part of the prairie_control package
sys.path.append(os.path.join(get_package_share_directory("prairie_control"), "helpers"))
from stabilizer import Stabilizer

from prairie_control.gz_standing import STANDING_COM


class MujocoSim:
    def __init__(self, scene: str, kP, kD, home_pose, headless=True):
        self.model = mujoco.MjModel.from_xml_path(scene)
        self.data = mujoco.MjData(self.model)
        self.reset_to_pose(home_pose)
        self.kP = kP
        self.kD = kD
        self.stabilizer = Stabilizer()

        self.viewer = None
        if not headless:
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.model, self.data)

    def reset_to_pose(self, pose):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[7:] = pose
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
    
    def step_sim(self, tp: np.ndarray, target_vel: np.ndarray):
        self.stabilizer.update_simulation(self.data.qpos[7:], self.data.qvel[6:])
        ff_torque = self.stabilizer.calculate_joint_torques(STANDING_COM)
        target_pos = tp
        cur_pos = self.data.qpos[7:]
        cur_vel = self.data.qvel[6:]
        torques = (target_pos - cur_pos) * self.kP + (target_vel - cur_vel) * self.kD + ff_torque
        self.data.ctrl[:] = ff_torque
        mujoco.mj_step(self.model, self.data)

    def get_quat(self) -> np.ndarray:
        """x, y, z, w"""
        return np.concat([self.data.qpos[4:7], self.data.qpos[3]])
    def get_joint_pos(self) -> np.ndarray:
        return self.data.qpos[7:]
    def get_joint_vel(self) -> np.ndarray:
        return self.data.qvel[6:]
    def get_joint_torque(self) -> np.ndarray:
        return self.data.ctrl[:]

    # returns imu data in local reference frame
    def get_pelvis_gyro(self) -> np.ndarray:
        return self.data.sensor("gyro").data
    def get_pelvis_local_linvel(self) -> np.ndarray:
        return self.data.sensor("local_linvel").data
    def get_pelvis_local_linaccel(self) -> np.ndarray:
        return self.data.sensor("accelerometer").data

    # returns imu data in global reference frame
    def get_imu_orientation(self) -> np.ndarray:
        """Returns in wxyz format"""
        return self.data.sensor("orientation").data
    def get_imu_position(self) -> np.ndarray:
        return self.data.sensor("position").data
    def get_imu_linvel(self) -> np.ndarray:
        return self.data.sensor("global_linvel").data
    def get_imu_angvel(self) -> np.ndarray:
        return self.data.sensor("global_angvel").data

    def update_viewer(self):
        if self.viewer is not None:
            if not self.viewer.is_running():
                raise SystemExit(0)
            self.viewer.sync()


if __name__ == "__main__":
    ms = MujocoSim("src/prairie_control/urdf/nemo5.xml", 0, 0, False)
    while True:
        ms.update_viewer()
        time.sleep(1./60.)
import mujoco
import numpy as np


class MujocoSim:

    def __init__(self, scene: str, kP, kD):
        self.model = mujoco.MjModel.from_xml_path(scene)
        self.data = mujoco.MjData(self.model)
        self.kP = kP
        self.kD = kD

    def step_sim(self, target_pos: np.ndarray, target_vel: np.ndarray):
        cur_pos = self.data.qpos[7:]
        cur_vel = self.data.qvel[6:]
        torques = (target_pos - cur_pos) * self.kP + (target_vel - cur_vel) * self.kD
        mujoco.mj_step(self.model, self.data)
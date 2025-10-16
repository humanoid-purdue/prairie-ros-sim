import numpy as np

class SimpleAnkleStabilizer:

    def __init__(self):
        self.kp = 2
        self.kd = 1

        self.m = np.sum(model.body_mass)
        self.g = 9.8
        self.com = data.subtree_com[0]
        self.prev_com = self.com.copy()
        self.robot_center = [0, 0, 0]
        self.pz_req = [0, 0]

        self.l_pitch_id = 4
        self.r_pitch_id = 5
        self.l_roll_id = 10
        self.r_roll_id = 11

    @staticmethod
    def lin_interp(t, t_total, start, end):
        return start + (t / t_total) * (end - start)

    def apply_pitch_torque(self, pz_req):
        torque = 0.5 * self.m * self.g * (pz_req - self.data.xpos[self.l_pitch_body][0])
        self.data.ctrl[self.l_pitch_act] = self.kp * torque - self.kd * self.data.qvel[self.l_pitch_joint]
        self.data.ctrl[self.r_pitch_act] = self.kp * torque - self.kd * self.data.qvel[self.r_pitch_joint]

    def apply_roll_torque(self, pz_req):
        torque = -0.5 * self.m * self.g * (pz_req - self.robot_center[1])
        self.data.ctrl[self.l_roll_act] = self.kp * torque
        self.data.ctrl[self.r_roll_act] = self.kp * torque

    def step(self, dt, desired_com, robot_center):
        self.robot_center = robot_center
        com_vel = (self.com - self.prev_com) / dt
        self.prev_com = self.com.copy()
        omega2 = self.g / self.com[2]
        desired_accel_x = (-50 * (self.com[0] - desired_com[0]) - 5 * com_vel[0]) / self.m
        desired_accel_y = (-50 * (self.com[1] - desired_com[1]) - 5 * com_vel[1]) / self.m
        self.pz_req = [self.com[0] - desired_accel_x / omega2, self.com[1] - desired_accel_y / omega2]
        self.apply_pitch_torque(self.pz_req[0])
        self.apply_roll_torque(self.pz_req[1])

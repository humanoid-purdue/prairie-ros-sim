import sys
print(sys.executable)
try:
    import numpy as np
    import scipy
    from scipy import sparse
except:
    print("Unable to load PY dependencies")

try:
    from geometry_msgs.msg import Point, Pose, Quaternion
    from hrc_msgs.msg import InverseCommand, BipedalCommand
except:
    print("Unable to load ROS dependencies")
import os
import yaml


def makeJointList():
    try:
        from ament_index_python.packages import get_package_share_directory
        joint_path = os.path.join(
                    get_package_share_directory('prairie_control'),
                    "config/joints_list.yaml")
    except:
        joint_path = os.getcwd()[:-7] + "config/joints_list.yaml"
    with open(joint_path, 'r') as infp:
        pid_txt = infp.read()
        joints_dict = yaml.load(pid_txt, Loader = yaml.Loader)
    JOINT_LIST_COMPLETE = []
    JOINT_LIST_MOVABLE = []
    JOINT_LIST_LEG = []
    for c in range(len(joints_dict.keys())):
        JOINT_LIST_COMPLETE += [joints_dict[c]['name']]
        if joints_dict[c]['movable']:
            JOINT_LIST_MOVABLE += [joints_dict[c]['name']]
        if joints_dict[c]['leg']:
            JOINT_LIST_LEG += [joints_dict[c]['name']]
    return JOINT_LIST_COMPLETE, JOINT_LIST_MOVABLE, JOINT_LIST_LEG



def quaternion_rotation_matrix(Q):
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

# r02 r12 r22
# 2 * ( q0^2 + q3^2 ) - 1

class JointInterpolation:
    def __init__(self, joint_num, position_error, velocity_error):
        self.pos_err = position_error
        self.vel_err = velocity_error
        self.joint_num = joint_num

        self.pos_arr = None
        self.vel_arr = None
        self.timelist = None
        self.cs_pos = None
        self.cs_vel = None
        self.cs_tau = None
        self.cs_centroid = None


        self.consecutive_fails = 0
        self.fail_thresh = 5

    def updateJointState(self, timelist, pos, vel, centroid_vec = None):
        #Of shape (seq len, joint num)
        #make existing sequency with cs and interpolate and filter
        if self.cs_pos is not None:
            match_pos = self.cs_pos(timelist)
            match_vel = self.cs_vel(timelist)
            new_pos = pos * 0.5 + match_pos * 0.5
            new_vel = vel * 0.5 + match_vel * 0.5
        else:
            new_pos = pos
            new_vel = vel
        cs_pos = scipy.interpolate.CubicSpline(timelist, new_pos, axis = 0)
        cs_vel = scipy.interpolate.CubicSpline(timelist, new_vel, axis = 0)
        if centroid_vec is not None:
            self.cs_centroid = scipy.interpolate.CubicSpline(timelist, centroid_vec, axis = 0)

        if self.pos_arr is None:
            self.pos_arr = pos
            self.vel_arr = vel
            self.timelist = timelist
            self.cs_pos = cs_pos
            self.cs_vel = cs_vel
            return True, 0, 0
        check_pos = cs_pos(timelist[1])
        check_vel = cs_vel(timelist[1])
        inrange = self.checkDelta(timelist[1], check_pos, check_vel)
        if inrange:
            self.pos_arr = pos
            self.vel_arr = vel
            self.timelist = timelist
            self.cs_pos = cs_pos
            self.cs_vel = cs_vel
            return True, np.mean(np.abs(pos - check_pos)), np.mean(np.abs(vel - check_vel))
        else:
            self.consecutive_fails += 1
            if self.consecutive_fails > self.fail_thresh:
                self.consecutive_fails = 0
                self.pos_arr = pos
                self.vel_arr = vel
                self.timelist = timelist
                self.cs_pos = cs_pos
                self.cs_vel = cs_vel
            return False, np.mean(np.abs(pos - check_pos)), np.mean(np.abs(vel - check_vel))

    def forceUpdateState(self, timelist, pos, vel, tau):
        if self.cs_pos is not None and self.cs_pos is not None:
            match_pos = self.cs_pos(timelist)
            match_vel = self.cs_vel(timelist)
            match_tau = self.cs_tau(timelist)

            sz = match_pos.shape[0]
            weight_vec = 0.0 + 2 * (( np.arange(sz) - 1 ) / sz )
            weight_vec[weight_vec > 1] = 1
            weight_vec[weight_vec < 0] = 0
            weight_vec = weight_vec[:, None]

            new_pos = pos * weight_vec + match_pos * (1 - weight_vec)
            new_vel = vel * weight_vec + match_vel * (1 - weight_vec)
            new_tau = tau * weight_vec + match_tau * (1 - weight_vec)
        else:
            new_pos = pos
            new_vel = vel
            new_tau = tau
        self.cs_pos = scipy.interpolate.CubicSpline(timelist, new_pos, axis=0)
        self.cs_vel = scipy.interpolate.CubicSpline(timelist, new_vel, axis=0)
        self.cs_tau = scipy.interpolate.CubicSpline(timelist, new_tau, axis=0)

    def updateMixState(self, current_time, timelist, pos, vel, tau):
        self.cs_tau = scipy.interpolate.CubicSpline(timelist, tau, axis=0)
        if self.cs_pos is None or self.cs_vel is None:
            self.cs_pos = scipy.interpolate.CubicSpline(timelist, pos, axis=0)
            self.cs_vel = scipy.interpolate.CubicSpline(timelist, vel, axis=0)
        else:
            new_timelist = np.concatenate([np.array([current_time]), timelist[:]], axis = 0)
            new_timelist = np.sort(np.array(list(set(list(new_timelist)))))
            new_timelist = new_timelist[np.where(new_timelist == current_time)[0][0] : ]

            new_pos = scipy.interpolate.CubicSpline(timelist, pos, axis=0)(new_timelist)
            new_vel = scipy.interpolate.CubicSpline(timelist, vel, axis=0)(new_timelist)

            match_pos = self.cs_pos(new_timelist)
            match_vel = self.cs_vel(new_timelist)
            sz = match_pos.shape[0]
            weight_vec = 0.0 + 2 * ((np.arange(sz) - 1) / sz)
            weight_vec[weight_vec > 1] = 1
            weight_vec[weight_vec < 0] = 0
            weight_vec = weight_vec[:, None]

            new_pos = new_pos * weight_vec + match_pos * (1 - weight_vec)
            new_vel = new_vel * weight_vec + match_vel * (1 - weight_vec)

            self.cs_pos = scipy.interpolate.CubicSpline(timelist, pos, axis=0)
            self.cs_vel = scipy.interpolate.CubicSpline(timelist, vel, axis=0)

    def updateX(self, timelist, x):
        centroid_pos = x[:, 0:7]
        pos = x[:, 7:7 + self.joint_num]
        centroid_vel = x[:, 7 + self.joint_num: 13 + self.joint_num]
        vel = x[:, 13 + self.joint_num:]
        centroid = np.concatenate([centroid_pos, centroid_vel], axis = 1)
        return self.updateJointState(timelist, pos, vel, centroid_vec = centroid)


    def checkDelta(self, check_time, pos, vel):
        check_pos = self.cs_pos(check_time)
        check_vel = self.cs_vel(check_time)
        pc = np.sum(np.abs(pos - check_pos) < self.pos_err)
        vc = np.sum(np.abs(vel - check_vel) < self.vel_err)
        return pc == 0 and vc == 0

    def getInterpolation(self, timestamp, pos_delta = 0):
        pos = self.cs_pos(timestamp + pos_delta)
        vel = self.cs_vel(timestamp)
        if self.cs_tau is None:
            return pos, vel
        else:
            tau = self.cs_tau(timestamp)
            return pos, vel, tau

    def getX(self, t):
        pos, vel = self.getInterpolation(t)
        centroid = self.cs_centroid(t)
        centroid[3:7] = centroid[3:7] / (np.sum(centroid[3:7] ** 2) ** 0.5)
        x0 = np.concatenate([centroid[:7], pos, centroid[7:], vel], axis=0)
        return x0

    def getSeedX(self, timestamps):
        x = []
        for t in timestamps:
            x0 = self.getX(t)
            x += [x0]
        return x

    def hasHistory(self):
        return not(self.cs_pos is None or self.cs_vel is None)


class JointSpaceFilter:
    def __init__(self, joint_num, position_error, velocity_error):

        self.timelist = None
        self.cs_pos = None
        self.cs_vel = None
        self.cs_tau = None

        self.joint_num = joint_num

        self.max_points = 100

        self.state_samples = np.zeros([0, joint_num * 2])
        self.tau_samples = np.zeros([0, joint_num])

        self.A = np.zeros([joint_num, joint_num * 2])
        self.B = np.zeros([joint_num])

    def getInterpolation(self, pos, vel, timestamp):
        state_r = np.concatenate([pos, vel], axis = 0)
        error = np.linalg.norm(self.state_samples[:, :self.joint_num] - state_r[:self.joint_num], axis = 1)
        index = np.argmin(error)
        ref = self.tau_samples[index, :]
        weight = 0.5 + 0.0 * np.arange(timestamp.shape[0]) / timestamp.shape[0]
        weight[weight > 1] = 1

        pos = self.cs_pos(timestamp)
        vel = self.cs_vel(timestamp)
        tau = self.cs_tau(timestamp) # * weight[:, None] + np.tile(ref[None, :], [timestamp.shape[0], 1]) * (1 - weight)[:, None]


        return pos, vel, tau

    def hasHistory(self):
        return not(self.cs_pos is None or self.cs_vel is None or self.cs_tau is None)



    def updateMixState(self, current_time, timelist, pos, vel, tau):
        state = np.concatenate([np.array(pos[1:, :]), np.array(vel[1:, :])], 1)
        self.state_samples = np.concatenate([state, self.state_samples], axis = 0)
        self.tau_samples = np.concatenate([tau[1:, :], self.tau_samples], axis = 0)
        if self.state_samples.shape[0] > self.max_points:
            self.state_samples = self.state_samples[:self.max_points, :]
            self.tau_samples = self.tau_samples[:self.max_points, :]
        self.cs_tau = scipy.interpolate.CubicSpline(timelist[:], tau[:, :], axis = 0)

        if self.cs_pos is None or self.cs_vel is None:
            self.cs_pos = scipy.interpolate.CubicSpline(timelist, pos, axis=0)
            self.cs_vel = scipy.interpolate.CubicSpline(timelist, vel, axis=0)
        else:
            new_timelist = np.concatenate([np.array([current_time]), timelist[:]], axis = 0)
            new_timelist = np.sort(np.array(list(set(list(new_timelist)))))
            new_timelist = new_timelist[np.where(new_timelist == current_time)[0][0] : ]

            new_pos = scipy.interpolate.CubicSpline(timelist, pos, axis=0)(new_timelist)
            new_vel = scipy.interpolate.CubicSpline(timelist, vel, axis=0)(new_timelist)

            match_pos = self.cs_pos(new_timelist)
            match_vel = self.cs_vel(new_timelist)
            sz = match_pos.shape[0]
            weight_vec = 0.0 + 2 * ((np.arange(sz) - 1) / sz)
            weight_vec[weight_vec > 1] = 1
            weight_vec[weight_vec < 0] = 0
            weight_vec = weight_vec[:, None]

            new_pos = new_pos * weight_vec + match_pos * (1 - weight_vec)
            new_vel = new_vel * weight_vec + match_vel * (1 - weight_vec)

            self.cs_pos = scipy.interpolate.CubicSpline(new_timelist, new_pos, axis=0)
            self.cs_vel = scipy.interpolate.CubicSpline(new_timelist, new_vel, axis=0)

class SignalFilter:
    def __init__(self, params, freq, cutoff):
        #self.b, self.a = scipy.signal.butter(4, cutoff, btype='low', analog=False, fs = freq)
        nyquist = 0.5 * freq
        normal_cutoff = cutoff / nyquist
        self.sos = scipy.signal.butter(4, normal_cutoff, btype='low', analog=False, output='sos')
        self.zi = []
        self.y = np.zeros(params)
        for c in range(params):
            self.zi += [scipy.signal.sosfilt_zi(self.sos)]


    def update(self, vec):
        for c in range(vec.shape[0]):
            filtered_point, self.zi[c] = scipy.signal.sosfilt(self.sos, vec[c:c+1], zi=self.zi[c], axis = 0)
            self.y[c] = filtered_point[0]
    def get(self):
        return self.y

class CSVDump:
    def __init__(self, params, name_list):
        self.abs_path = "/home/aurum/RosProjects/prairie/datadump"
        self.name_list = name_list
        self.max_num = 5000
        self.arr = np.zeros([self.max_num, params , len(name_list)])

    def update(self, vecs):
        self.arr = np.roll(self.arr, -1, axis = 0)
        for c in range(len(vecs)):
            self.arr[-1, :, c] = vecs[c]

    def save(self):
        if os.path.exists(self.abs_path):
            for c in range(len(self.name_list)):
                name = self.name_list[c]
                path = self.abs_path + "/{}.csv".format(name)
                np.savetxt(path, self.arr[:, :, c], delimiter = ',')

class discreteIntegral:
    def __init__(self, params):
        self.integral = np.zeros([params])
        self.prev_time = -1
    def update(self, timestamp, x):
        if self.prev_time == -1:
            self.prev_time = timestamp
        else:
            self.integral += x * (timestamp - self.prev_time)
        return self.integral


def makeFwdTraj(current_state, target):
    delta = target - current_state
    delta_range = (np.arange(100) + 1) * 0.001
    if delta > 0:
        delta_list = delta - delta_range
        delta_list[delta_list < 0] = 0
    else:
        delta_list = delta + delta_range
        delta_list[delta_list > 0] = 0
    new_pos = delta_list + current_state
    new_vel = ( new_pos[1:] - new_pos[:-1] ) / 0.001
    new_vel = np.concatenate([new_vel, np.array([0])], axis = 0)
    return new_pos, new_vel

def makePose(pos, rot):
    pose = Pose()
    point = Point()
    point.x = float(pos[0])
    point.y = float(pos[1])
    point.z = float(pos[2])
    orien = Quaternion()
    orien.x = float(rot[0])
    orien.y = float(rot[1])
    orien.z = float(rot[2])
    orien.w = float(rot[3])
    pose.position = point
    pose.orientation = orien

    return pose

def np2Point(arr):
    a = Point()
    a.x = arr[0]
    a.y = arr[1]
    a.z = arr[2]
    return a


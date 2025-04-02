import sys

print(sys.executable)
try:
    import numpy as np
    import scipy
    import scipy.signal
    # import crocoddyl
    import pinocchio as pin
    import osqp
    from scipy import sparse
except:
    print("Unable to load PY dependencies")

try:
    from geometry_msgs.msg import Point, Pose, Quaternion
    # from hrc_msgs.msg import InverseCommand, BipedalCommand
except:
    print("Unable to load ROS dependencies")
import os
import yaml


def makeJointList():
    """
    Loads a list of robot joints from a given YAML file. The joints are then categorized into a complete joint list,
    movable joints, and leg joints.

    :return: A tuple of three lists containing names of every joint, movable joints, and leg joints of the robot.
    :rtype: tuple (list of str, list of str, list of str)
    """

    try:
        from ament_index_python.packages import get_package_share_directory
        joint_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "config/joints_list.yaml")
    except:
        joint_path = os.getcwd()[:-7] + "config/joints_list.yaml"

    with open(joint_path, 'r') as infp:
        pid_txt = infp.read()
        joints_dict = yaml.load(pid_txt, Loader=yaml.Loader)
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


class JointInterpolation:
    """
    A class to manage interpolation of robot joint states (position, velocity, torque) over time.

    This is done with cubic spline interpolation of joint positions, velocities, and torques.

    :ivar pos_err: The allowable error in position.
    :vartype pos_err: float
    :ivar vel_err: The allowable error in velocity.
    :vartype vel_err: float
    :ivar joint_num: The number of joints to be interpolated.
    :vartype joint_num: int
    :ivar pos_arr: The latest joint position array.
    :vartype pos_arr: np.ndarray
    :ivar vel_arr: The latest joint velocity array.
    :vartype vel_arr: np.ndarray
    :ivar timelist: List of timestamps corresponding to joint states.
    :vartype timelist: np.ndarray
    :ivar cs_pos: The current cubic spline interpolation for joint positions.
    :vartype cs_pos: np.ndarray
    :ivar cs_vel: The current cubic spline interpolation for joint velocities.
    :vartype cs_vel: np.ndarray
    :ivar cs_tau: The current cubic spline interpolation for joint torques.
    :vartype cs_tau: np.ndarray
    :ivar cs_centroid: The current cubic spline interpolation for the centroid of the robot.
    :vartype cs_centroid: np.ndarray
    :ivar consecutive_fails: The number of consecutive fails (i.e. error crosses threshold).
    :vartype consecutive_fails: int
    :ivar fail_thresh: The threshold for the number of consecutive failures.
    :vartype  fail_thresh: int

    """

    def __init__(self, joint_num, position_error, velocity_error):
        """
        Constructor method
        """
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

    def updateJointState(self, timelist, pos, vel, centroid_vec=None):
        """
        Updates the joint state with new positions and velocities, using cubic splines to interpolate.

        :param timelist: The list of timestamps corresponding to the sequence of joint states.
        :type timelist: np.ndarray
        :param pos: The new joint positions (shape: [sequence length, joint_num]).
        :type pos: np.ndarray
        :param vel: The new joint velocities (shape: [sequence length, joint_num]).
        :type vel: np.ndarray
        :param centroid_vec: _description_, defaults to None
        :type centroid_vec: _type_, optional
        :return: A tuple containing:
            - True if the update succeeded, otherwise false
            - The mean position error
            - The mean velocity error
        :rtype: tuple (bool, float, float)
        """

        # Of shape (seq len, joint num)
        # make existing sequency with cs and interpolate and filter
        if self.cs_pos is not None:
            match_pos = self.cs_pos(timelist)
            match_vel = self.cs_vel(timelist)
            new_pos = pos * 0.5 + match_pos * 0.5
            new_vel = vel * 0.5 + match_vel * 0.5
        else:
            new_pos = pos
            new_vel = vel
        cs_pos = scipy.interpolate.CubicSpline(timelist, new_pos, axis=0)
        cs_vel = scipy.interpolate.CubicSpline(timelist, new_vel, axis=0)
        if centroid_vec is not None:
            self.cs_centroid = scipy.interpolate.CubicSpline(timelist, centroid_vec, axis=0)

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
        """
        Forcefully updates the interpolations for joint positions, velocities, and torques with the
        given data. If previous interpolations exist, the new and old data is blended together,
        weighting the old data higher and new data lower in earlier time steps, and vice versa
        for later time steps.

        :param timelist: The list of timestamps corresponding to the sequence of joint states.
        :type timelist: np.ndarray
        :param pos: The sequence of new joint positions.
        :type pos: np.ndarray
        :param vel: The sequence of new joint velocities.
        :type vel: np.ndarray
        :param tau: The sequence of new joint torques.
        :type tau: np.ndarray
        """

        if self.cs_pos is not None and self.cs_vel is not None:
            match_pos = self.cs_pos(timelist)
            match_vel = self.cs_vel(timelist)
            match_tau = self.cs_tau(timelist)

            sz = match_pos.shape[0]
            weight_vec = 0.0 + 2 * ((np.arange(sz) - 1) / sz)
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
        """
        Updates the interpolations for joint positions, velcities, and torques with timestamps
        in timelist that are past current_time. If previous interpolations exist, the new and old
        data is blended together, weighting the old data higher and new data lower in earlier time
        steps, and vice versa for later time steps.

        :param current_time: The current timestamp at which the interpolation starts
        :type current_time: float
        :param timelist: The list of timestamps corresponding to the sequence of joint states.
        :type timelist: np.ndarray
        :param pos: The sequence of new joint positions.
        :type pos: np.ndarray
        :param vel: The sequence of new joint velocities.
        :type vel: np.ndarray
        :param tau: The sequence of new joint torques.
        :type tau: np.ndarray
        """

        self.cs_tau = scipy.interpolate.CubicSpline(timelist, tau, axis=0)
        if self.cs_pos is None or self.cs_vel is None:
            self.cs_pos = scipy.interpolate.CubicSpline(timelist, pos, axis=0)
            self.cs_vel = scipy.interpolate.CubicSpline(timelist, vel, axis=0)
        else:
            new_timelist = np.concatenate([np.array([current_time]), timelist[:]], axis=0)
            new_timelist = np.sort(np.array(list(set(list(new_timelist)))))
            new_timelist = new_timelist[np.where(new_timelist == current_time)[0][0]:]

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
        """
        Updates the joint and centroid states based on a provided state vector.

        :param timelist: The list of timestamps corresponding to the sequence of joint states.
        :type timelist: np.ndarray
        :param x: The state vector containing information about the centroid and joints
        :type x: np.ndarray
        :return: Result of the updateJointState method
        :rtype: tuple (bool, float, float)
        """

        centroid_pos = x[:, 0:7]
        pos = x[:, 7:7 + self.joint_num]
        centroid_vel = x[:, 7 + self.joint_num: 13 + self.joint_num]
        vel = x[:, 13 + self.joint_num:]
        centroid = np.concatenate([centroid_pos, centroid_vel], axis=1)
        return self.updateJointState(timelist, pos, vel, centroid_vec=centroid)

    def checkDelta(self, check_time, pos, vel):
        """
        Checks that the given joint positions and velocities at a specific time point are within
        the acceptable error range of the interpolated joint positions and velocities.

        :param check_time: The point in time to compare positions and velocities.
        :type check_time: float or int
        :param pos: The provided joint positions.
        :type pos: np.ndarray
        :param vel: The provided joint velocities.
        :type vel: np.ndarray
        :return: True if difference between given and interpolated joint positions/velocities are within the acceptable error range.
        :rtype: bool
        """

        check_pos = self.cs_pos(check_time)
        check_vel = self.cs_vel(check_time)
        pc = np.sum(np.abs(pos - check_pos) < self.pos_err)
        vc = np.sum(np.abs(vel - check_vel) < self.vel_err)
        return pc == 0 and vc == 0

    def getInterpolation(self, timestamp, pos_delta_time=0):
        """
        Retrieves interpolated joint position, velocity, (and torque if available) at a given timestamp,
        with an optional adjustment to the time at which the position is retrieved from the interpolation.

        :param timestamp: The timestamp at which to retrieve the interpolated position and velocity
        :type timestamp: float
        :param pos_delta_time: An optional adjustment to the timestamp for position interpolation, defaults to 0
        :type pos_delta_time: float, optional
        :return: A tuple containing the interpolated position, velocity, and torque (if available)
        :rtype: tuple (np.ndarray, np.ndarray) or (np.ndarray, np.ndarray, np.ndarray)
        """

        pos = self.cs_pos(timestamp + pos_delta_time)
        vel = self.cs_vel(timestamp)
        if self.cs_tau is None:
            return pos, vel
        else:
            tau = self.cs_tau(timestamp)
            return pos, vel, tau

    def getX(self, t):
        """
        Retrieves and reconstructs the full state vector of the robot from the interpolation at a given time t.

        :param t: The timestamp at which to retriev the state vector
        :type t: float
        :return: The robot state vector, where the first 7 values contain information about the centroid
                 (translation, quaternion), the next n_joints entries contain the joint positions, the
                 next 6 entries are the linear and angular velocities, and the last n_joints
                 entries contain the joint velocities.
        :rtype: numpy.ndarray
        """

        pos, vel = self.getInterpolation(t)
        centroid = self.cs_centroid(t)
        centroid[3:7] = centroid[3:7] / (np.sum(centroid[3:7] ** 2) ** 0.5)
        x0 = np.concatenate([centroid[:7], pos, centroid[7:], vel], axis=0)
        return x0

    def getSeedX(self, timestamps):
        """
        Retrieves a list of robot state vectors for a series of timestamps.

        :param timestamps: A list or array of timestamps for which to retrieve state vectors for
        :type timestamps: list or np.ndarray
        :return: A list of robot state vectors that correspond to the given timestamps
        :rtype: list of np.ndarray
        """
        x = []
        for t in timestamps:
            x0 = self.getX(t)
            x += [x0]
        return x

    def hasHistory(self):
        """
        Checks if there is any previous interpolation stored in the class.

        :return: True if both self.cs_pos and self.cs_vel exist
        :rtype: bool
        """
        return not (self.cs_pos is None or self.cs_vel is None)


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
        state_r = np.concatenate([pos, vel], axis=0)
        error = np.linalg.norm(self.state_samples[:, :self.joint_num] - state_r[:self.joint_num], axis=1)
        index = np.argmin(error)
        ref = self.tau_samples[index, :]
        weight = 0.5 + 0.0 * np.arange(timestamp.shape[0]) / timestamp.shape[0]
        weight[weight > 1] = 1

        pos = self.cs_pos(timestamp)
        vel = self.cs_vel(timestamp)
        tau = self.cs_tau(
            timestamp)  # * weight[:, None] + np.tile(ref[None, :], [timestamp.shape[0], 1]) * (1 - weight)[:, None]

        return pos, vel, tau

    def hasHistory(self):
        return not (self.cs_pos is None or self.cs_vel is None or self.cs_tau is None)

    def updateMixState(self, current_time, timelist, pos, vel, tau):
        state = np.concatenate([np.array(pos[1:, :]), np.array(vel[1:, :])], 1)
        self.state_samples = np.concatenate([state, self.state_samples], axis=0)
        self.tau_samples = np.concatenate([tau[1:, :], self.tau_samples], axis=0)
        if self.state_samples.shape[0] > self.max_points:
            self.state_samples = self.state_samples[:self.max_points, :]
            self.tau_samples = self.tau_samples[:self.max_points, :]
        self.cs_tau = scipy.interpolate.CubicSpline(timelist[:], tau[:, :], axis=0)

        if self.cs_pos is None or self.cs_vel is None:
            self.cs_pos = scipy.interpolate.CubicSpline(timelist, pos, axis=0)
            self.cs_vel = scipy.interpolate.CubicSpline(timelist, vel, axis=0)
        else:
            new_timelist = np.concatenate([np.array([current_time]), timelist[:]], axis=0)
            new_timelist = np.sort(np.array(list(set(list(new_timelist)))))
            new_timelist = new_timelist[np.where(new_timelist == current_time)[0][0]:]

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
    """
    A class to help with applying a low-pass Butterworth filter to a signal.

    :ivar sos: The second-order sections representation of the low-pass filter.
    :vartype: np.ndarray
    :ivar zi: A list of the initial conditions of the filters
    :vartype: list of np.ndarray
    :ivar y: An array of the most recent filtered values for each signal point.
    :vartype: np.ndarray
    """

    def __init__(self, n_signal, freq, cutoff):
        """
        Constructor method

        :param n_signal: The number of signal points (dimension) to filter.
        :type params: int
        :param freq: The sampling frequency of the signal.
        :type freq: float
        :param cutoff: The cutoff frequency of the filter.
        :type cutoff: float
        """

        # self.b, self.a = scipy.signal.butter(4, cutoff, btype='low', analog=False, fs = freq)
        nyquist = 0.5 * freq
        normal_cutoff = cutoff / nyquist
        self.sos = scipy.signal.butter(4, normal_cutoff, btype='low', analog=False, output='sos')
        self.zi = []
        self.y = np.zeros(n_signal)  # filter results
        for c in range(n_signal):
            self.zi += [scipy.signal.sosfilt_zi(self.sos)]

    def update(self, vec):
        """
        Apply the filter to a set of input signals and update the conditions of the filters.

        :param vec: A 2-D array where each row is a signal to be filtered. (the length of it should = n_signal)
        :type vec: np.ndarray
        """
        for c in range(vec.shape[0]):
            filtered_point, self.zi[c] = scipy.signal.sosfilt(self.sos, vec[c:c + 1], zi=self.zi[c], axis=0)
            self.y[c] = filtered_point[0]

    def get(self):
        """
        Retrieves list of filtered signals.

        :return: List of filtered signals.
        :rtype: List of np.ndarray
        """

        return self.y


class CSVDump:
    def __init__(self, params, name_list):
        self.abs_path = "/home/aurum/RosProjects/prairie/datadump"
        self.name_list = name_list
        self.max_num = 5000
        self.arr = np.zeros([self.max_num, params, len(name_list)])

    def update(self, vecs):
        self.arr = np.roll(self.arr, -1, axis=0)
        for c in range(len(vecs)):
            self.arr[-1, :, c] = vecs[c]

    def save(self):
        if os.path.exists(self.abs_path):
            for c in range(len(self.name_list)):
                name = self.name_list[c]
                path = self.abs_path + "/{}.csv".format(name)
                np.savetxt(path, self.arr[:, :, c], delimiter=',')


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


class ForwardPoser:
    def __init__(self, urdf_path, joint_list, leg_joints=None):
        self.joint_list = joint_list
        if leg_joints is not None:
            self.leg_joints = leg_joints
        else:
            self.leg_joints = joint_list

        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        lock_joints = []
        for joint in self.joint_list:
            if not (joint in self.leg_joints):
                num = self.model.getJointId(joint)
                if num not in lock_joints:
                    lock_joints += [num]
        self.q = np.zeros([len(self.joint_list) + 7])
        self.q_r = np.zeros([len(self.leg_joints) + 7])
        self.q_r[6] = 1
        self.q[6] = 1
        self.model_r = pin.buildReducedModel(self.model,
                                             list_of_joints_to_lock=lock_joints,
                                             reference_configuration=self.q)
        self.data_r = self.model_r.createData()
        self.data = self.model.createData()

        self.m = osqp.OSQP()
        self.qp_weights = {"control": 1}
        self.epsilons = {"com_pos": np.array([0.01, 0.01, 0.01]),
                         "link": np.array([0.001, 0.001, 0.001, 0.01, 0.01, 0.01])}

    def updateReducedQ(self, centroid_pos, centroid_orien, joint_pos_dict):
        vec = np.zeros([len(self.leg_joints)])
        for key in joint_pos_dict.keys():
            if key in self.leg_joints:
                index = self.model_r.getJointId(key) - 2
                vec[index] = joint_pos_dict[key]
        self.q_r = np.concatenate([centroid_pos, centroid_orien, vec], axis=0)

    def config2Vec(self, config_dict):
        num_joints = len(self.joint_list)
        vec = np.zeros([num_joints])
        for key in config_dict.keys():
            index = self.model.getJointId(key) - 2
            vec[index] = config_dict[key]
        return vec

    def updateData(self, centroid_pos, centroid_orien, joint_pos_dict):
        # pos: xyz, orien: xyzw
        q_joints = self.config2Vec(joint_pos_dict)
        self.q = np.concatenate([centroid_pos, centroid_orien, q_joints], axis=0)
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

    def getLinkPose(self, link_name):
        id = self.model.getFrameId(link_name)
        pos = np.array(self.data.oMf[id].translation)
        return pos

    def getRLinkSE3(self, link_name):
        id = self.model_r.getFrameId(link_name)
        return self.data_r.oMf[id]

    def updateDataR(self):
        pin.forwardKinematics(self.model_r, self.data_r, self.q_r)
        pin.updateFramePlacements(self.model_r, self.data_r)

    def getCOMPos(self):
        if self.q is not None:
            com_pos = np.array(pin.centerOfMass(self.model, self.data, self.q))
            return com_pos
        return None

    def jacobianCOMCorrection(self, desired_cpos, desired_corien, desired_jpos, contacts):
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        current_com = np.array(pin.centerOfMass(self.model, self.data, self.q))

        jac_com = pin.jacobianCenterOfMass(self.model, self.data, self.q)[:, 6:]

        jac_contacts = np.zeros([jac_com.shape[0], jac_com.shape[1]])
        for contact in contacts:
            jac_contact = pin.computeFrameJacobian(self.model, self.data, self.q,
                                                   self.model.getFrameId(contact))
            jac_contacts += jac_contact[:3, 6:] / len(contacts)

        jac = jac_com - jac_contacts
        inv_jac = np.linalg.pinv(jac)

        q_joints = self.config2Vec(desired_jpos)
        q_desired = np.concatenate([desired_cpos, desired_corien, q_joints], axis=0)
        pin.forwardKinematics(self.model, self.data, q_desired)
        pin.updateFramePlacements(self.model, self.data)
        desired_com = np.array(pin.centerOfMass(self.model, self.data, q_desired))
        delta_com = desired_com - current_com
        delta_r = np.sum(inv_jac * np.tile(delta_com[None, :], [inv_jac.shape[0], 1]), axis=1)

        names = self.model.names.tolist()
        joint_dict = {}
        for c in range(len(names) - 2):
            joint_dict[names[c + 2]] = delta_r[c]

        return joint_dict

    def jacobianTorqueForce(self, torques, contacts):
        torque_vec = self.config2Vec(torques)
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        contact_forces = []
        for contact in contacts:
            jac_contacts = pin.computeFrameJacobian(self.model, self.data, self.q,
                                                    self.model.getFrameId(contact))[:3, 6:]
            contact_force = np.sum(jac_contacts * torque_vec[None, :], axis=1)
            contact_forces += [contact_force]
        return contact_forces

    def ikSolver(self, target_com, link_target_dict):
        def makeQC(a, b):
            q = np.matmul(np.transpose(a), a)
            c = -1 * np.matmul(np.transpose(a), b)
            return q, c

        # compute IK solution for achieving a target com position, target l foot pos, and target r foot pos
        pin_dict = {}
        for link in link_target_dict.keys():
            if isinstance(link_target_dict[link], np.ndarray):
                se3 = pin.SE3(np.eye(3), link_target_dict[link])
            else:
                se3 = link_target_dict[link]
            pin_dict[link] = (se3,
                              self.model_r.getFrameId(link))
        ref_state = self.q_r.copy()
        for c in range(2):
            pin.forwardKinematics(self.model_r, self.data_r, ref_state)
            pin.updateFramePlacements(self.model_r, self.data_r)
            current_com = np.array(pin.centerOfMass(self.model_r, self.data_r, ref_state, False))
            r_mat = np.zeros([self.model_r.nv, self.model_r.nv])
            d_vec = np.zeros([self.model_r.nv])

            a = np.eye(len(self.leg_joints))
            a = np.concatenate([np.zeros([a.shape[0], 6]), a], axis=1)

            u = np.array(self.model_r.upperPositionLimit[7:]) - ref_state[7:]
            l = np.array(self.model_r.lowerPositionLimit[7:]) - ref_state[7:]

            j_com = pin.jacobianCenterOfMass(self.model_r, self.data_r, ref_state, False)
            j_com = j_com
            d_com = (target_com - current_com)

            a = np.concatenate([a, j_com], axis=0)
            u = np.concatenate([u, d_com + self.epsilons["com_pos"]], axis=0)
            l = np.concatenate([l, d_com - self.epsilons["com_pos"]], axis=0)

            # (matrix 6 x model.nv)
            # Each column represents the x y z roll pitch yaw
            for link in pin_dict.keys():
                inv_transform = pin_dict[link][0].actInv(self.data_r.oMf[pin_dict[link][1]])
                err_vecs = np.array(pin.log(inv_transform)) * -1
                j = pin.computeFrameJacobian(self.model_r, self.data_r, ref_state, pin_dict[link][1])
                j = j
                d = err_vecs

                a = np.concatenate([a, j], axis=0)
                u = np.concatenate([u, d + self.epsilons["link"]], axis=0)
                l = np.concatenate([l, d - self.epsilons["link"]], axis=0)

            j_norm = np.eye(self.model_r.nv) * self.qp_weights["control"]
            d_norm = np.zeros(self.model_r.nv)
            q, c = makeQC(j_norm, d_norm)
            r_mat += q
            d_vec += c

            # R mat and d vec constructed, build P and q from least squares

            p = np.matmul(np.transpose(r_mat), r_mat)
            p = sparse.csc_matrix(p)
            q = np.matmul(np.transpose(r_mat), d_vec)
            # Constraints are joint limits

            a = sparse.csc_matrix(a)

            # u = np.ones(u.shape) * np.inf
            # l = np.ones(u.shape) * -np.inf
            self.m = osqp.OSQP()
            self.m.setup(P=p, q=q, A=a, l=l, u=u, verbose=False)
            results = self.m.solve()
            ref_state = pin.integrate(self.model_r, ref_state, results.x * 1)

        pos, quat, jd = self.qUnpack(ref_state)
        return pos, quat, jd, ref_state

    def qUnpack(self, q):
        names = self.model_r.names.tolist()
        joint_dict = {}
        for c in range(len(names) - 2):
            joint_dict[names[c + 2]] = q[c + 7]
        pos = q[0:3]
        quat = q[3:7]
        return pos, quat, joint_dict


"""
class SimpleFwdInvSM:
    def __init__(self, poser):
        self.poser = poser
        self.us = None
        self.solved = False

    def makeFwdInvProblem(self, timestamps, inverse_commands):
        self.poser.pose_dict = None
        ts_prev = 0
        models = []
        for timestamp, inverse_command in zip(timestamps, inverse_commands):
            dmodel = self.poser.makeInverseCmdDmodel(inverse_command)
            model = self.poser.makeD2M(dmodel, timestamp - ts_prev)
            ts_prev = timestamp
            models += [model]
        final_dmodel = self.poser.makeInverseCmdDmodel(inverse_commands[-1])
        final_model = self.poser.makeD2M(final_dmodel, 0)
        return models, final_model

    def nextMPC(self, timestamps, inverse_commands, xs):
        traj, final = self.makeFwdInvProblem(timestamps, inverse_commands)
        x0 = self.poser.x.copy()
        q0 = x0[0:7 + len(self.poser.leg_joints)]
        problem = crocoddyl.ShootingProblem(x0, traj, final)
        fddp = crocoddyl.SolverBoxFDDP(problem)
        fddp.th_stop = 1e5
        if xs is None:
            init_xs = [x0] * (problem.T + 1)
        else:
            init_xs = []
            for c in range(xs.shape[0]):
                init_xs += [xs[c, :]]

        init_us = []
        if len(inverse_commands) > 50:
            maxiter = 200
        else:
            maxiter = 10
        regInit = 1.0
        self.solved = fddp.solve(init_xs, init_us, maxiter, False, regInit)
        xs = np.array(fddp.xs)
        self.us = np.array(fddp.us)
        return xs, self.us

class SquatSM:
    def __init__(self, poser, com_pos):
        self.poser = poser
        self.fast_dt = 0.01
        self.slow_dt = 0.05
        self.com_pos = com_pos
        #self.com_pos = np.array([0., 0., np.sin(time.time()) * 0.1 + 0.5])
        self.ts_xs = None
        self.xs = None
        self.y = None
        self.us = None


    def makeSquatProblem(self, timesteps, dt):
        dmodel = self.poser.dualSupportDModel(com_target=self.com_pos)
        model = self.poser.makeD2M(dmodel, dt)
        models = [model] * timesteps
        final = self.poser.makeD2M(dmodel , 0.)
        return models, final


    def simpleNextMPC(self, init_xs):
        traj, final = self.makeSquatProblem(9, 0.03)
        x0 = self.poser.x.copy()
        q0 = x0[0:7+len(self.poser.leg_joints)]
        l, r, com = self.poser.getPos(q0)
        problem = crocoddyl.ShootingProblem(x0, traj, final)
        fddp = crocoddyl.SolverFDDP(problem)
        fddp.th_stop = 1e5
        if init_xs is None:
            init_xs = [x0] * (problem.T + 1)
        init_us = []
        maxiter = 20
        regInit = 0.1
        solved = fddp.solve(init_xs, init_us, maxiter, False, regInit)
        #print(solved)
        xs = np.array(fddp.xs)
        self.us = np.array(fddp.us)
        return xs

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

def constrained_regression(X, Y, fixed_point_X, fixed_point_y):
    n, m = X.shape

    D = np.hstack([X, np.ones((n, 1))])  # n x (m+1)

    M = D.T @ D  # (m+1) x (m+1)
    DTy = D.T @ Y  # (m+1) x 1

    C = np.append(fixed_point_X, 1)  # (m+1) vector

    AUG_MAT = np.zeros((m+2, m+2))
    AUG_MAT[:m+1, :m+1] = M
    AUG_MAT[:m+1, m+1] = C
    AUG_MAT[m+1, :m+1] = C
    AUG_MAT[m+1, m+1] = 0

    AUG_VEC = np.zeros(m+2)
    AUG_VEC[:m+1] = DTy
    AUG_VEC[m+1] = fixed_point_y

    SOL = np.linalg.solve(AUG_MAT, AUG_VEC)
    beta = SOL[:m+1]

    A = beta[:m]
    B = beta[m]

    return A, B

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


class BipedalGait:

    def __init__(self, step_length, step_height):
        self.step_height = step_height
        self.step_length = step_length
        self.jlc, self.jlm, self.leg_joints = makeJointList()

    def dualSupport(self, com, com_vel, com_acc, support_contact):
        ic = InverseCommand()

        ic.state_cost = float(1e1)
        ic.torque_cost = float(1e1)
        pelvis_pose = makePose([0, 0, 0], [0, 0, 0, 1])
        ic.link_poses = [pelvis_pose]
        ic.link_pose_names = ["pelvis"]
        ic.link_costs = [float(1e8)]
        ic.link_orien_weight = [float(10000)]
        ic.link_vel_costs = [float(0.)]
        ic.link_clip_costs = [0.]
        ic.link_contacts = ["left_ankle_roll_link", "right_ankle_roll_link"]
        ic.contact_vel_costs = [float(1e11), float(1e11)]


        if support_contact == "left_ankle_roll_link":
            ic.friction_contact_costs = [float(1e12), float(1e12)]
            ic.cop_costs = [float(0.), float(0)]
            ic.force_limit_costs = [float(1e3), float(1e2)]
        else:
            ic.friction_contact_costs = [float(1e12), float(1e12)]
            ic.cop_costs = [float(0), float(0.)]
            ic.force_limit_costs = [float(1e2), float(1e3)]
        ic.contact_force_cap = [float(1e4), float(1e4)]
        ic.com_pos = np2Point(com)
        ic.com_vel = np2Point(com_vel)
        ic.com_acc = np2Point(com_acc)
        ic.com_cost = [float(1e10), float(0.), float(0.)]

        ic.max_linear_vel = 0.8
        ic.max_ang_vel = 0.8
        ic.state_limit_cost = 1e8
        ic.centroid_vel_cost = 0.
        ic.joint_acceleration_cost = 1e2
        ic.com_height_only = False
        return ic

    def singleSupport(self, contact_link, move_link, move_pos, move_orien, com, com_vel, com_acc):
        ic = InverseCommand()

        ic.state_cost = float(1e1)
        ic.torque_cost = float(1e0)
        move_pose = makePose(move_pos, move_orien)
        zero_pos = makePose(np.array([0, 0, 0]), np.array([0,0,0,1]))
        ic.link_poses = [move_pose, zero_pos]
        ic.link_pose_names = [move_link, "pelvis"]
        ic.link_costs = [float(1e8), float(1e8)]
        ic.link_orien_weight = [float(1), float(10000)]
        ic.link_vel_costs = [float(0.), 0.]
        ic.link_clip_costs = [float(1e7), 0.]
        ic.link_contacts = [contact_link]
        ic.contact_vel_costs = [float(1e8)]
        ic.friction_contact_costs = [float(1e3)]
        ic.force_limit_costs = [float(0)]
        ic.cop_costs = [float(1e3)]
        ic.contact_force_cap = [float(1e9)]
        ic.com_pos = np2Point(com)
        ic.com_vel = np2Point(com_vel)
        ic.com_acc = np2Point(com_acc)
        ic.com_cost = [float(1e10), float(0.), float(0.)]
        ic.max_linear_vel = 0.8
        ic.max_ang_vel = 0.8
        ic.state_limit_cost = 1e7
        ic.centroid_vel_cost = 0.
        ic.joint_acceleration_cost = 1e2
        ic.com_height_only = True
        return ic

    def swingTrajectory(self, initial_pos, final_pos, prop):
        #initial pos and final pos are  3d with 0 in index 2
        ref_xy = initial_pos * (1 - prop) + final_pos * prop
        z = ( prop * (1 - prop) ) * 4 * self.step_height
        ref_xy[2] = ref_xy[2] + z
        return ref_xy

    def reflessWalk(self):
        left_pos = np.array([-0.003, 0.12, 0.01])
        right_pos = np.array([-0.003, -0.12, 0.01])
        com_pos = np.array([0., 0., 0.65])

        bpc = BipedalCommand()

        timestamps = 0.02 + np.arange(160) * 0.02
        #Dual support settle for 0.4 secs or 40 ts
        ics = []
        for c in range(40):
            prop = c / 39
            #settle_com = com_pos * (1 - prop) + prop * np.array([0., 0.08, 0.55])
            ic = self.dualSupport(np.array([0., 0.08, 0.55]), None, "left_ankle_roll_link")
            ics += [ic]
        com_pos = np.array([0., 0.12, 0.55])
        #single support left foot
        right_pos2 = right_pos.copy() + np.array([self.step_length/2, 0., 0.0])
        #right_pos2 = np.array([0.1, -0.12, 0.01])
        for c in range(30):
            prop = c / 29
            rlink = self.swingTrajectory(right_pos, right_pos2, prop)
            shift_com = np.array([self.step_length/2, -0.09, 0.55])
            ic = self.singleSupport("left_ankle_roll_link", "right_ankle_roll_link", rlink, np.array([0,0,0,1]), shift_com)
            ics += [ic]
        right_pos = right_pos2.copy()
        com_pos = np.array([self.step_length/2, -0.09, 0.55])
        #dual support 0.1 s
        for c in range(5):
            ic = self.dualSupport(com_pos, None, "right_ankle_roll_link")
            ics += [ic]

        left_pos2 = left_pos + np.array([self.step_length, 0., 0.0])
        for c in range(40):
            prop = c / 39
            llink = self.swingTrajectory(left_pos, left_pos2, prop)
            shift_com = np.array([self.step_length, 0.09, 0.55])
            ic = self.singleSupport("right_ankle_roll_link", "left_ankle_roll_link", llink, np.array([0,0,0,1]), shift_com)
            ics += [ic]

        com_pos = np.array([self.step_length, 0.07, 0.55])

        for c in range(5):
            ic = self.dualSupport(com_pos, None, "left_ankle_roll_link")
            ics += [ic]

        right_pos2 = right_pos.copy() + np.array([self.step_length, 0., 0.0])
        for c in range(40):
            prop = c / 39
            rlink = self.swingTrajectory(right_pos, right_pos2, prop)
            shift_com = np.array([self.step_length*3/2, -0.09, 0.55])
            ic = self.singleSupport("left_ankle_roll_link", "right_ankle_roll_link", rlink, np.array([0,0,0,1]), shift_com)
            ics += [ic]

        com_pos = np.array([self.step_length*3/2, -0.07, 0.55])
        for c in range(5):
            ic = self.dualSupport(com_pos, None, "right_ankle_roll_link")
            ics += [ic]

        bpc.inverse_timestamps = timestamps.tolist()
        bpc.inverse_commands = ics
        bpc.inverse_joints = self.leg_joints
        return bpc

def idleInvCmd():
    bpc = BipedalCommand()
    bpc.inverse_timestamps = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    ics = []

    for c in range(9):
        ic = InverseCommand()

        ic.state_cost = float(2e2)
        ic.torque_cost = float(1e-2)
        pelvis_pose = makePose([0, 0, 0.7], [0, 0, 0, 1])
        ic.link_poses = [pelvis_pose]
        ic.link_pose_names = ["pelvis"]
        ic.link_costs = [float(1e6)]
        ic.link_orien_weight = [float(100000), float(1), float(1)]
        ic.link_contacts = ["left_ankle_roll_link", "right_ankle_roll_link"]
        ic.friction_contact_costs = [float(1e3), float(1e3)]
        ic.contact_lock_costs = [float(0), float(0)]
        com_pos = Point()
        com_pos.x = 0.03
        com_pos.y = 0.0
        com_pos.z = 0.63
        ic.com_pos = com_pos
        ic.com_cost = float(0)

        ics += [ic]

    bpc.inverse_commands = ics
    _, _, leg_joints = makeJointList()
    bpc.inverse_joints = leg_joints
    return bpc

class WalkingSM:
    def __init__(self):
        #4 states DS_SR, DS_CR, SR, DS_SL, DS_CL, SL
        #DS_SR: Dual support state end cond, prep for right swing
        #DS_CR: Dual support countdown end cond, prep for right swing
        #SR: right swing foot
        #DS_SL: Dual support state end cond, prep for left swing
        #DS_CL: Dual support countdown end cond, prep for left swing
        #SL: left swing foot
        JOINT_LIST_FULL, JOINT_LIST, LEG_JOINTS = makeJointList()
        self.current_state = "DS_SR"
        self.countdown_start = 0
        self.countdown_duration = 0.05
        from ament_index_python.packages import get_package_share_directory
        urdf_config_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "urdf/g1_meshless.urdf")
        self.fwd_poser = ForwardPoser(urdf_config_path, JOINT_LIST)

    def updatePoser(self, state_dict):
        self.fwd_poser.updateData(state_dict["pos"], state_dict["orien"], state_dict["joint_pos"])

    def updateSM(self, state_time, start_swing_target, current_swing_target):
        if self.current_state[:4] == "DS_C":
            if state_time - self.countdown_start > self.countdown_duration:
                if self.current_state == "DS_CR":
                    self.current_state = "SR"
                else:
                    self.current_state = "SL"
                return True
            else:
                return False
        if self.current_state[0] == "S":
            if self.current_state == "SL":
                swing_link = "left_ankle_roll_link"
            else:
                swing_link = "right_ankle_roll_link"
            link_pos = self.fwd_poser.getLinkPose(swing_link)
            xy_t = np.linalg.norm(link_pos[0:2] - current_swing_target[0:2])
            xy_i = np.linalg.norm(link_pos[0:2] - start_swing_target[0:2])
            z = abs(link_pos[2] - current_swing_target[2])
            if xy_t < xy_i and link_pos[2] < 0.02:
                if self.current_state == "SL":
                    self.current_state = "DS_SR"
                else:
                    self.current_state = "DS_SL"
                return True
            return False
        if self.current_state[:4] == "DS_S":
            com_pos = self.fwd_poser.getCOMPos()
            if self.current_state == "DS_SL":
                support_pos = self.fwd_poser.getLinkPose("right_ankle_roll_link")
            else:
                support_pos = self.fwd_poser.getLinkPose("left_ankle_roll_link")
            xy = np.linalg.norm(com_pos[1] - support_pos[1])
            if xy < 0.07:
                self.countdown_start = state_time
                if self.current_state == "DS_SL":
                    self.current_state = "DS_CL"
                    return True
                else:
                    self.current_state = "DS_CR"
                    return True
            else:
                return False
        return False

    def updateState(self, state_dict, state_time, start_swing_target, current_swing_target):
        self.updatePoser(state_dict)
        #SL SR determined by xy being within 0.05m of the target and z being 0.01 off
        #DS_SL/R determined by COM xy being withing 0.05m of support foot
        return self.updateSM(state_time, start_swing_target, current_swing_target)

class SimpleFootstepPlan:
    def __init__(self):
        self.step_length = 0.2
        self.step_height = 0.05
        self.swing_time = 0.4
        self.support_time = 0.1
        self.z_height = 0.6
        self.step_speed = self.step_length / self.swing_time
        self.com_speed = 0.06 / self.support_time

        self.horizon_ts = [0.01, 0.02, 0.03]

        self.initial_l_pos = np.array([-0.003, 0.12, 0.01])
        self.initial_r_pos = np.array([-0.003, -0.12, 0.01])

        self.plan = self.makeFootStepPlan()

    def makeFootStepPlan(self):
        step_no = 10
        left_pos = np.array([-0.003, 0.08, 0.01]) + np.array([self.step_length * 1, 0, 0])
        right_pos = np.array([-0.003, -0.08, 0.01]) + np.array([self.step_length * 0.5, 0, 0])
        ref_plan = []
        ref_plan += [("R", right_pos.copy(), self.initial_r_pos, self.initial_l_pos),
                     ("L", left_pos.copy(), self.initial_l_pos, right_pos.copy())]
        for c in range(step_no):
            left_pos += np.array([self.step_length, 0, 0])
            right_pos += np.array([self.step_length, 0, 0])
            ref_plan += [
                ("R", right_pos.copy(), right_pos.copy() - np.array([self.step_length, 0, 0]), left_pos.copy()  - np.array([self.step_length, 0, 0])),
                ("L", left_pos.copy(), left_pos.copy() - np.array([self.step_length, 0, 0]), right_pos.copy())]

        return ref_plan

    def swingFootPoints(self, initial_pos, swing_target, pos_c):
        time_remaining = np.linalg.norm(swing_target[:2] - pos_c[:2]) / self.step_speed
        time_remaining = min(self.swing_time, max(time_remaining, 0))

        horizon_ts = self.horizon_ts

        link_pos = []

        for c in range(len(horizon_ts)):
            prop = horizon_ts[c] / time_remaining
            prop = min(max(prop, 0), 1)
            xy_pos = swing_target * prop + pos_c * (1 - prop)
            xy_i = np.linalg.norm(xy_pos - initial_pos)
            xy_f = np.linalg.norm(xy_pos - swing_target)
            length = np.linalg.norm(initial_pos - swing_target)

            blind_height = 1 - ((xy_f - xy_i) ** 2 / (length ** 2))
            blind_height = blind_height * self.step_height
            blind_height = min(max(0, blind_height), self.step_height)
            xy_pos[2] = pos_c[2] * (1 - prop) + (blind_height + initial_pos[2]) * prop
            xy_pos[2] = min(xy_pos[2], self.step_height)
            if xy_f > xy_i:
                xy_pos[2] = max(xy_pos[2], 0.04)
            link_pos += [xy_pos]
        horizon_ts = list(horizon_ts)
        return horizon_ts, link_pos
"""

"""
class SVFwdKinematics:
    def __init__(self):
        from ament_index_python.packages import get_package_share_directory
        JOINT_LIST_FULL, JOINT_LIST, LEG_JOINTS = makeJointList()
        urdf_config_path = os.path.join(
            get_package_share_directory('hrc_handler'),
            "urdf/g1_meshless.urdf")
        self.fwd_poser = ForwardPoser(urdf_config_path, JOINT_LIST)

        self.com_vel_filt = SignalFilter(3, 10000, 400)
        self.com_acc_filt = SignalFilter(3, 10000, 400)
        self.prev_com = np.zeros([3])
        self.prev_com_vel = np.zeros([3])

        self.prev_time = 0

    def update(self, state_vector):
        j_pos_config = dict(zip(state_vector.joint_name, list(state_vector.joint_pos)))
        self.fwd_poser.updateData(state_vector.pos, state_vector.orien_quat, j_pos_config)
        com = self.fwd_poser.getCOMPos()
        state_vector.com_pos = com.tolist()
        state_vector.l_foot_pos = self.fwd_poser.getLinkPose("left_ankle_roll_link").tolist()
        state_vector.r_foot_pos = self.fwd_poser.getLinkPose("right_ankle_roll_link").tolist()

        dt = state_vector.time - self.prev_time

        if dt != 0:
            com_vel = (com - self.prev_com) / dt
            com_acc = (com_vel - self.prev_com_vel) / dt
            self.prev_com_vel = com_vel.copy()
            self.com_vel_filt.update(com_vel)
            self.com_acc_filt.update(com_acc)
            state_vector.com_vel = self.com_vel_filt.get().tolist()
            state_vector.com_acc = self.com_acc_filt.get().tolist()
            #state_vector.com_vel = com_vel
            #state_vector.com_acc = com_acc
        else:
            state_vector.com_vel = np.zeros([3]).tolist()
            state_vector.com_acc = np.zeros([3]).tolist()
        self.prev_com = com.copy()
        self.prev_time = state_vector.time

        return state_vector
"""

"""
if __name__ == "__main__":
    import time
    joint_list, joint_list_movable, leg_joints = makeJointList()
    urdf_config_path = os.getcwd()[:-7] + "urdf/g1_meshless.urdf"
    poser = BipedalPoser(urdf_config_path, joint_list, leg_joints, "left_ankle_roll_link",
                                      "right_ankle_roll_link")
    dmodel = poser.dualSupportDModel()
    q = poser.x[0 : 7 + len(leg_joints)]
    mat = pin.jacobianCenterOfMass(poser.model_r, poser.data_r, q)
    print(mat.shape)


    mat2 = pin.computeFrameJacobian(poser.model_r, poser.data_r, q, poser.model_r.getFrameId("left_ankle_roll_link"))

    print(mat2.shape)

    print(mat2[:, 6])
    pin.forwardKinematics(poser.model_r, poser.data_r, q)
    pin.updateFramePlacements(poser.model_r, poser.data_r)
    pos_i = poser.data_r.oMf[poser.model_r.getFrameId("left_ankle_roll_link")].translation.copy()
    com_i = pin.centerOfMass(poser.model_r, poser.data_r)
    q[7] = 0.01
    pin.forwardKinematics(poser.model_r, poser.data_r, q)
    pin.updateFramePlacements(poser.model_r, poser.data_r)
    com_f = pin.centerOfMass(poser.model_r, poser.data_r)
    pos_f = poser.data_r.oMf[poser.model_r.getFrameId("left_ankle_roll_link")].translation.copy()
    emp_j = (com_f - com_i) / 0.01
    emp_p = (pos_f - pos_i) / 0.01
    print(emp_j)
    print(emp_p)
"""
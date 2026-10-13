"""Math helpers for filtering, quaternion operations, and cross-product matrices"""

import numpy as np


def exponential_filter(previous: np.ndarray, current: np.ndarray, alpha: float):
    """
    Reduce high freq noise by taking a weighted average between the previous value and the new measurement

    :param previous: (n,) Filtered signal value from the previous timestep
    :param current: (n,) New, raw measurement
    :param alpha: Weight placed on previous value
    :return: (n,) Updated, filtered signal
    """
    return alpha * previous + (1 - alpha) * current


def quat_conjugate(q: np.ndarray):
    """
    Compute conjugate of given quaternion

    :param q: (4,) Quaternion in form (w, x, y, z)
    :return: (4,) Conjugate
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(q1: np.ndarray, q2: np.ndarray):
    """
    Compute multiplication of 2 quaternions

    :param q1: (4,) Left quaternion in form (w, x, y, z)
    :param q2: (4,) Right quaternion in form (w, x, y, z)
    :return: (4,) q1 multiplied by q2
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_from_mat(r: np.ndarray):
    """
    Convert 3x3 rotation matrix to quaternion

    :param r: (3, 3) 3D rotation matrix
    :return: (4,) Quaternion representation of rotation matrix in form (w, x, y, z)
    """
    qw = np.sqrt(1.0 + np.trace(r)) / 2.0
    qx = (r[2, 1] - r[1, 2]) / (4.0 * qw)
    qy = (r[0, 2] - r[2, 0]) / (4.0 * qw)
    qz = (r[1, 0] - r[0, 1]) / (4.0 * qw)
    return np.array([qw, qx, qy, qz])


def skew(v: np.ndarray):
    """
    Converts a vector into its cross product matrix

    :param v: (3,) The vector v to be used in the expression v cross x
    :return: (3, 3) The matrix V where Vx = v cross x
    """
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])

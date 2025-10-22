import numpy as np

JOINT_LIST_COMPLETE = ["l_hip_pitch_joint", "l_hip_roll_joint", "l_hip_yaw_joint", "l_knee_joint", "l_foot_pitch_joint", "l_foot_roll_joint",
                       "r_hip_pitch_joint", "r_hip_roll_joint", "r_hip_yaw_joint", "r_knee_joint", "r_foot_pitch_joint", "r_foot_roll_joint",
                       "l_shoulder_pitch_joint", "l_shoulder_roll_joint", "l_elbow_joint",
                       "r_shoulder_pitch_joint", "r_shoulder_roll_joint", "r_elbow_joint"]

def fill_obs_dict(msg):
    obs = {}
    obs['time'] = msg.time
    name_list = list(msg.joint_name)
    # Map incoming joint names to indices
    idx_map = {n: i for i, n in enumerate(name_list)}
    in_pos = np.asarray(msg.joint_pos)
    in_vel = np.asarray(msg.joint_vel)
    # Reorder to JOINT_LIST_COMPLETE; fill missing with NaN
    ordered_pos = []
    ordered_vel = []
    for jn in JOINT_LIST_COMPLETE:
        if jn in idx_map:
            i = idx_map[jn]
            ordered_pos.append(in_pos[i])
            ordered_vel.append(in_vel[i])
        else:
            ordered_pos.append(np.nan)
            ordered_vel.append(np.nan)
    obs['angular_velocity'] = np.array(msg.ang_vel)
    obs['grav_vec'] = np.array(msg.grav_vec)
    obs['linear_acceleration'] = np.array(msg.lin_acc)
    obs['joint_position'] = np.array(ordered_pos)
    obs['joint_velocity'] = np.array(ordered_vel)
    obs['linear_velocity'] = np.array(msg.lin_vel)
    return obs

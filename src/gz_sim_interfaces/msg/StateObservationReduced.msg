# Ovservation message for Deep Reinforcement Learning policy network

#builtin_interfaces/Time sim_time  # Gazebo simulation time
float64 time

# Robot state information
string[] joint_name        # joint names
float64[] joint_pos        # Joint positions
float64[] joint_vel        # Joint velocities (qvel)
float64[3] ang_vel          # Angular velocity of pelvis from IMU
float64[3] grav_vec        # gravitation vector w.r.t current orientation
float64[3] lin_vel         # linear velocity
float64[3] lin_acc          # linear acc
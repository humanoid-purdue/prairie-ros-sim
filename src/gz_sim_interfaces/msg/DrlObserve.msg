# Ovservation message for Deep Reinforcement Learning policy network

builtin_interfaces/Time sim_time  # Gazebo simulation time

# Robot state information
float64[] position         # Joint positions
float64[] velocity         # Joint velocities (qvel)
float64[3] angvel          # Angular velocity
float64[3] vel             # Center of mass velocity

# Site positions (flattened)
float64[] prev_sites       # Previous site positions
float64[] current_sites    # Current site positions

# Control and reward-related variables
float64[] prev_action      # Previous action taken
float64 l_coeff            # Left cycle coefficient
float64 r_coeff            # Right cycle coefficient

# Time step or step count
#int32 steps

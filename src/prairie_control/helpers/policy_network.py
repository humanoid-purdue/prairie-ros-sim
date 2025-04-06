import numpy as np
import math
import time
import jax.numpy as jnp
import jax
from brax import math
from lstm import HIDDEN_SIZE, DEPTH
from brax.io import html, mjcf, model
import os

policy_path = os.path.join(
            get_package_share_directory('prairie_control'),
            "walk_policy")

OBS_SIZE = 334
ACT_SIZE = 24


def makeIFN():
    from brax.training.agents.ppo import networks as ppo_networks
    from lstm import make_ppo_networks
    import functools
    from brax.training.acme import running_statistics
    mpn = make_ppo_networks
    network_factory = functools.partial(
        mpn,
        policy_hidden_layer_sizes=(512, 256, 256, 128))
    # normalize = running_statistics.normalize
    normalize = lambda x, y: x
    obs_size = OBS_SIZE
    ppo_network = network_factory(
        obs_size, ACT_SIZE, preprocess_observations_fn=normalize
    )
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    return make_inference_fn

def tanh2Action(action: jnp.ndarray):
    pos_t = action[:ACT_SIZE//2]
    vel_t = action[ACT_SIZE//2:]
    vel_sp = vel_t * 10

    #pos_sp = ((pos_t + 1) * (top_limit - bottom_limit) / 2 + bottom_limit)
    pos_sp = pos_t * 1.0

    return jnp.concatenate([pos_sp, vel_sp])

def make_obs(carry, 
             prev_action,
             joint_pos, 
             joint_vel, 
             angvel, 
             grav_vec, 
             linvel,
             vel_target, angvel_target, phase, halt):
    # all values are numpy and need conversion except for lstm_hidden
    position = jnp.array(joint_pos)
    velocity = jnp.array(joint_vel)
    angvel = jnp.array(angvel)
    grav_vec = jnp.array(grav_vec)
    linvel = jnp.array(linvel)
    cmd = jnp.array([vel_target[0], vel_target[1], angvel_target[0], halt])
    phase_clock = jnp.array([jnp.sin(phase[0]), jnp.cos(phase[0]),
                             jnp.sin(phase[1]), jnp.cos(phase[1])])
    obs = jnp.concatenate([carry, linvel,
                           angvel, grav_vec, position, velocity, prev_action, phase_clock, cmd
                           ])
    return obs

class walk_policy():
    def __init__():
        make_inference_fn = makeIFN()
        saved_params = model.load_params('walk_policy17')
        self.inference_fn = make_inference_fn(saved_params)
        self.hidden = jnp.zeros([HIDDEN_SIZE * DEPTH * 2])
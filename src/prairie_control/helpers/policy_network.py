import numpy as np
import math
import time
import jax.numpy as jnp
import jax
from brax import math
from lstm import HIDDEN_SIZE, DEPTH
from brax.io import html, mjcf, model
import os
from ament_index_python.packages import get_package_share_directory
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
    home_pose = np.array([-0.698132,
                               0,
                                 0,
                                   1.22173,
                                     -0.523599,
                                       0,
                                        -0.698132,
                                        0,
                                        0,
                                        1.22173,
                                        -0.523599, 0])
    pos_t = action[:ACT_SIZE//2]
    vel_t = action[ACT_SIZE//2:]
    vel_sp = vel_t * 10

    #pos_sp = ((pos_t + 1) * (top_limit - bottom_limit) / 2 + bottom_limit)
    pos_sp = pos_t * 2.0 + home_pose

    return jnp.concatenate([pos_sp, vel_sp])


class walk_policy():
    def __init__(self, t = 0.0):
        make_inference_fn = makeIFN()
        saved_params = model.load_params(policy_path + '/walk_policy_acc')
        inference_fn = make_inference_fn(saved_params)
        self.jit_inference_fn = jax.jit(inference_fn)
        self.rng = jax.random.PRNGKey(0)
        self.hidden = jnp.zeros([HIDDEN_SIZE * DEPTH * 2])
        self.prev_action = jnp.zeros([ACT_SIZE])
        self.phase = np.array([0., np.pi])
        self.prev_t = t

    def reinit(self, t = 0.0):
        self.hidden = jnp.zeros([HIDDEN_SIZE * DEPTH * 2])
        self.prev_action = jnp.zeros([ACT_SIZE])
        self.phase = np.array([0., np.pi])
        self.prev_t = t

    def apply_net(self, 
                  joint_pos, 
                  joint_vel, 
                  angvel, 
                  grav_vec, 
                  lin_acc, 
                  vel_target, 
                  angvel_target, 
                  halt, t):
        
        dt = t - self.prev_t
        self.prev_t = t
        
        phase = 2 * np.pi * dt / 1.0
        self.phase += phase
        self.phase = np.mod(self.phase, jnp.pi * 2)

        if (halt == 1):
            self.phase = np.array([0., np.pi])


        position = jnp.array(joint_pos)
        velocity = jnp.array(joint_vel)
        angvel = jnp.array(angvel)
        grav_vec = jnp.array(grav_vec)
        lin_acc = jnp.array(lin_acc)
        lin_acc = lin_acc - grav_vec * 9.81
        cmd = jnp.array([vel_target[0], vel_target[1], angvel_target[0], halt])
        phase_clock = jnp.array([jnp.sin(self.phase[0]), jnp.cos(self.phase[0]),
                             jnp.sin(self.phase[1]), jnp.cos(self.phase[1])])
        #print("awfaefaw", linvel, angvel, grav_vec, position, velocity, phase_clock, cmd)
        obs = jnp.concatenate([self.hidden, lin_acc,
                           angvel, grav_vec, position, velocity, self.prev_action, phase_clock, cmd
                           ])
        act_rng, self.rng = jax.random.split(self.rng)
        y, _ = self.jit_inference_fn(obs, act_rng)
        raw_action = y[2 * HIDDEN_SIZE * DEPTH:]
        self.prev_action = raw_action
        act = tanh2Action(raw_action)
        self.hidden = y[:2 * HIDDEN_SIZE * DEPTH]
        return act[:12], act[12:]
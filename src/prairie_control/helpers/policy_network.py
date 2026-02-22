import numpy as np
import math
import time
import jax.numpy as jnp
import jax
from brax import math
from ts_networks import make_ppo_networks, make_inference_fn
#from lstm import HIDDEN_SIZE, DEPTH
from brax.io import html, mjcf, model
import os
from ament_index_python.packages import get_package_share_directory
from brax.training.acme import running_statistics
policy_path = os.path.join(
            get_package_share_directory('prairie_control'),
            "walk_policy")

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
                        -0.523599, 0,
                        0, 0.05, 0,
    0, -0.05, 0])

#kp = jnp.array([35., 25., 25., 35., 35., 25.,
#                        35., 25., 25., 35., 35., 25.,
#                        15., 15., 15.,
#                        15., 15., 15.])
kp = jnp.array([120., 70., 70., 120., 120., 70.,
                120., 70., 70., 120., 120., 70.,
                15., 15., 15.,
                15., 15., 15.])
kd = jnp.array([2., 1., 1., 2., 2., 1.,
                       2., 1., 1., 2., 2., 1.,
                       1., 1., 1.,
                       1., 1., 1.,])

a = 20.
tau_limit = jnp.array([2 * a, a ,a , 2 * a, 2 * a, a,
                      2 * a, a ,a , 2 * a, 2 * a, a,
                      a, a, a,
                      a, a, a])


network_factory_params = {
    "emb_dim":64,
    "max_len":50,
    "num_layers":1,
    "num_heads":8,
    "mlp_dim":256,
    "value_hidden_layer_sizes":(512, 256, 256, 128),
    "policy_obs_key": "history",
    "value_obs_key":"privileged_state"
}

def makeIFN():
    import functools
    network_factory = functools.partial(
        make_ppo_networks,
        **network_factory_params
    )
    # normalize = running_statistics.normalize
    #normalize = lambda x, y: x
    normalize = running_statistics.normalize
    obs_size = 67 * 50#env.observation_size
    ppo_network = network_factory(
        obs_size, 18, preprocess_observations_fn=normalize
    )
    make_inference_fn_ = make_inference_fn(ppo_network)
    return make_inference_fn_

def tanh2Action(action: jnp.ndarray):

    #pos_sp = ((pos_t + 1) * (top_limit - bottom_limit) / 2 + bottom_limit)
    pos_sp = action * 1.5 + home_pose

    return pos_sp

def scale_action(action, joint_pos, joint_vel):
    # Return closest action (joint positions) such that
    # torque = kp*(action - joint_pos) - kd*joint_vel lies within +/- tau_limit
    a = jnp.asarray(action)
    q = jnp.asarray(joint_pos)
    dq = jnp.asarray(joint_vel)

    # Compute feasible action interval per joint
    # x_min <= action <= x_max ensures |torque| <= tau_limit
    eps = 1e-8
    kp_safe = jnp.maximum(kp, eps)
    x_min = q + (-tau_limit + kd * dq) / kp_safe
    x_max = q + ( tau_limit + kd * dq) / kp_safe

    # Clamp to nearest feasible action
    a_clamped = jnp.clip(a, x_min, x_max)
    return a_clamped
    

class walk_policy():
    def __init__(self, t = 0.0):
        make_inference_fn = makeIFN()
        saved_params = model.load_params(policy_path + '/walk_policy_highgain')
        inference_fn = make_inference_fn(saved_params)
        self.jit_inference_fn = jax.jit(inference_fn)
        self.rng = jax.random.PRNGKey(0)
        self.prev_action = jnp.zeros([18])
        self.action_history = jnp.zeros([50, 67])
        self.phase = np.array([0., np.pi])
        self.prev_t = t

    def reinit(self, t = 0.0):
        self.action_history = jnp.zeros([50, 67])
        self.prev_action = jnp.zeros([18])
        self.phase = np.array([0., np.pi])
        self.prev_t = t

    def apply_net(self, 
                  joint_pos, 
                  joint_vel, 
                  angvel,  
                  lin_acc, 
                  cmd, 
                  t):
        
        dt = t - self.prev_t
        self.prev_t = t
        
        phase = 2 * np.pi * dt * 1.35
        self.phase += phase
        self.phase = np.mod(self.phase, jnp.pi * 2)


        position = jnp.array(joint_pos) - home_pose
        velocity = jnp.array(joint_vel)
        angvel = jnp.array(angvel)
        #grav_vec = jnp.array(grav_vec)
        acc = jnp.array(lin_acc)
        
        # If norm of command is small, halt phase
        if (jnp.linalg.norm(cmd) < 0.1):
            self.phase = np.array([0., np.pi])

        phase_clock = jnp.array([jnp.cos(self.phase[0]), jnp.cos(self.phase[1]),
                             jnp.sin(self.phase[0]), jnp.sin(self.phase[1])])
        #print("awfaefaw", linvel, angvel, grav_vec, position, velocity, phase_clock, cmd)
        obs = jnp.concatenate([angvel, acc,
                           cmd, position, velocity,  self.prev_action, phase_clock
                           ])
        self.action_history = jnp.roll(self.action_history, shift=1, axis=0)
        self.action_history = self.action_history.at[0, :].set(obs)

        act_rng, self.rng = jax.random.split(self.rng)

        net_obs = self.action_history.reshape(-1)
        net_obs = {"history": net_obs,}
        raw_action, _ = self.jit_inference_fn(net_obs, act_rng)
        self.prev_action = raw_action
        act = tanh2Action(raw_action)
        act = scale_action(act, joint_pos, joint_vel)
        return act
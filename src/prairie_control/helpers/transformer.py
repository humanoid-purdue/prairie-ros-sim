import dataclasses
import functools
from typing import Any, Callable, Literal, Mapping, Sequence, Tuple
import warnings

from brax.training import types
from brax.training.acme import running_statistics
from brax.training.spectral_norm import SNDense
from flax import linen
from flax import linen as nn
import jax
import jax.numpy as jnp
from brax.training.networks import FeedForwardNetwork, normalizer_select, _get_obs_state_size
from transformer_policy import TransformerPolicy, TransformerPolicyModuleWithStd

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]

def make_policy_network(
	param_size: int,
	obs_size: types.ObservationSize,
	preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
	emb_dim = 128,
	max_len = 500,
	num_layers = 8,
	num_heads = 8,
	mlp_dim = 256,
	kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
	obs_key: str = 'state',
	history_obs_key: str = 'history',
	distribution_type: Literal['normal', 'tanh_normal'] = 'tanh_normal',
	noise_std_type: Literal['scalar', 'log'] = 'scalar',
	init_noise_std: float = 1.0,
	state_dependent_std: bool = False,
) -> FeedForwardNetwork:
	"""Creates a policy network."""
	if distribution_type == 'tanh_normal':
		policy_module = TransformerPolicy(
			obs_dim=obs_size,
			action_dim=param_size,
			emb_dim=emb_dim,
			max_len=max_len,
			num_layers=num_layers,
			num_heads=num_heads,
			mlp_dim=mlp_dim,
			kernel_init=kernel_init
        )
	elif distribution_type == 'normal':
		policy_module = TransformerPolicyModuleWithStd(
			obs_dim=obs_size,
			action_dim=param_size,
			emb_dim=emb_dim,
			max_len=max_len,
			num_layers=num_layers,
			num_heads=num_heads,
			mlp_dim=mlp_dim,
			kernel_init=kernel_init,
			noise_std_type=noise_std_type,
            init_noise_std=init_noise_std,
            state_dependent_std=state_dependent_std,
		)
	else:
		raise ValueError(
			f'Unsupported distribution type: {distribution_type}. Must be one'
			' of "normal" or "tanh_normal".'
			)

	# Compute flat feature size per timestep statically to avoid tracer shape issues
	flat_obs_size = _get_obs_state_size(obs_size, obs_key)
	if flat_obs_size % max_len != 0:
		raise ValueError(f'observation size {flat_obs_size} must be divisible by max_len={max_len}')
	per_t_features = flat_obs_size // max_len

	def apply(processor_params, policy_params, obs):
		# Select and normalize observations
		obs_ = obs[obs_key]
		if isinstance(obs, Mapping):
			obs_norm = preprocess_observations_fn(
                obs_, normalizer_select(processor_params, obs_key)
            )
		else:
			obs_norm = preprocess_observations_fn(obs_, processor_params)

		# Reshape to (..., T, D) with only static sizes in the new shape
		# Uses obs_norm.shape for leading batch dims (static) and computed per_t_features
		new_shape = obs_norm.shape[:-1] + (max_len, per_t_features)
		hist_mat = jnp.reshape(obs_norm, new_shape)

		output =  policy_module.apply(policy_params, hist_mat)
		return output
	
	# Dummy inputs for initialization with concrete static shape
	dummy_obs = jnp.zeros((1, flat_obs_size))

	def init(key):
		dummy_obs_ = dummy_obs.reshape(1, max_len, per_t_features)
		policy_module_params = policy_module.init(key, dummy_obs_)
		return policy_module_params

	return FeedForwardNetwork(init=init, apply=apply)
from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from brax.training.networks import Param, LogParam


class SinusoidalPositionalEncoding(nn.Module):
	max_len: int
	features: int

	@nn.compact
	def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
		# x: [..., T, D]
		seq_len = self.max_len
		dtype = x.dtype
		position = jnp.arange(seq_len, dtype=dtype)[:, None]
		div_term = jnp.exp(
			jnp.arange(0, self.features, 2, dtype=dtype)
			* (-jnp.log(10000.0) / self.features)
		)
		pe = jnp.zeros((seq_len, self.features), dtype=dtype)
		pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
		pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
		# pe: [T, D] will broadcast across any leading batch dims
		return x + pe


class MLP(nn.Module):
	hidden_dim: int
	out_dim: int
	dropout_rate: float = 0.0

	@nn.compact
	def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
		x = nn.Dense(self.hidden_dim)(x)
		x = nn.gelu(x)
		x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
		x = nn.Dense(self.out_dim)(x)
		return x


class EncoderBlock(nn.Module):
	emb_dim: int
	num_heads: int
	mlp_dim: int
	dropout_rate: float = 0.0

	@nn.compact
	def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
		# Pre-LN Transformer Encoder block
		y = nn.LayerNorm()(x)
		y = nn.SelfAttention(
			num_heads=self.num_heads,
			qkv_features=self.emb_dim,
			out_features=self.emb_dim,
			use_bias=True,
			dropout_rate=self.dropout_rate,
		)(y, deterministic=not train)
		x = x + nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)

		y = nn.LayerNorm()(x)
		y = MLP(hidden_dim=self.mlp_dim, out_dim=self.emb_dim, dropout_rate=self.dropout_rate)(y, train=train)
		x = x + nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
		return x


class TransformerPolicy(nn.Module):
	"""Transformer encoder policy mapping a sequence of observations to an action.

	Inputs: obs_seq with shape [..., T, obs_dim]
	Outputs: action with shape [..., action_dim]
	"""
	obs_dim: int
	action_dim: int
	emb_dim: int
	max_len: int
	num_layers: int
	num_heads: int
	mlp_dim: int
	dropout_rate: float = 0.0
	use_cls_token: bool = True
	kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.lecun_uniform()

	@nn.compact
	def __call__(self, obs_seq: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
		# Project observations to embedding space
		x = nn.Dense(self.emb_dim)(obs_seq)  # [..., T, D]
		x = SinusoidalPositionalEncoding(max_len=self.max_len, features=self.emb_dim)(x)

		if self.use_cls_token:
			# Create a single learnable CLS token and broadcast to match batch dims
			cls = self.param('cls', nn.initializers.zeros, (self.emb_dim,))  # [D]
			cls = jnp.reshape(cls, (1, self.emb_dim))  # [1, D]
			batch_shape = x.shape[:-2]
			cls_tiled = jnp.broadcast_to(cls, batch_shape + (1, self.emb_dim))  # [..., 1, D]
			x = jnp.concatenate([cls_tiled, x], axis=-2)  # [..., 1+T, D]

		for _ in range(self.num_layers):
			x = EncoderBlock(
				emb_dim=self.emb_dim,
				num_heads=self.num_heads,
				mlp_dim=self.mlp_dim,
				dropout_rate=self.dropout_rate,
			)(x, train=train)

		# Pooling to fixed-size representation
		if self.use_cls_token:
			h = x[..., 0, :]  # [..., D]
		else:
			h = jnp.mean(x, axis=-2)  # [..., D]

		h = nn.LayerNorm()(h)
		y = nn.Dense(self.mlp_dim, kernel_init = self.kernel_init)(h)
		y = nn.gelu(y)
		y = nn.Dense(self.mlp_dim, kernel_init = self.kernel_init)(y)
		y = nn.gelu(y)
		action = nn.Dense(self.action_dim, kernel_init = self.kernel_init)(y)
		return action

class TransformerPolicyModuleWithStd(nn.Module):
	"""Transformer encoder policy mapping a sequence of observations to an action.

	Inputs: obs_seq with shape [..., T, obs_dim]
	Outputs: action with shape [..., action_dim]
	"""
	obs_dim: int
	action_dim: int
	emb_dim: int
	max_len: int
	num_layers: int
	num_heads: int
	mlp_dim: int
	noise_std_type: str = 'scalar'  # 'scalar' or 'log'
	init_noise_std: float = 1.0
	state_dependent_std: bool = False
	dropout_rate: float = 0.0
	kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.lecun_uniform()
	use_cls_token: bool = True

	@nn.compact
	def __call__(self, obs_seq: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
		# Project observations to embedding space
		x = nn.Dense(self.emb_dim)(obs_seq)  # [..., T, D]
		x = SinusoidalPositionalEncoding(max_len=self.max_len, features=self.emb_dim)(x)

		if self.use_cls_token:
			cls = self.param('cls', nn.initializers.zeros, (self.emb_dim,))  # [D]
			cls = jnp.reshape(cls, (1, self.emb_dim))  # [1, D]
			batch_shape = x.shape[:-2]
			cls_tiled = jnp.broadcast_to(cls, batch_shape + (1, self.emb_dim))  # [..., 1, D]
			x = jnp.concatenate([cls_tiled, x], axis=-2)  # [..., 1+T, D]

		for _ in range(self.num_layers):
			x = EncoderBlock(
				emb_dim=self.emb_dim,
				num_heads=self.num_heads,
				mlp_dim=self.mlp_dim,
				dropout_rate=self.dropout_rate,
			)(x, train=train)

		# Pooling to fixed-size representation
		if self.use_cls_token:
			h = x[..., 0, :]  # [..., D]
		else:
			h = jnp.mean(x, axis=-2)  # [..., D]

		h = nn.LayerNorm()(h)
		mean_params = nn.Dense(self.action_dim, kernel_init=self.kernel_init)(h)
		if self.state_dependent_std:
			log_std_output = nn.Dense(self.action_dim, kernel_init=self.kernel_init)(h)
			if self.noise_std_type == 'log':
				std_params = jnp.exp(log_std_output)
			else:
				std_params = log_std_output
		else:
			if self.noise_std_type == 'scalar':
				std_module = Param(
					self.init_noise_std, size=self.action_dim, name='std_param')
			else:
				std_module = LogParam(
                    self.init_noise_std, size=self.action_dim, name='std_logparam'
                )
			std_params = std_module()
		return mean_params, jnp.broadcast_to(std_params, mean_params.shape)
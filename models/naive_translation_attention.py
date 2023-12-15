from functools import partial
from typing import Optional, Tuple
import jax
from jax import numpy as jnp, random

class NaiveAttentionModel:
  def __init__(self, rng: jnp.ndarray, N: int, d: int, lr: float = 0.01):
    rngs = random.split(rng, 10)
    self.keys = random.normal(rngs[0], shape = (N, d))
    self.values = random.normal(rngs[1], shape = (N, d))
    self.beta = jnp.array(1.0)
    self.N = N
    self.d = d

    self.params = [self.keys, self.values, self.beta]
    self.lr = lr

  def convolve(self, x: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
    circulant_key = jnp.stack([jnp.roll(key, i) for i in range(self.d)])
    return jnp.sum(circulant_key * x[None, :], axis = -1)

  def weights(self, x: jnp.ndarray, keys: jnp.ndarray, beta: float) -> jnp.ndarray:
    convolutions = jax.vmap(self.convolve, in_axes = [None, 0])(x, keys)
    return jax.nn.softmax(beta * convolutions)

  def combine(self, weight: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
    circulant_value = jnp.stack([jnp.roll(value, i) for i in range(self.d)])
    return weight[:, None] * circulant_value

  def call_fn(self, x: jnp.ndarray, keys: jnp.ndarray, values: jnp.ndarray, beta: float) -> jnp.ndarray:
    return jax.vmap(self.combine, (0, 0))(self.weights(x, keys, beta), values).sum(axis = [0, 1])

  def batched_call_fn(self, xs: jnp.ndarray, keys: jnp.ndarray, values: jnp.ndarray, beta: float) -> jnp.ndarray:
    return jax.vmap(self.call_fn, in_axes = [0, None, None, None])(xs, keys, values, beta)

  def loss(self, pred_y: jnp.ndarray, true_y: jnp.ndarray) -> float:
    return jnp.mean((pred_y - true_y)**2) / 2

  @partial(jax.jit, static_argnums = (0))
  def compute_loss(self, keys: jnp.ndarray, values: jnp.ndarray, beta: float, x: jnp.ndarray, y: jnp.ndarray) -> float:
    pred_y = self.batched_call_fn(x, keys, values, beta)
    return self.loss(pred_y, y)
  
  def train_step(self, x: jnp.ndarray, y: jnp.ndarray):
    loss, gradients = jax.value_and_grad(self.compute_loss, [0, 1, 2])(self.keys, self.values, self.beta, x, y)
    [self.keys, self.values, self.beta] = [p - self.lr * grad_p for p, grad_p in zip(self.params, gradients)]
    return loss



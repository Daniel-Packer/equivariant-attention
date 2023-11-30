from typing import Optional
import jax
import jax.numpy as jnp


class TranslationAttention:
    def __init__(self, rng: jax.Array, d: int, n: int, lr: float = 0.01):
        rngs = jax.random.split(rng, 2)
        self.keys = jax.random.normal(key=rngs[0], shape=[n, d])
        self.values = jax.random.normal(key=rngs[1], shape=[n, d])
        self.fft_keys = jnp.fft.fft(self.keys)
        self.fft_values = jnp.fft.fft(self.values)
        self.lr = lr

    def set(self, **kwargs):
        for name, val in kwargs.items():
            match name:
                case "keys":
                    self.keys = val
                    self.fft_keys = jnp.fft.fft(self.keys)
                case "values":
                    self.values = val
                    self.fft_values = jnp.fft.fft(self.values)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._call_fn(x, self.fft_keys, self.fft_values)

    def train(self, x: jnp.ndarray, y: jnp.ndarray, n_epochs: int, lr: Optional[float]):
        self.lr = self.lr if lr is None else lr
        try:
            assert x.shape == y.shape
        except:
            raise ValueError("x and y are not the same shape")


@jax.jit
def update(params: list[jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray, lr: float):
    grads = jax.grad(loss)(params, x, y)
    return [p - lr * grad_p for p, grad_p in zip(params, grads)]


def loss(params: list[jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray):
    pred_y = batched_call_fn(x, *params)
    return jnp.mean(jnp.square(jnp.abs(pred_y - y)))


@jax.jit
def batched_call_fn(
    x: jnp.ndarray, fft_keys: jnp.ndarray, fft_values: jnp.ndarray, beta: float
):
    return jax.vmap(call_fn, in_axes=[0, None, None, None])(
        x, fft_keys, fft_values, beta
    )


def call_fn(
    x: jnp.ndarray, fft_keys: jnp.ndarray, fft_values: jnp.ndarray, beta: float
):
    fft_x = jnp.fft.fft(x)
    x_corr_K = jnp.fft.ifft(fft_x[None, :] * fft_keys[:, :])
    weights = jnp.flip(jax.nn.softmax(jnp.real(beta * x_corr_K), axis=[0, 1]), axis=-1)
    return jnp.flip(jnp.sum(jnp.fft.ifft(jnp.fft.fft(weights) * fft_values[:, :]), axis = 0))


def score(x: jnp.ndarray, fft_keys: jnp.ndarray, fft_values: jnp.ndarray, beta: float):
    fft_x = jnp.fft.fft(x)
    x_corr_K = jnp.roll(jnp.fft.ifft(fft_x[None, :] * fft_keys[:, :]), 1, axis=-1)
    weights = jax.nn.softmax(jnp.real(beta * x_corr_K), axis=[0, 1])
    return weights


def loss_ungrouped(
    x: jnp.ndarray,
    y: jnp.ndarray,
    fft_keys: jnp.ndarray,
    fft_values: jnp.ndarray,
    beta: float,
) -> float:
    pred_y = batched_call_fn(x, fft_keys, fft_values, beta)
    return jnp.mean(jnp.square(jnp.abs(pred_y - y)))

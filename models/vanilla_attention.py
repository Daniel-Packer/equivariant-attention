import jax
import jax.numpy as jnp
from typing import Callable
from tqdm.notebook import tqdm


@jax.jit
def update(params: list[jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray, lr: float):
    grads = jax.grad(loss)(params, x, y)
    return [p - lr * grad_p for p, grad_p in zip(params, grads)]

@jax.jit
def batched_loss(
    x: jnp.ndarray,
    y: jnp.ndarray,
    keys_hist: jnp.ndarray,
    values_hist: jnp.ndarray,
    betas_hist: jnp.ndarray,
):
    return jax.vmap(loss_ungrouped, (None, None, 0, 0, 0))(
        x, y, keys_hist, values_hist, betas_hist,
    )

def loss_ungrouped(x: jnp.ndarray, y: jnp.ndarray, keys: jnp.ndarray, values: jnp.ndarray, beta: jnp.ndarray):
    pred_y = batched_call_fn(x, keys, values, beta)
    return jnp.mean(jnp.square(pred_y - y))

def loss(params: list[jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray):
    pred_y = batched_call_fn(x, *params)
    return jnp.mean(jnp.square(pred_y - y))


@jax.jit
def batched_call_fn(x: jnp.ndarray, keys: jnp.ndarray, values: jnp.ndarray, beta: jnp.ndarray):
    return jax.vmap(call_fn, in_axes=[0, None, None, None])(x, keys, values, beta)


def call_fn(x: jnp.ndarray, keys: jnp.ndarray, values: jnp.ndarray, beta: jnp.ndarray):
    weights = jax.nn.softmax(keys @ x * beta)
    return jnp.sum((weights[:, None] * values), axis=0)


def train(
    rng: jax.Array,
    X_train_encoded: jnp.ndarray,
    n_keys: int,
    n_epochs: int,
    lr: float,
    Y_train: jnp.ndarray,
    verbose: bool = False,
    init_beta: float = 1.0,
):
    rngs = jax.random.split(rng, 4)

    keys = jax.random.choice(rngs[0], X_train_encoded, shape = [n_keys]) + (jax.random.normal(rngs[1], shape=[n_keys, X_train_encoded.shape[-1]]) * 1e-3)
    values = jax.random.choice(rngs[0], Y_train, shape = [n_keys]) + (jax.random.normal(rngs[1], shape=[n_keys, Y_train.shape[-1]]) * 1e-3)
    beta = init_beta

    keys_hist = jnp.zeros([n_epochs, *keys.shape])
    values_hist = jnp.zeros([n_epochs, *values.shape])
    beta_hist = jnp.zeros([n_epochs])

    pbar = tqdm(range(n_epochs)) if verbose else range(n_epochs)

    for i in pbar:
        keys_hist = keys_hist.at[i].set(keys)
        values_hist = values_hist.at[i].set(values)
        beta_hist = beta_hist.at[i].set(beta)
        keys, values, beta = update([keys, values, beta], X_train_encoded, Y_train, lr=lr)

    return keys_hist, values_hist, beta_hist


def plot_predictions(encoding, keys_hist, values_hist, cmap, ax, X_test=None):
    n_epochs = keys_hist.shape[0]
    for i in range(0, n_epochs, int(n_epochs / 50)):
        keys, values = keys_hist[i], values_hist[i]

        X_test = (
            jnp.linspace(-1.5, 1.5, num=1_000)[:, None] if X_test is None else X_test
        )
        X_test_pred = batched_call_fn(encoding(X_test), keys, values)

        ax.plot(X_test, X_test_pred, color=cmap(i / n_epochs))

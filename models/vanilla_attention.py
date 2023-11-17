import jax
import jax.numpy as jnp
from typing import Callable
from tqdm.notebook import tqdm


@jax.jit
def update(params: list[jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray, lr: float):
    grads = jax.grad(loss)(params, x, y)
    return [p - lr * grad_p for p, grad_p in zip(params, grads)]


def loss(params: list[jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray):
    pred_y = batched_call_fn(x, *params)
    return jnp.mean(jnp.square(pred_y - y))


@jax.jit
def batched_call_fn(x: jnp.ndarray, keys: jnp.ndarray, values: jnp.ndarray):
    return jax.vmap(call_fn, in_axes=[0, None, None])(x, keys, values)


def call_fn(x: jnp.ndarray, keys: jnp.ndarray, values: jnp.ndarray):
    weights = jax.nn.softmax(x @ keys.T)
    return jnp.sum((weights[:, None] * values), axis=0)


def train(
    rng: jax.Array,
    X_train: jnp.ndarray,
    encoding: Callable[[jnp.ndarray], jnp.ndarray],
    n_keys: int,
    n_epochs: int,
    lr: float,
    f: Callable[[jnp.ndarray], jnp.ndarray],
    verbose: bool = False,
):
    rngs = jax.random.split(rng, 4)
    f_X_train = f(X_train)

    # keys = jax.random.normal(key=rngs[2], shape=(n_keys, 2))
    keys = jax.random.choice(rngs[2], encoding(X_train), shape=[n_keys], axis=0)
    values = jax.random.normal(key=rngs[3], shape=(n_keys, 1))

    X_train_enc = encoding(X_train)

    keys_hist = jnp.zeros([n_epochs, *keys.shape])
    values_hist = jnp.zeros([n_epochs, *values.shape])

    pbar = tqdm(range(n_epochs)) if verbose else range(n_epochs)

    for i in pbar:
        keys_hist = keys_hist.at[i].set(keys)
        values_hist = values_hist.at[i].set(values)
        keys, values = update([keys, values], X_train_enc, f_X_train, lr=lr)

    return keys_hist, values_hist


def plot_predictions(encoding, keys_hist, values_hist, cmap, ax, X_test=None):
    n_epochs = keys_hist.shape[0]
    for i in range(0, n_epochs, int(n_epochs / 50)):
        keys, values = keys_hist[i], values_hist[i]

        X_test = (
            jnp.linspace(-1.5, 1.5, num=1_000)[:, None] if X_test is None else X_test
        )
        X_test_pred = batched_call_fn(encoding(X_test), keys, values)

        ax.plot(X_test, X_test_pred, color=cmap(i / n_epochs))

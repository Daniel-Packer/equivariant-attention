from typing import Callable, Optional
import jax.numpy as jnp
import jax
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
from matplotlib import collections

# For SO(2) equivariance acting on R^2:


def group_samples(n_samples: int, extra_dims: int) -> jnp.ndarray:
    thetas = jnp.linspace(0, 2 * jnp.pi, num=n_samples, endpoint=False)
    rotation_mats = jnp.stack(
        [
            jnp.stack([jnp.cos(thetas), jnp.sin(thetas)], axis=-1),
            jnp.stack([-jnp.sin(thetas), jnp.cos(thetas)], axis=-1),
        ],
        axis=-1,
    )
    return jax.vmap(
        jax.scipy.linalg.block_diag, (0, *[None for _ in range(extra_dims)])
    )(rotation_mats, *[1 for _ in range(extra_dims)])


@jax.jit
def update(
    params: list[jnp.ndarray],
    x: jnp.ndarray,
    key_reps: jnp.ndarray,
    value_reps: jnp.ndarray,
    y: jnp.ndarray,
    lr: float,
):
    grads = jax.grad(loss)(params, x, key_reps, value_reps, y)
    return [p - lr * grad_p for p, grad_p in zip(params, grads)]


def loss(
    params: list[jnp.ndarray],
    x: jnp.ndarray,
    key_reps: jnp.ndarray,
    value_reps: jnp.ndarray,
    y: jnp.ndarray,
):
    pred_y = batched_call_fn(x, params[0], key_reps, params[1], value_reps, params[2])
    return jnp.mean(jnp.square(pred_y - y))


@jax.jit
def batched_loss(
    x: jnp.ndarray,
    y: jnp.ndarray,
    keys_hist: jnp.ndarray,
    key_reps: jnp.ndarray,
    values_hist: jnp.ndarray,
    value_reps: jnp.ndarray,
    betas_hist: jnp.ndarray,
):
    return jax.vmap(loss_ungrouped, (None, None, 0, None, 0, None, 0))(
        x, y, keys_hist, key_reps, values_hist, value_reps, betas_hist
    )


def loss_ungrouped(
    x: jnp.ndarray,
    y: jnp.ndarray,
    keys: jnp.ndarray,
    key_reps: jnp.ndarray,
    values: jnp.ndarray,
    value_reps: jnp.ndarray,
    beta: float,
) -> float:
    pred_y = batched_call_fn(x, keys, key_reps, values, value_reps, beta)
    return jnp.mean(jnp.square(pred_y - y))


@jax.jit
def batched_call_fn(
    x: jnp.ndarray,
    keys: jnp.ndarray,
    key_reps: jnp.ndarray,
    values: jnp.ndarray,
    value_reps: jnp.ndarray,
    beta: float,
):
    return jax.vmap(call_fn, in_axes=[0, None, None, None, None, None])(
        x,
        keys,
        key_reps,
        values,
        value_reps,
        beta,
    )


def scores(
    x: jnp.ndarray,
    keys: jnp.ndarray,
    key_reps: jnp.ndarray,
    beta: jnp.ndarray,
) -> jnp.ndarray:
    g_keys = jnp.sum(key_reps[None, :, :, :] * keys[:, None, None, :], axis=-1)
    return jax.nn.softmax(g_keys @ x * beta, axis=[0, 1])


def call_fn(
    x: jnp.ndarray,
    keys: jnp.ndarray,
    key_reps: jnp.ndarray,
    values: jnp.ndarray,
    value_reps: jnp.ndarray,
    beta: jnp.ndarray,
) -> jnp.ndarray:
    g_keys = jnp.sum(key_reps[None, :, :, :] * keys[:, None, None, :], axis=-1)
    g_values = jnp.sum(value_reps[None, :, :, :] * values[:, None, None, :], axis=-1)

    scores = jax.nn.softmax(g_keys @ x * beta, axis=[0, 1])

    return jnp.sum(scores[:, :, None] * g_values, axis=[0, 1])


def train(
    rng: jax.Array,
    X_train: jnp.ndarray,
    encoding: Callable[[jnp.ndarray], jnp.ndarray],
    n_keys: int,
    n_group_samples: int,
    n_epochs: int,
    lr: float,
    Y_train: jnp.ndarray,
    verbose: bool = False,
    init_keys: Optional[jnp.ndarray] = None,
    init_values: Optional[jnp.ndarray] = None,
    init_beta: float = 1.0,
):
    rngs = jax.random.split(rng, 4)

    X_train_enc = encoding(X_train)

    # keys = (
    #     jax.random.normal(key=rngs[2], shape=(n_keys, X_train_enc.shape[-1]))
    #     if init_keys is None
    #     else init_keys
    # )
    # values = (
    #     jax.random.normal(key=rngs[3], shape=(n_keys, f_X_train.shape[-1]))
    #     if init_values is None
    #     else init_values
    # )
    beta = init_beta
    keys = jax.random.choice(rngs[0], X_train_enc, shape = [n_keys]) + (jax.random.normal(rngs[1], shape=[n_keys, X_train_enc.shape[-1]]) * 1e-6)
    values = jax.random.choice(rngs[0], Y_train, shape = [n_keys]) + (jax.random.normal(rngs[1], shape=[n_keys, Y_train.shape[-1]]) * 1e-6)

    key_reps = group_samples(n_group_samples, X_train_enc.shape[-1] - 2)
    value_reps = group_samples(n_group_samples, Y_train.shape[-1] - 2)

    keys_hist = jnp.zeros([n_epochs, *keys.shape])
    values_hist = jnp.zeros([n_epochs, *values.shape])
    betas_hist = jnp.zeros([n_epochs])

    pbar = tqdm(range(n_epochs)) if verbose else range(n_epochs)

    for i in pbar:
        keys_hist = keys_hist.at[i].set(keys)
        values_hist = values_hist.at[i].set(values)
        betas_hist = betas_hist.at[i].set(beta)
        keys, values, beta = update(
            [keys, values, beta], X_train_enc, key_reps, value_reps, Y_train, lr=lr
        )

    return keys_hist, values_hist, key_reps, value_reps, betas_hist


def plot_predictions(X_test, X_test_pred, Y_test):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    n_samples = X_test.shape[0]

    segments = [[X_test[i], X_test_pred[i]] for i in range(n_samples)]
    lines = collections.LineCollection(segments, zorder=-1, alpha=0.4)
    axs[0].add_collection(lines)
    axs[0].scatter(X_test[:, 0], X_test[:, 1], label="original data")
    axs[0].scatter(
        X_test_pred[:, 0], X_test_pred[:, 1], label="predicted data", marker="x"
    )
    axs[0].legend()
    axs[0].set(aspect="equal")

    segments = [[X_test[i], Y_test[i]] for i in range(n_samples)]
    lines = collections.LineCollection(segments, zorder=-1, alpha=0.4)
    axs[1].add_collection(lines)
    axs[1].scatter(X_test[:, 0], X_test[:, 1], label="original data")
    axs[1].scatter(
        Y_test[:, 0], Y_test[:, 1], label="true data", marker="D", color="C2"
    )
    axs[1].legend()
    axs[1].set(aspect="equal")

    segments = [[X_test_pred[i], Y_test[i]] for i in range(n_samples)]
    lines = collections.LineCollection(segments, zorder=-1, alpha=0.4, color="red")
    axs[2].add_collection(lines)
    axs[2].scatter(
        Y_test[:, 0], Y_test[:, 1], label="true data", marker="D", color="C2"
    )
    axs[2].scatter(
        X_test_pred[:, 0],
        X_test_pred[:, 1],
        label="predicted data",
        color="C1",
        marker="x",
    )
    axs[2].legend()
    axs[2].set(aspect="equal")

    return fig, axs

from typing import Callable
import jax
import jax.numpy as jnp


def check_equivariance(
    f: Callable[[jnp.ndarray], jnp.ndarray],
    x: jnp.ndarray,
    domain_reps: jnp.ndarray,
    codomain_reps: jnp.ndarray,
):
    orbit_x = jnp.sum(domain_reps * x[None, None, :], axis=-1)
    f_orbit_x = jax.vmap(f)(orbit_x)
    f_x = f(x)
    orbit_f_x = jnp.sum(codomain_reps * f_x[None, None, :], axis=-1)
    return jnp.sum(jnp.square(f_orbit_x - orbit_f_x))

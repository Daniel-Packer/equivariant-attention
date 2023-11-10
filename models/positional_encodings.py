import jax.numpy as jnp
from typing import Literal

normalize = lambda v: v / jnp.linalg.norm(v)

def positional_encoding(v: jnp.ndarray, method=Literal["trig", "inv_proj", "fold", "stereographic"]):
    if len(v.shape) == 1:
        v = v[None, :]

    match method:
        case "trig":
            norm_v = jnp.linalg.norm(v, axis=-1)
            return jnp.concatenate(
                [
                    v / norm_v[:, None],
                    jnp.stack([jnp.cos(norm_v), jnp.sin(norm_v)], axis=-1),
                ],
                axis=-1,
            )
        case "inv_proj":
            norm_v = jnp.linalg.norm(v, axis=-1)
            return jnp.concatenate([v, jnp.sqrt(1 - jnp.square(norm_v))[:, None]], axis=-1)

        case "fold":
            v_lift = jnp.concatenate([v, jnp.ones(v.shape[0])[:, None]], axis=-1)
            return v_lift / jnp.linalg.norm(v_lift, axis=-1)[:, None]
          
        case "stereographic":
            s_sq = jnp.sum(v * v, axis = -1)
            return jnp.concatenate([
              2 * v,
              (jnp.full(v.shape[0], s_sq - 1))[:, None]
            ], axis = -1) / (s_sq + 1)[:, None]
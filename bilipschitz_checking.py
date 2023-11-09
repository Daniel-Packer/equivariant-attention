import numpy as np
from typing import Literal

rng = np.random.default_rng(seed=123)

normalize = lambda v: v / np.linalg.norm(v)


def ball_to_sphere(v, method=Literal["trig", "inv_proj", "fold", "stereographic"]):
    if len(v.shape) == 1:
        v = v[None, :]

    match method:
        case "trig":
            norm_v = np.linalg.norm(v, axis=-1)
            return np.concatenate(
                [
                    v / norm_v[:, None],
                    np.stack([np.cos(norm_v), np.sin(norm_v)], axis=-1),
                ],
                axis=-1,
            )
        case "inv_proj":
            norm_v = np.linalg.norm(v, axis=-1)
            return np.concatenate([v, np.sqrt(1 - np.square(norm_v))[:, None]], axis=-1)

        case "fold":
            v_lift = np.concatenate([v, np.ones(v.shape[0])[:, None]], axis=-1)
            return v_lift / np.linalg.norm(v_lift, axis=-1)[:, None]
          
        case "stereographic":
            s_sq = np.sum(v * v, axis = -1)
            return np.concatenate([
              2 * v,
              (np.full(v.shape[0], s_sq - 1))[:, None]
            ], axis = -1) / (s_sq + 1)[:, None]


def uniform_ball_samples(N: int, d: int, r: float):
    inputs = rng.normal(0.0, 1.0, size=[N, d])
    inputs /= np.linalg.norm(inputs, axis=-1)[:, None]
    radii = np.power(rng.uniform(0.0, np.power(r, d), size=[N]), 1 / d)
    return inputs * radii[:, None]


def pairwise_distances(pts: np.ndarray):
    assert len(pts.shape) == 2
    return np.linalg.norm(pts[None, :, :] - pts[:, None, :], axis=-1)


def get_distances(N: int, d: int, r: float, method: str = "trig"):
    inputs = uniform_ball_samples(N, d, r)
    outputs = ball_to_sphere(inputs, method)
    return pairwise_distances(inputs).flatten(), pairwise_distances(outputs).flatten()


def get_bilipschitz_constant(N: int, d: int, r: float, method: str = "trig"):
    ratios = np.divide(*get_distances(N, d, r, method=method))
    ratios = ratios[~np.isnan(ratios)]
    return np.max(ratios) / np.min(ratios)


def estimated_bilipschitz_constant(N: int, d: int, r: float, *, n_trials=100):
    return np.mean([get_bilipschitz_constant(N, d, r) for _ in range(n_trials)])
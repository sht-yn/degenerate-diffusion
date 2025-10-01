"""JAX-native parameter estimation utilities."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import infer
from numpyro.distributions import transforms

Bounds = Sequence[tuple[float | None, float | None]]
Array1D = jnp.ndarray


def _normalize_bounds(search_bounds: Bounds) -> jnp.ndarray:
    if not isinstance(search_bounds, Sequence):  # type: ignore[arg-type]
        msg = "search_bounds must be a sequence of (low, high) tuples."
        raise TypeError(msg)

    bounds_list: list[tuple[float, float]] = []
    for idx, bound in enumerate(search_bounds):
        if not isinstance(bound, tuple) or len(bound) != 2:
            msg = f"Bound at index {idx} must be a (low, high) tuple."
            raise ValueError(msg)
        low, high = bound
        low_f = -math.inf if low is None else float(low)
        high_f = math.inf if high is None else float(high)
        if low_f > high_f:
            msg = f"Lower bound {low_f} exceeds upper bound {high_f} at index {idx}."
            raise ValueError(msg)
        bounds_list.append((low_f, high_f))
    return jnp.asarray(bounds_list)


def _clip(theta: Array1D, bounds: jnp.ndarray) -> Array1D:
    lows = bounds[:, 0]
    highs = bounds[:, 1]
    return jnp.clip(theta, lows, highs)


def m_estimate(
    objective_function: Callable[[Array1D], float],
    search_bounds: Bounds,
    initial_guess: Sequence[float],
    *,
    learning_rate: float = 1e-3,
    max_iters: int = 1000,
    tol: float = 1e-6,
    log_interval: int | None = None,
) -> np.ndarray:
    """Gradient-ascent M-estimation implemented with JAX.

    The ``learning_rate`` の既定値を ``1e-3`` に下げ、必要に応じて ``log_interval``（反復回数）で
    勾配ノルムと現在の θ をログ表示できるようにした。
    """
    bounds_raw = _normalize_bounds(search_bounds)
    theta0 = jnp.asarray(initial_guess)

    if theta0.shape != (bounds_raw.shape[0],):
        msg = f"initial_guess must have shape ({bounds_raw.shape[0]},)."
        raise ValueError(msg)

    bounds = bounds_raw.astype(theta0.dtype)

    @jax.jit
    def step(theta: Array1D) -> tuple[Array1D, jnp.ndarray]:
        grad = jax.grad(objective_function)(theta)
        theta_next = theta + learning_rate * grad
        theta_next = _clip(theta_next, bounds)
        grad_norm = jnp.linalg.norm(grad)
        return theta_next, grad_norm

    theta_opt = theta0
    grad_norm_value = float("inf")
    for iteration in range(1, max_iters + 1):
        theta_opt, grad_norm = step(theta_opt)
        grad_norm_value = float(grad_norm)
        if log_interval and iteration % log_interval == 0:
            print(
                f"[m_estimate] iter={iteration} grad_norm={grad_norm_value:.3e} theta={theta_opt}"
            )
        if grad_norm_value <= tol:
            break

    return np.asarray(theta_opt)


def one_step_estimate(
    objective_function: Callable[[Array1D], float],
    search_bounds: Bounds,
    initial_estimator: Sequence[float],
) -> np.ndarray:
    """Single Newton step using JAX gradients and Hessians."""
    bounds_raw = _normalize_bounds(search_bounds)
    theta0 = jnp.asarray(initial_estimator)

    if theta0.shape != (bounds_raw.shape[0],):
        msg = f"initial_estimator must have shape ({bounds_raw.shape[0]},)."
        raise ValueError(msg)

    bounds = bounds_raw.astype(theta0.dtype)

    grad = jax.grad(objective_function)(theta0)
    hessian = jax.hessian(objective_function)(theta0)

    eye = jnp.eye(theta0.shape[0])
    hessian_safe = hessian + 1e-6 * eye
    delta = jnp.linalg.solve(hessian_safe, grad)
    theta_new = theta0 - delta
    theta_new = _clip(theta_new, bounds)
    return np.asarray(theta_new)


def newton_solve(
    objective_function: Callable[[Array1D], float],
    search_bounds: Bounds,
    initial_guess: Sequence[float],
    *,
    max_iters: int = 50,
    tol: float = 1e-6,
    damping: float = 1.0,
    log_interval: int | None = None,
) -> np.ndarray:
    """多次元ニュートン法で推定方程式 ``∇θ V(θ) = 0`` を解く。

    ``damping`` を 1 未満にすると過大ステップを抑えられる。解が境界の外に出そうな場合でも
    ``_clip`` で常に ``search_bounds`` 内に戻す。
    """
    bounds_raw = _normalize_bounds(search_bounds)
    theta = jnp.asarray(initial_guess)
    if theta.shape != (bounds_raw.shape[0],):
        msg = f"initial_guess must have shape ({bounds_raw.shape[0]},)."
        raise ValueError(msg)

    bounds = bounds_raw.astype(theta.dtype)

    def grad_and_hessian(theta_val: Array1D) -> tuple[Array1D, jnp.ndarray]:
        grad_val = jax.grad(objective_function)(theta_val)
        hessian_val = jax.hessian(objective_function)(theta_val)
        return grad_val, hessian_val

    for iteration in range(1, max_iters + 1):
        grad_val, hessian_val = grad_and_hessian(theta)
        grad_norm = float(jnp.linalg.norm(grad_val))
        if log_interval and iteration % log_interval == 0:
            print(f"[newton_solve] iter={iteration} grad_norm={grad_norm:.3e} theta={theta}")
        if grad_norm <= tol:
            break

        eye = jnp.eye(theta.shape[0])
        # 解析的に非正定な Hessian でも解けるように微小対角成分を加える。
        hessian_safe = hessian_val + 1e-6 * eye
        delta = jnp.linalg.solve(hessian_safe, grad_val)
        theta = theta - damping * delta
        theta = _clip(theta, bounds)

    return np.asarray(theta)


def _prior_from_bounds(low: float, high: float) -> dist.Distribution:
    if math.isfinite(low) and math.isfinite(high):
        return dist.Uniform(low, high)
    if math.isfinite(low):
        return dist.TransformedDistribution(
            dist.Exponential(1.0), transforms.AffineTransform(low, 1.0)
        )
    if math.isfinite(high):
        return dist.TransformedDistribution(
            dist.Exponential(1.0),
            transforms.AffineTransform(high, -1.0),
        )
    return dist.Normal(0.0, 10.0)


def bayes_estimate(
    objective_function: Callable[[Array1D], float],
    search_bounds: Bounds,
    initial_guess: Sequence[float],
    *,
    prior_log_pdf: Callable[[Array1D], float] | None = None,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 1,
    rng_seed: int = 0,
) -> np.ndarray:
    """Bayesian estimation via NumPyro and JAX."""
    bounds_raw = _normalize_bounds(search_bounds)
    theta0 = jnp.asarray(initial_guess)

    if theta0.shape != (bounds_raw.shape[0],):
        msg = f"initial_guess must have shape ({bounds_raw.shape[0]},)."
        raise ValueError(msg)

    bounds = bounds_raw.astype(theta0.dtype)
    bounds_np = np.asarray(bounds)

    def model():
        theta_components = []
        for i in range(bounds_np.shape[0]):
            low = float(bounds_np[i, 0])
            high = float(bounds_np[i, 1])
            prior = _prior_from_bounds(low, high)
            theta_i = numpyro.sample(f"theta_{i}", prior)
            theta_components.append(theta_i)
        theta = jnp.stack(theta_components)
        log_like = jnp.asarray(objective_function(theta))
        numpyro.factor("log_likelihood", log_like)
        if prior_log_pdf is not None:
            numpyro.factor("user_prior", jnp.asarray(prior_log_pdf(theta)))
        numpyro.deterministic("theta", theta)

    nuts_kernel = infer.NUTS(model)
    mcmc = infer.MCMC(
        nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains
    )
    rng_key = jax.random.PRNGKey(rng_seed)
    mcmc.run(rng_key)
    samples = mcmc.get_samples()
    theta_components = [samples[f"theta_{i}"] for i in range(bounds.shape[0])]
    theta_stack = jnp.stack(theta_components, axis=-1)
    posterior_mean = jnp.mean(theta_stack, axis=0)
    return np.asarray(posterior_mean)


__all__ = [
    "bayes_estimate",
    "m_estimate",
    "newton_solve",
    "one_step_estimate",
]

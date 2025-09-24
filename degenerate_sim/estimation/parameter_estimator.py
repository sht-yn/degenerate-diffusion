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
    return jnp.asarray(bounds_list, dtype=jnp.float64)


def _clip(theta: Array1D, bounds: jnp.ndarray) -> Array1D:
    lows = bounds[:, 0]
    highs = bounds[:, 1]
    return jnp.clip(theta, lows, highs)


def m_estimate(
    objective_function: Callable[[Array1D], float],
    search_bounds: Bounds,
    initial_guess: Sequence[float],
    *,
    learning_rate: float = 1e-2,
    max_iters: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """Gradient-ascent M-estimation implemented with JAX."""
    bounds = _normalize_bounds(search_bounds)
    theta0 = jnp.asarray(initial_guess, dtype=jnp.float64)

    if theta0.shape != (bounds.shape[0],):
        msg = f"initial_guess must have shape ({bounds.shape[0]},)."
        raise ValueError(msg)

    @jax.jit
    def step(theta: Array1D) -> tuple[Array1D, jnp.ndarray]:
        grad = jax.grad(objective_function)(theta)
        theta_next = theta + learning_rate * grad
        theta_next = _clip(theta_next, bounds)
        grad_norm = jnp.linalg.norm(grad)
        return theta_next, grad_norm

    theta_opt = theta0
    grad_norm_value = float("inf")
    for _ in range(max_iters):
        theta_opt, grad_norm = step(theta_opt)
        grad_norm_value = float(grad_norm)
        if grad_norm_value <= tol:
            break

    return np.asarray(theta_opt)


def one_step_estimate(
    objective_function: Callable[[Array1D], float],
    search_bounds: Bounds,
    initial_estimator: Sequence[float],
) -> np.ndarray:
    """Single Newton step using JAX gradients and Hessians."""
    bounds = _normalize_bounds(search_bounds)
    theta0 = jnp.asarray(initial_estimator, dtype=jnp.float64)

    if theta0.shape != (bounds.shape[0],):
        msg = f"initial_estimator must have shape ({bounds.shape[0]},)."
        raise ValueError(msg)

    grad = jax.grad(objective_function)(theta0)
    hessian = jax.hessian(objective_function)(theta0)

    eye = jnp.eye(theta0.shape[0])
    hessian_safe = hessian + 1e-6 * eye
    delta = jnp.linalg.solve(hessian_safe, grad)
    theta_new = theta0 - delta
    theta_new = _clip(theta_new, bounds)
    return np.asarray(theta_new)


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
    bounds = _normalize_bounds(search_bounds)
    theta0 = jnp.asarray(initial_guess, dtype=jnp.float64)

    if theta0.shape != (bounds.shape[0],):
        msg = f"initial_guess must have shape ({bounds.shape[0]},)."
        raise ValueError(msg)

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
    "one_step_estimate",
]

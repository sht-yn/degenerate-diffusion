from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest

from degenerate_diffusion.estimation.parameter_estimator_new import (
    build_bayes_transition,
    build_newton_ascent_solver,
    build_one_step_ascent,
)


def _concave_quadratic(
    mu: jnp.ndarray, A_posdef: jnp.ndarray
) -> Callable[[jnp.ndarray, object], jnp.ndarray]:
    """f(theta) = -(1/2) (theta-mu)^T A (theta-mu) (maximize -> converges to mu)."""

    def f(theta: jax.Array, _aux_unused: object) -> jax.Array:
        d = theta - mu
        return -0.5 * (d @ (A_posdef @ d))

    return f


def _gaussian_logprob_closed(
    mean: jnp.ndarray, cov: jnp.ndarray
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Log-probability of N(mean, cov) up to an additive constant."""
    inv_cov = jnp.linalg.inv(cov)

    def logprob(theta: jax.Array) -> jax.Array:
        d = theta - mean
        return cast("jax.Array", -0.5 * (d @ (inv_cov @ d)))

    return logprob


def test_newton_solver_converges_to_mu() -> None:
    dim = 3
    mu = jnp.arange(1.0, 1.0 + dim)
    # A is positive definite (e.g., diagonal 2,3,4).
    A = jnp.diag(jnp.arange(2.0, 2.0 + dim))
    objective = _concave_quadratic(mu, A)

    bounds = [(-math.inf, math.inf)] * dim  # 制約なし相当

    solver = build_newton_ascent_solver(
        objective, bounds, max_iters=50, tol=1e-10, damping=1.0, use_adam_fallback=False
    )
    theta0 = jnp.zeros((dim,))
    theta_hat = solver(theta0, None)

    assert jnp.allclose(theta_hat, mu, atol=1e-8)


def test_one_step_equals_solver_single_iter() -> None:
    dim = 2
    mu = jnp.array([1.0, -2.0])
    A = jnp.diag(jnp.array([3.0, 5.0]))
    objective = _concave_quadratic(mu, A)
    bounds = [(-math.inf, math.inf)] * dim

    step = build_one_step_ascent(objective, bounds, damping=1.0, eps=1e-9)
    solver_1 = build_newton_ascent_solver(
        objective, bounds, max_iters=1, tol=0.0, damping=1.0, use_adam_fallback=False
    )
    theta0 = jnp.array([10.0, 10.0])

    th_step = step(theta0, None)
    th_solver = solver_1(theta0, None)

    assert jnp.allclose(th_step, th_solver, atol=1e-10)


def test_bounds_clipping_projects_into_box() -> None:
    dim = 2
    mu = jnp.array([10.0, 10.0])  # 最大点は境界外
    A = jnp.eye(dim)
    objective = _concave_quadratic(mu, A)
    # 境界を [-1, 1] に固定
    bounds = [(-1.0, 1.0)] * dim

    step = build_one_step_ascent(objective, bounds, damping=1.0)
    theta0 = jnp.array([0.5, -0.5])

    theta_next = step(theta0, None)
    assert jnp.all(theta_next <= 1.0 + 1e-9)
    assert jnp.all(theta_next >= -1.0 - 1e-9)


@pytest.mark.skipif(
    pytest.importorskip("blackjax", reason="blackjax not installed") is None,
    reason="blackjax missing",
)
def test_bayes_transition_scan_reproducible_and_mean_moves() -> None:
    # 1次元の正規 N(1.0, 1.0)
    mean = jnp.array([1.0])
    cov = jnp.eye(1)
    logprob = _gaussian_logprob_closed(mean, cov)

    step = build_bayes_transition(
        logprob, step_size=0.2, max_num_doublings=6, inverse_mass_matrix=None
    )

    key = jax.random.PRNGKey(0)
    theta0 = jnp.array([0.0])

    def run_chain(key_in: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        def body(
            carry: tuple[jax.Array, jax.Array], _unused: None
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            th, k = carry
            th_next, k_next = step(th, k)
            return (th_next, k_next), th_next

        (th_fin, k_fin), samples = jax.lax.scan(body, (theta0, key_in), xs=None, length=1200)
        burnin = 300
        return samples[burnin:], th_fin, k_fin

    samples1, _, _ = run_chain(key)
    samples2, _, _ = run_chain(key)

    # Same key should yield identical chains.
    assert jnp.allclose(samples1, samples2)

    # Posterior mean approaches the target mean (tolerant threshold).
    m1 = jnp.mean(samples1, axis=0)
    assert jnp.all(jnp.abs(m1 - mean) < 0.35)

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest

from degenerate_diffusion.estimation.parameter_estimator_new import (
    build_b,
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
def test_bayes_sampler_reproducible_and_mean_moves() -> None:
    # 1次元の正規 N(1.0, 1.0)
    mean = jnp.array([1.0])
    cov = jnp.eye(1)
    logprob = _gaussian_logprob_closed(mean, cov)

    # aux を使わない場合でも、aux 版 API に合わせて引数を受ける関数にする
    def logprob_with_aux(theta: jax.Array, _aux_unused: object) -> jax.Array:
        return logprob(theta)

    sampler = build_b(
        logprob_with_aux,
        step_size=0.2,
        inverse_mass_matrix=None,
        num_warmup=300,
        num_samples=900,
        thin=1,
    )

    key = jax.random.PRNGKey(0)
    theta0 = jnp.array([0.0])

    # 同じキー・初期値・aux(None)なら再現性(同一結果)
    m1 = sampler(theta0, key, None)
    m2 = sampler(theta0, key, None)
    assert jnp.allclose(m1, m2)

    # 事後平均が真の平均に近づいているかを確認
    assert jnp.all(jnp.abs(m1 - mean) < 0.35)

"""JAX-native estimation kernels (builder style) for alternating assembly.

English: Pure JAX kernels returned by builders to be composed into alternating M/B schemes.
Japanese: 交互推定 (M/B) に合成しやすいよう、ビルダーが返す純 JAX カーネル群。
"""

from __future__ import annotations

# isort: skip_file

from collections.abc import Callable, Sequence
from typing import Any, TYPE_CHECKING, cast

import blackjax
import jax
import jax.numpy as jnp
import optax

if TYPE_CHECKING:  # pragma: no cover - type-only import
    from blackjax.base import SamplingState
else:  # pragma: no cover - runtime fallback
    SamplingState = Any


# Public API
__all__ = [
    "build_b",
    "build_m",
    "build_s",
]


# Type aliases
JaxArray = jax.Array
Bounds = Sequence[tuple[float | None, float | None]]
Aux = Any  # PyTree expected; numeric leaves only
ObjectiveFn = Callable[[JaxArray, Aux], JaxArray]  # maximize (scalar)
DerivativeFn = Callable[[JaxArray], JaxArray]
# Removed closed-over logprob transition API; only aux-based sampler is public


PAIR_LENGTH = 2


def _normalize_bounds(search_bounds: Bounds) -> JaxArray:
    """Normalize possibly-None bounds into a JAX array of shape (d, 2).

    English: Replace None with +/-inf and return as JAX array.
    Japanese: None を無限大に置換し、形状 (d,2) の JAX 配列に整形して返す。
    """
    lows: list[float] = []
    highs: list[float] = []
    for idx, bound in enumerate(search_bounds):
        if not isinstance(bound, tuple) or len(bound) != PAIR_LENGTH:
            msg = f"Bound at index {idx} must be a (low, high) tuple."
            raise ValueError(msg)
        low, high = bound
        low_f = float("-inf") if low is None else float(low)
        high_f = float("inf") if high is None else float(high)
        if low_f > high_f:
            msg = f"Lower bound {low_f} exceeds upper bound {high_f} at index {idx}."
            raise ValueError(msg)
        lows.append(low_f)
        highs.append(high_f)
    return jnp.stack([jnp.asarray(lows), jnp.asarray(highs)], axis=1)


def _clip_to_bounds(theta: JaxArray, bounds: JaxArray) -> JaxArray:
    """Project theta into [low, high] box constraints using jnp.clip."""
    return jnp.clip(theta, bounds[:, 0], bounds[:, 1])


def _prepare_theta_and_bounds(theta: JaxArray, bounds_arr: JaxArray) -> tuple[JaxArray, JaxArray]:
    """Cast theta to JAX array and align bounds dtype with theta."""
    theta_arr = jnp.asarray(theta)
    return theta_arr, bounds_arr.astype(theta_arr.dtype)


def _make_objective_derivatives(
    objective_fn: ObjectiveFn,
    aux: Aux,
) -> tuple[DerivativeFn, DerivativeFn]:
    """Wrap objective_fn to ensure JAX types and return grad/hess callables."""

    def obj(th: JaxArray) -> JaxArray:
        return jnp.asarray(objective_fn(th, aux))

    grad_fn = cast("DerivativeFn", jax.grad(obj))
    hess_fn = cast("DerivativeFn", jax.hessian(obj))
    return grad_fn, hess_fn


def _projected_newton_update(
    theta: JaxArray,
    grad: JaxArray,
    hess_sym: JaxArray,
    *,
    damping: float,
    eps: float,
    eye: JaxArray,
    bounds: JaxArray,
) -> JaxArray:
    """Apply a damped Newton ascent step followed by box projection."""
    hess_safe = hess_sym - eps * eye
    delta = jnp.linalg.solve(hess_safe, grad)
    theta_next = theta - damping * delta
    return _clip_to_bounds(theta_next, bounds)


def build_s(
    objective_fn: ObjectiveFn,
    bounds: Bounds | JaxArray,
    *,
    damping: float = 0.1,
    eps: float = 1e-8,
) -> Callable[[JaxArray, Aux], JaxArray]:
    """Build a single Newton-like ascent step: (theta, aux) -> theta_next.

    English: One damped Newton ascent step with Hessian stabilization and box projection.
    Japanese: 減衰付きニュートン上昇を 1 回行い、ヘッセ安定化とボックス射影を適用する。
    """
    bounds_arr = _normalize_bounds(bounds) if not isinstance(bounds, jax.Array) else bounds

    def step(theta: JaxArray, aux: Aux) -> JaxArray:
        theta_arr, bounds_cast = _prepare_theta_and_bounds(theta, bounds_arr)
        eye = jnp.eye(theta_arr.shape[0], dtype=theta_arr.dtype)
        grad_fn, hess_fn = _make_objective_derivatives(objective_fn, aux)
        grad = grad_fn(theta_arr)
        hess = hess_fn(theta_arr)
        hess_sym = 0.5 * (hess + hess.T)
        return _projected_newton_update(
            theta_arr,
            grad,
            hess_sym,
            damping=damping,
            eps=eps,
            eye=eye,
            bounds=bounds_cast,
        )

    return jax.jit(step)


def _newton_only_solver_factory(
    objective_fn: ObjectiveFn,
    bounds_arr: JaxArray,
    max_iters: int,
    tol: float,
    damping: float,
    eps: float,
) -> Callable[[JaxArray, Aux], JaxArray]:
    def solver(theta0: JaxArray, aux: Aux) -> JaxArray:
        theta0_arr, bounds_cast = _prepare_theta_and_bounds(theta0, bounds_arr)
        eye = jnp.eye(theta0_arr.shape[0], dtype=theta0_arr.dtype)
        false_flag = jnp.zeros((), dtype=jnp.bool_)

        grad_fn, hess_fn = _make_objective_derivatives(objective_fn, aux)

        def cond(carry: tuple[JaxArray, JaxArray, JaxArray]) -> JaxArray:
            _th, it, converged = carry
            return jnp.logical_and(it < max_iters, jnp.logical_not(converged))

        def body(
            carry: tuple[JaxArray, JaxArray, JaxArray],
        ) -> tuple[JaxArray, JaxArray, JaxArray]:
            th, it, _ = carry
            g = grad_fn(th)
            H = hess_fn(th)
            grad_norm = jnp.linalg.norm(g)
            converged_now = grad_norm <= tol
            H_sym = 0.5 * (H + H.T)
            th_next = _projected_newton_update(
                th,
                g,
                H_sym,
                damping=damping,
                eps=eps,
                eye=eye,
                bounds=bounds_cast,
            )
            return th_next, it + 1, converged_now

        th_fin, _it_fin, _cv_fin = jax.lax.while_loop(
            cond,
            body,
            (theta0_arr, jnp.asarray(0, dtype=jnp.int32), false_flag),
        )
        return th_fin

    return jax.jit(solver)


def _newton_with_adam_solver_factory(
    objective_fn: ObjectiveFn,
    bounds_arr: JaxArray,
    max_iters: int,
    tol: float,
    damping: float,
    learning_rate: float,
    weight_decay: float,
    clip_norm: float,
    eps: float,
) -> Callable[[JaxArray, Aux], JaxArray]:
    adam = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    optimizer = optax.chain(optax.clip_by_global_norm(clip_norm), adam)

    def solver(theta0: JaxArray, aux: Aux) -> JaxArray:
        theta0_arr, bounds_cast = _prepare_theta_and_bounds(theta0, bounds_arr)
        eye = jnp.eye(theta0_arr.shape[0], dtype=theta0_arr.dtype)
        false_flag = jnp.zeros((), dtype=jnp.bool_)

        grad_fn, hess_fn = _make_objective_derivatives(objective_fn, aux)

        opt_state0 = optimizer.init(theta0_arr)

        def cond(
            carry: tuple[JaxArray, optax.OptState, JaxArray, JaxArray],
        ) -> JaxArray:
            _th, _st, it, converged = carry
            return jnp.logical_and(it < max_iters, jnp.logical_not(converged))

        def body(
            carry: tuple[JaxArray, optax.OptState, JaxArray, JaxArray],
        ) -> tuple[JaxArray, optax.OptState, JaxArray, JaxArray]:
            th, st, it, _ = carry
            g = grad_fn(th)
            H = hess_fn(th)
            grad_norm = jnp.linalg.norm(g)
            converged_now = grad_norm <= tol

            H_sym = 0.5 * (H + H.T)
            max_eig = jnp.max(jnp.linalg.eigvalsh(H_sym))
            use_newton = max_eig < 0.0

            def do_newton(
                cur: tuple[JaxArray, optax.OptState],
            ) -> tuple[JaxArray, optax.OptState]:
                t, s = cur
                t_next = _projected_newton_update(
                    t,
                    g,
                    H_sym,
                    damping=damping,
                    eps=eps,
                    eye=eye,
                    bounds=bounds_cast,
                )
                return t_next, s

            def do_adam(
                cur: tuple[JaxArray, optax.OptState],
            ) -> tuple[JaxArray, optax.OptState]:
                t, s = cur
                update, s_next = optimizer.update(-g, s, t)
                t_next = optax.apply_updates(t, update)
                return t_next, s_next

            th_next, st_next = jax.lax.cond(use_newton, do_newton, do_adam, (th, st))
            th_next = _clip_to_bounds(th_next, bounds_cast)
            return th_next, st_next, it + 1, converged_now

        th_fin, _st_fin, _it_fin, _cv_fin = jax.lax.while_loop(
            cond,
            body,
            (theta0_arr, opt_state0, jnp.asarray(0, dtype=jnp.int32), false_flag),
        )
        return th_fin

    return jax.jit(solver)


def build_m(
    objective_fn: ObjectiveFn,
    bounds: Bounds | JaxArray,
    *,
    max_iters: int = 10_000,
    tol: float = 1e-7,
    damping: float = 0.1,
    use_adam_fallback: bool = True,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.0,
    clip_norm: float = 1e2,
    eps: float = 1e-8,
) -> Callable[[JaxArray, Aux], JaxArray]:
    """Build a damped Newton ascent solver, with optional AdamW fallback.

    English: Returns (theta0, aux) -> theta_hat using lax.while_loop to enforce
    ||grad||<=tol.
    Japanese: ||grad||<=tol になるまで lax.while_loop で反復する上昇ソルバを返す。
    """
    bounds_arr = _normalize_bounds(bounds) if not isinstance(bounds, jax.Array) else bounds

    if use_adam_fallback:
        return _newton_with_adam_solver_factory(
            objective_fn,
            bounds_arr,
            max_iters,
            tol,
            damping,
            learning_rate,
            weight_decay,
            clip_norm,
            eps,
        )

    return _newton_only_solver_factory(
        objective_fn,
        bounds_arr,
        max_iters,
        tol,
        damping,
        eps,
    )


# (removed) build_bayes_transition: use aux-based internals only; inline transition in sampler


def _log_prior_component(theta_val: JaxArray, low: JaxArray, high: JaxArray) -> JaxArray:
    finite_low = jnp.isfinite(low)
    finite_high = jnp.isfinite(high)

    def both_finite(_: None) -> JaxArray:
        width = high - low
        log_norm = -jnp.log(width)
        inside = jnp.logical_and(theta_val >= low, theta_val <= high)
        return jnp.where(inside, log_norm, -jnp.inf)

    def only_low(_: None) -> JaxArray:
        shifted = theta_val - low
        log_pdf = -shifted
        return jnp.where(theta_val >= low, log_pdf, -jnp.inf)

    def only_high(_: None) -> JaxArray:
        shifted = high - theta_val
        log_pdf = -shifted
        return jnp.where(theta_val <= high, log_pdf, -jnp.inf)

    def both_inf(_: None) -> JaxArray:
        log_norm = -jnp.log(10.0 * jnp.sqrt(2.0 * jnp.pi))
        standardized = theta_val / 10.0
        return log_norm - 0.5 * standardized**2

    return cast(
        "JaxArray",
        jax.lax.cond(
            jnp.logical_and(finite_low, finite_high),
            both_finite,
            lambda _: jax.lax.cond(
                finite_low,
                only_low,
                lambda _: jax.lax.cond(finite_high, only_high, both_inf, None),
                None,
            ),
            None,
        ),
    )


def _log_prior_from_bounds(theta: JaxArray, bounds: JaxArray) -> JaxArray:
    logps = jax.vmap(_log_prior_component)(theta, bounds[:, 0], bounds[:, 1])
    return jnp.sum(logps)


def build_b(
    logprob_fn: Callable[[JaxArray, Aux], JaxArray],
    bounds: Bounds | JaxArray | None = None,
    *,
    step_size: float = 1e-1,
    inverse_mass_matrix: JaxArray | None = None,
    num_warmup: int = 1000,
    num_samples: int = 3000,
    thin: int = 1,
    target_acceptance_rate: float = 0.75,
) -> Callable[[JaxArray, JaxArray, Aux], JaxArray]:
    """Build a NUTS sampler that returns the mean of posterior draws.

    (theta0, key, aux) -> theta_mean
    - Assumes the new BlackJAX API: pass step_size and inverse_mass_matrix when
      constructing the kernel; call nuts.step(key, state) without extra kwargs.
      Optional thinning via `thin`.
    """
    num_warmup_i = int(num_warmup)
    num_samples_i = int(num_samples)
    thin_i = int(max(thin, 1))
    bounds_arr = None
    if bounds is not None:
        bounds_arr = _normalize_bounds(bounds) if not isinstance(bounds, jax.Array) else bounds

    def run(theta0: JaxArray, key: JaxArray, aux: Aux) -> JaxArray:
        theta0 = jnp.asarray(theta0)
        bounds_cast = None
        if bounds_arr is not None:
            theta0, bounds_cast = _prepare_theta_and_bounds(theta0, bounds_arr)

        def logprob(th: JaxArray) -> JaxArray:
            base = jnp.asarray(logprob_fn(th, aux))
            if bounds_cast is not None:
                base = base + _log_prior_from_bounds(th, bounds_cast)
            return base

        def _thin_transition(
            carry: tuple[SamplingState, JaxArray],
            _i: JaxArray,
        ) -> tuple[tuple[SamplingState, JaxArray], None]:
            st, k_cur = carry
            k_cur, k_use = jax.random.split(k_cur)
            st_new, _info = nuts.step(k_use, st)
            return (st_new, k_cur), None

        def _sample_transition(
            carry: tuple[SamplingState, JaxArray],
            _i: JaxArray,
        ) -> tuple[tuple[SamplingState, JaxArray], JaxArray]:
            st, k_cur = carry
            (st_new, k_after), _ = jax.lax.scan(
                _thin_transition,
                (st, k_cur),
                jnp.arange(thin_i),
            )
            return (st_new, k_after), st_new.position

        has_user_metric = inverse_mass_matrix is not None
        metric_init = jnp.ones_like(theta0) if inverse_mass_matrix is None else inverse_mass_matrix

        if num_warmup_i > 0 and not has_user_metric:
            warmup_algo = blackjax.window_adaptation(
                blackjax.nuts,
                logprob,
                initial_step_size=step_size,
                target_acceptance_rate=target_acceptance_rate,
            )
            key_warmup, key_samples = jax.random.split(key)
            warmup_result, _adapt_info = warmup_algo.run(key_warmup, theta0, num_warmup_i)
            tuned_params = warmup_result.parameters
            tuned_step = tuned_params["step_size"]
            tuned_metric = tuned_params["inverse_mass_matrix"]
            nuts = blackjax.nuts(logprob, step_size=tuned_step, inverse_mass_matrix=tuned_metric)
            state_init = warmup_result.state
            key_use = key_samples
        else:
            nuts = blackjax.nuts(logprob, step_size=step_size, inverse_mass_matrix=metric_init)
            state_init = nuts.init(theta0)
            key_use = key

        (_, _), samples = jax.lax.scan(
            _sample_transition,
            (state_init, key_use),
            jnp.arange(num_samples_i),
        )
        return jnp.mean(samples, axis=0)

    return jax.jit(run)

"""JAX-native estimation kernels (builder style) for alternating assembly.

English: Pure JAX kernels returned by builders to be composed into alternating M/B schemes.
Japanese: 交互推定 (M/B) に合成しやすいよう、ビルダーが返す純 JAX カーネル群。
"""

from __future__ import annotations

# isort: skip_file

from collections.abc import Callable, Sequence
from typing import Any, cast

import blackjax
import jax
import jax.numpy as jnp
import optax


# Public API
__all__ = [
    "build_bayes_transition",
    "build_bayes_transition_with_aux",
    "build_newton_ascent_solver",
    "build_one_step_ascent",
]


# Type aliases
JaxArray = jax.Array
Bounds = Sequence[tuple[float | None, float | None]]
Aux = Any  # PyTree expected; numeric leaves only
ObjectiveFn = Callable[[JaxArray, Aux], JaxArray]  # maximize (scalar)
LogProbFn = Callable[[JaxArray], JaxArray]


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


def build_one_step_ascent(
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
        theta = jnp.asarray(theta)
        b = bounds_arr.astype(theta.dtype)

        def obj(th: JaxArray) -> JaxArray:
            return jnp.asarray(objective_fn(th, aux))

        grad = jax.grad(obj)(theta)
        hess = jax.hessian(obj)(theta)
        hess_sym = 0.5 * (hess + hess.T)
        eye = jnp.eye(theta.shape[0], dtype=theta.dtype)
        hess_safe = hess_sym - eps * eye
        # Newton ascent: theta <- theta - damping * inv(H) * grad
        delta = jnp.linalg.solve(hess_safe, grad)
        theta_next = theta - damping * delta
        return _clip_to_bounds(theta_next, b)

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
        theta0 = jnp.asarray(theta0)
        b = bounds_arr.astype(theta0.dtype)
        eye = jnp.eye(theta0.shape[0], dtype=theta0.dtype)
        false_flag = jnp.zeros((), dtype=jnp.bool_)

        def obj(th: JaxArray) -> JaxArray:
            return jnp.asarray(objective_fn(th, aux))

        def grad_val(th: JaxArray) -> JaxArray:
            # jax.grad has incomplete type info; cast to JaxArray for mypy
            return cast("JaxArray", jax.grad(obj)(th))

        def hess_val(th: JaxArray) -> JaxArray:
            # jax.hessian has incomplete type info; cast to JaxArray for mypy
            return cast("JaxArray", jax.hessian(obj)(th))

        def cond(carry: tuple[JaxArray, JaxArray, JaxArray]) -> JaxArray:
            _th, it, converged = carry
            return jnp.logical_and(it < max_iters, jnp.logical_not(converged))

        def body(
            carry: tuple[JaxArray, JaxArray, JaxArray],
        ) -> tuple[JaxArray, JaxArray, JaxArray]:
            th, it, _ = carry
            g = grad_val(th)
            H = hess_val(th)
            grad_norm = jnp.linalg.norm(g)
            converged_now = grad_norm <= tol
            H_sym = 0.5 * (H + H.T)
            H_safe = H_sym - eps * eye
            delta = jnp.linalg.solve(H_safe, g)
            th_next = _clip_to_bounds(th - damping * delta, b)
            return th_next, it + 1, converged_now

        th_fin, _it_fin, _cv_fin = jax.lax.while_loop(
            cond,
            body,
            (theta0, jnp.asarray(0, dtype=jnp.int32), false_flag),
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
        theta0 = jnp.asarray(theta0)
        b = bounds_arr.astype(theta0.dtype)
        eye = jnp.eye(theta0.shape[0], dtype=theta0.dtype)
        false_flag = jnp.zeros((), dtype=jnp.bool_)

        def obj(th: JaxArray) -> JaxArray:
            return jnp.asarray(objective_fn(th, aux))

        def grad_val(th: JaxArray) -> JaxArray:
            return cast("JaxArray", jax.grad(obj)(th))

        def hess_val(th: JaxArray) -> JaxArray:
            return cast("JaxArray", jax.hessian(obj)(th))

        opt_state0 = optimizer.init(theta0)

        def cond(
            carry: tuple[JaxArray, optax.OptState, JaxArray, JaxArray],
        ) -> JaxArray:
            _th, _st, it, converged = carry
            return jnp.logical_and(it < max_iters, jnp.logical_not(converged))

        def body(
            carry: tuple[JaxArray, optax.OptState, JaxArray, JaxArray],
        ) -> tuple[JaxArray, optax.OptState, JaxArray, JaxArray]:
            th, st, it, _ = carry
            g = grad_val(th)
            H = hess_val(th)
            grad_norm = jnp.linalg.norm(g)
            converged_now = grad_norm <= tol

            H_sym = 0.5 * (H + H.T)
            max_eig = jnp.max(jnp.linalg.eigvalsh(H_sym))
            use_newton = max_eig < 0.0

            def do_newton(
                cur: tuple[JaxArray, optax.OptState],
            ) -> tuple[JaxArray, optax.OptState]:
                t, s = cur
                H_safe = H_sym - eps * eye
                delta = jnp.linalg.solve(H_safe, g)
                t_next = t - damping * delta
                return t_next, s

            def do_adam(
                cur: tuple[JaxArray, optax.OptState],
            ) -> tuple[JaxArray, optax.OptState]:
                t, s = cur
                update, s_next = optimizer.update(-g, s, t)
                t_next = optax.apply_updates(t, update)
                return t_next, s_next

            th_next, st_next = jax.lax.cond(use_newton, do_newton, do_adam, (th, st))
            th_next = _clip_to_bounds(th_next, b)
            return th_next, st_next, it + 1, converged_now

        th_fin, _st_fin, _it_fin, _cv_fin = jax.lax.while_loop(
            cond,
            body,
            (theta0, opt_state0, jnp.asarray(0, dtype=jnp.int32), false_flag),
        )
        return th_fin

    return jax.jit(solver)


def build_newton_ascent_solver(
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


def build_bayes_transition(
    logprob_fn: LogProbFn,
    *,
    step_size: float = 1e-1,
    max_num_doublings: int = 8,
    inverse_mass_matrix: JaxArray | None = None,
) -> Callable[[JaxArray, JaxArray], tuple[JaxArray, JaxArray]]:
    """Build a NUTS transition for a closed-over logprob: (theta, key) -> (theta', key').

    Use this when ``logprob_fn`` is already closed over any auxiliary data. The returned
    step function re-initializes the NUTS state from ``theta`` at each call and performs
    a single NUTS transition.
    """

    def step(theta: JaxArray, key: JaxArray) -> tuple[JaxArray, JaxArray]:
        theta = jnp.asarray(theta)
        inv_mass = inverse_mass_matrix
        # Provide a sane default metric if None is given (compat with older BlackJAX).
        inv_mass = jnp.ones_like(theta) if inv_mass is None else inv_mass

        def logprob(th: JaxArray) -> JaxArray:
            return jnp.asarray(logprob_fn(th))

        key, subkey = jax.random.split(key)

        # BlackJAX API compatibility: try new-style (pass step params to step),
        # fall back to old-style (pass to constructor; step without step_size).
        try:
            nuts = blackjax.nuts(logprob)
            state = nuts.init(theta)
            new_state, _info = nuts.step(
                subkey,
                state,
                step_size=step_size,
                inverse_mass_matrix=inv_mass,
                max_num_doublings=max_num_doublings,
            )
        except TypeError:
            # Older API: provide step_size/inv_mass at construction
            try:
                nuts = blackjax.nuts(logprob, step_size=step_size, inverse_mass_matrix=inv_mass)
            except TypeError:
                # Positional fallback for even older versions
                nuts = blackjax.nuts(logprob, step_size, inverse_mass_matrix=inv_mass)
            state = nuts.init(theta)
            try:
                new_state, _info = nuts.step(subkey, state, max_num_doublings=max_num_doublings)
            except TypeError:
                new_state, _info = nuts.step(subkey, state)
        return new_state.position, key

    return jax.jit(step)


def build_bayes_transition_with_aux(
    logprob_fn: Callable[[JaxArray, Aux], JaxArray],
    *,
    step_size: float = 1e-1,
    max_num_doublings: int = 8,
    inverse_mass_matrix: JaxArray | None = None,
) -> Callable[[JaxArray, JaxArray, Aux], tuple[JaxArray, JaxArray]]:
    """Build a NUTS transition that accepts auxiliary data: (theta, key, aux) -> (theta', key').

    English: For cases where the log-prob depends on data or parameters (aux), keep it stateless by
    passing aux at call-time.
    Japanese: 観測データなどの aux に依存する対数確率で、呼び出し時に aux を渡すステートレス版。
    """

    def step(theta: JaxArray, key: JaxArray, aux: Aux) -> tuple[JaxArray, JaxArray]:
        theta = jnp.asarray(theta)
        inv_mass = inverse_mass_matrix
        inv_mass = jnp.ones_like(theta) if inv_mass is None else inv_mass

        def logprob(th: JaxArray) -> JaxArray:
            return jnp.asarray(logprob_fn(th, aux))

        key, subkey = jax.random.split(key)

        try:
            nuts = blackjax.nuts(logprob)
            state = nuts.init(theta)
            new_state, _info = nuts.step(
                subkey,
                state,
                step_size=step_size,
                inverse_mass_matrix=inv_mass,
                max_num_doublings=max_num_doublings,
            )
        except TypeError:
            try:
                nuts = blackjax.nuts(logprob, step_size=step_size, inverse_mass_matrix=inv_mass)
            except TypeError:
                nuts = blackjax.nuts(logprob, step_size, inverse_mass_matrix=inv_mass)
            state = nuts.init(theta)
            try:
                new_state, _info = nuts.step(subkey, state, max_num_doublings=max_num_doublings)
            except TypeError:
                new_state, _info = nuts.step(subkey, state)
        return new_state.position, key

    return jax.jit(step)

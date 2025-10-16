"""JAX-native parameter estimation utilities."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
from numpyro import infer
from numpyro.distributions import transforms
from numpyro.infer import init_to_sample

PAIR_LENGTH = 2

Bounds = Sequence[tuple[float | None, float | None]]
Array1D = jnp.ndarray


def _normalize_bounds(search_bounds: Bounds) -> jnp.ndarray:
    if not isinstance(search_bounds, Sequence):  # type: ignore[arg-type]
        msg = "search_bounds must be a sequence of (low, high) tuples."
        raise TypeError(msg)

    bounds_list: list[tuple[float, float]] = []
    for idx, bound in enumerate(search_bounds):
        if not isinstance(bound, tuple) or len(bound) != PAIR_LENGTH:
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


def newton_solve(
    objective_function: Callable[[Array1D], float],
    search_bounds: Bounds,
    initial_guess: Sequence[float],
    *,
    max_iters: int = 10000,
    tol: float = 1e-7,
    damping: float = 0.1,
    log_interval: int | None = None,
    fallback_learning_rate: float = 1e-5,
    fallback_weight_decay: float = 0.0,
    clip_norm: float = 1e2,
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

    if fallback_learning_rate <= 0.0:
        msg = "fallback_learning_rate must be positive."
        raise ValueError(msg)
    if clip_norm <= 0.0:
        msg = "clip_norm must be positive."
        raise ValueError(msg)

    adamw_transform = optax.adamw(
        learning_rate=fallback_learning_rate,
        weight_decay=fallback_weight_decay,
    )
    optimizer = optax.chain(optax.clip_by_global_norm(clip_norm), adamw_transform)
    opt_state = optimizer.init(theta)

    def grad_and_hessian(theta_val: Array1D) -> tuple[Array1D, jnp.ndarray]:
        grad_val = jax.grad(objective_function)(theta_val)
        hessian_val = jax.hessian(objective_function)(theta_val)
        return grad_val, hessian_val

    eps = 1e-8

    for iteration in range(1, max_iters + 1):
        grad_val, hessian_val = grad_and_hessian(theta)
        grad_norm = float(jnp.linalg.norm(grad_val))
        if log_interval and iteration % log_interval == 0:
            print(f"[newton_solve] iter={iteration} grad_norm={grad_norm:.3e} theta={theta}")
        if grad_norm <= tol:
            break

        eye = jnp.eye(theta.shape[0])
        hessian_sym = 0.5 * (hessian_val + hessian_val.T)
        eigvals = jnp.linalg.eigvalsh(hessian_sym)
        max_eig = jnp.max(eigvals)
        all_negative = max_eig < 0

        opt_update, opt_state_candidate = optimizer.update(-grad_val, opt_state, theta)

        if all_negative:
            # Newton ステップ (自然勾配) を実行。数値安定化のため微小対角成分を加える。
            hessian_safe = hessian_sym - eps * eye
            delta = jnp.linalg.solve(hessian_safe, grad_val)
            theta = theta - damping * delta
            opt_state = opt_state_candidate
        else:
            # 曲率が反転している場合は一次法にフォールバック。
            if log_interval and iteration % log_interval == 0:
                print("  -> fallback to AdamW step with gradient clipping (indefinite Hessian)")
            theta = optax.apply_updates(theta, opt_update)
            opt_state = opt_state_candidate

        theta = _clip(theta, bounds)

    return np.asarray(theta)


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
    hessian_safe = hessian + 1e-8 * eye
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
    num_samples: int = 3000,
    num_chains: int = 1,
    rng_seed: int = 0,
    init_strategy: Callable[..., object] | None = None,
) -> np.ndarray:
    """Bayesian estimation via NumPyro and JAX."""
    bounds_raw = _normalize_bounds(search_bounds)
    theta0 = jnp.asarray(initial_guess)

    if theta0.shape != (bounds_raw.shape[0],):
        msg = f"initial_guess must have shape ({bounds_raw.shape[0]},)."
        raise ValueError(msg)

    bounds = bounds_raw.astype(theta0.dtype)
    bounds_np = np.asarray(bounds)

    def model() -> None:
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

    strategy = init_strategy or init_to_sample()
    nuts_kernel = infer.NUTS(model, init_strategy=strategy)
    mcmc = infer.MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False,
    )
    rng_key = jax.random.PRNGKey(rng_seed)
    mcmc.run(rng_key)
    samples = mcmc.get_samples()
    theta_components = [samples[f"theta_{i}"] for i in range(bounds.shape[0])]
    theta_stack = jnp.stack(theta_components, axis=-1)
    posterior_mean = jnp.mean(theta_stack, axis=0)
    return np.asarray(posterior_mean)


def m_estimate(
    objective_function: Callable[[Array1D], float],
    search_bounds: Bounds,
    initial_guess: Sequence[float],
    *,
    learning_rate: float = 1e-3,
    max_iters: int = 5000,
    tol: float = 1e-6,
    log_interval: int | None = None,
) -> np.ndarray:
    """Gradient-ascent M-estimation implemented with JAX.

    The ``learning_rate`` の既定値を ``1e-3`` に下げ、必要に応じて ``log_interval`` (反復回数) で
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


def m_estimate_jax(
    objective_function: Callable[[Array1D], float],
    search_bounds: Bounds,
    initial_guess: Sequence[float],
    *,
    learning_rate: float = 1e-3,
    max_iters: int = 1000,
    tol: float = 1e-6,
    log_interval: int | None = None,
) -> np.ndarray:
    """Optax ベースの JAX ネイティブ M 推定器。

    ``m_estimate`` と同様に目的関数を最大化するが、Optax のオプティマイザを用いて
    JIT 互換なループを構成する。 ``log_interval`` を設定すると ``jax.debug.print`` で
    進捗を表示できる。
    """
    if learning_rate <= 0.0:
        msg = "learning_rate must be positive."
        raise ValueError(msg)
    if max_iters <= 0:
        msg = "max_iters must be positive."
        raise ValueError(msg)
    if tol < 0.0:
        msg = "tol must be non-negative."
        raise ValueError(msg)

    bounds_raw = _normalize_bounds(search_bounds)
    theta0 = jnp.asarray(initial_guess)

    if theta0.shape != (bounds_raw.shape[0],):
        msg = f"initial_guess must have shape ({bounds_raw.shape[0]},)."
        raise ValueError(msg)

    bounds = bounds_raw.astype(theta0.dtype)
    theta0 = _clip(theta0, bounds)

    def objective(theta: Array1D) -> jnp.ndarray:
        return jnp.asarray(objective_function(theta))

    def loss(theta: Array1D) -> jnp.ndarray:
        return -objective(theta)

    optimizer = optax.sgd(learning_rate)

    @jax.jit
    def _run(theta_init: Array1D) -> tuple[Array1D, jnp.ndarray, jnp.ndarray]:
        opt_state = optimizer.init(theta_init)
        loss0, grad0 = jax.value_and_grad(loss)(theta_init)
        grad_norm0 = jnp.linalg.norm(grad0)
        iter0 = jnp.asarray(0, dtype=jnp.int32)

        def cond_fn(
            state: tuple[Array1D, optax.OptState, Array1D, jnp.ndarray, jnp.ndarray],
        ) -> jnp.ndarray:
            _, _, _, grad_norm_val, iter_val = state
            continue_grad = grad_norm_val > tol
            continue_iter = iter_val < max_iters
            return jnp.logical_and(continue_grad, continue_iter)

        def body_fn(
            state: tuple[Array1D, optax.OptState, Array1D, jnp.ndarray, jnp.ndarray],
        ) -> tuple[Array1D, optax.OptState, Array1D, jnp.ndarray, jnp.ndarray]:
            theta_curr, opt_state_curr, grad_curr, grad_norm_curr, iter_val = state
            updates, opt_state_next = optimizer.update(grad_curr, opt_state_curr, theta_curr)
            theta_next = optax.apply_updates(theta_curr, updates)
            theta_next = _clip(theta_next, bounds)
            _loss_next, grad_next = jax.value_and_grad(loss)(theta_next)
            grad_norm_next = jnp.linalg.norm(grad_next)
            iter_next = iter_val + 1

            if log_interval is not None and log_interval > 0:
                log_every = jnp.asarray(log_interval, dtype=iter_next.dtype)
                should_log = jnp.logical_and(
                    log_every > 0, jnp.equal(jnp.mod(iter_next, log_every), 0)
                )

                def _log_fn(_: None) -> None:
                    jax.debug.print(
                        "[m_estimate_jax] iter={iter} grad_norm={grad_norm:.3e} theta={theta}",
                        iter=iter_next,
                        grad_norm=grad_norm_next,
                        theta=theta_next,
                    )

                def _noop_fn(_: None) -> None:
                    return None

                jax.lax.cond(should_log, _log_fn, _noop_fn, operand=None)

            return theta_next, opt_state_next, grad_next, grad_norm_next, iter_next

        theta_final, opt_state_final, grad_final, grad_norm_final, iter_final = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (theta_init, opt_state, grad0, grad_norm0, iter0),
        )
        return theta_final, grad_final, grad_norm_final

    theta_opt, _, _ = _run(theta0)
    return np.asarray(theta_opt)


__all__ = [
    "bayes_estimate",
    "m_estimate",
    "m_estimate_jax",
    "newton_solve",
    "one_step_estimate",
]

"""JAX-compiled per-seed loop runner building on stateless evaluators.

This module provides a seed runner that performs:
1) Simulation with model.make_simulation_kernel
2) Alternating estimation for theta1/theta2/theta3 with M/B/S choices

It mirrors LoopEstimationAlgorithm._run_single_seed but returns a pure function
that can be jitted and later vmapped over seeds.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import jax
import jax.numpy as jnp

from degenerate_diffusion.estimation.parameter_estimator_new import (
    Bounds as JaxBounds,
    JaxArray,
    _normalize_bounds,
    build_bayes_transition,
)

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from degenerate_diffusion.evaluation.likelihood_evaluator import (
        LikelihoodEvaluator,
    )
    from degenerate_diffusion.processes.degenerate_diffusion_process import (
        DegenerateDiffusionProcess,
    )
else:  # pragma: no cover - runtime fallbacks
    pass

EstimatorKind = Literal["M", "B", "S"]

# Shorthand for 3-parameter tuple
ThetaTriple = tuple[JaxArray, JaxArray, JaxArray]
BranchRunner = Callable[[JaxArray, JaxArray], JaxArray]


def _kw_int(d: Mapping[str, object], key: str, default: int) -> int:
    v = d.get(key, default)
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    return default


def _kw_float(d: Mapping[str, object], key: str, default: float) -> float:
    v = d.get(key, default)
    if isinstance(v, (int, float)):
        return float(v)
    return default


def _kw_jax_array_opt(d: Mapping[str, object], key: str) -> JaxArray | None:
    v = d.get(key)
    return v if isinstance(v, jax.Array) else None


@dataclass(frozen=True)
class SeedRunnerConfig:
    """Configuration required to build the per-seed JAX loop runner.

    Attributes:
        true_theta: Ground-truth parameters used for simulation.
        t_max, h, burn_out, dt: Simulation settings forwarded to the model kernel.
        bounds_theta1..3: Box constraints for each theta component.
        newton_kwargs, nuts_kwargs, one_step_kwargs: Hyper-parameters for the estimators.

    """

    true_theta: tuple[JaxArray, JaxArray, JaxArray]
    t_max: float
    h: float
    burn_out: float
    dt: float
    bounds_theta1: JaxBounds
    bounds_theta2: JaxBounds
    bounds_theta3: JaxBounds
    # Estimator hyperparameters
    newton_kwargs: Mapping[str, object]
    nuts_kwargs: Mapping[str, object]
    one_step_kwargs: Mapping[str, object]


def build_simulator_kernel(
    model: DegenerateDiffusionProcess,
    *,
    t_max: float,
    h: float,
    burn_out: float,
    dt: float,
) -> Callable[
    [tuple[JaxArray, JaxArray, JaxArray], JaxArray, JaxArray, JaxArray],
    tuple[JaxArray, JaxArray],
]:
    """Return jitted simulator kernel from the model."""
    return model.make_simulation_kernel(t_max=t_max, h=h, burn_out=burn_out, dt=dt)


def _make_objectives(
    evaluator: LikelihoodEvaluator,
    k: int,
    x_series: JaxArray,
    y_series: JaxArray,
    h: float,
    theta_bar: tuple[JaxArray, JaxArray, JaxArray],
) -> tuple[
    Callable[[JaxArray], JaxArray],
    Callable[[JaxArray], JaxArray],
    Callable[[JaxArray], JaxArray],
    Callable[[JaxArray], JaxArray],
]:
    """Build l1', l1, l2, l3 stateless objectives closed over data and theta_bar."""
    t1b, t2b, t3b = (
        jnp.asarray(theta_bar[0]),
        jnp.asarray(theta_bar[1]),
        jnp.asarray(theta_bar[2]),
    )
    l1p = evaluator.make_stateless_quasi_l1_prime_evaluator(k=k)
    l1 = evaluator.make_stateless_quasi_l1_evaluator(k=k)
    l2 = evaluator.make_stateless_quasi_l2_evaluator(k=k)
    l3 = evaluator.make_stateless_quasi_l3_evaluator(k=k)

    def obj_l1p(th: JaxArray) -> JaxArray:
        return jnp.asarray(l1p(jnp.asarray(th), t1b, t2b, t3b, x_series, y_series, jnp.asarray(h)))

    def obj_l1(th: JaxArray) -> JaxArray:
        return jnp.asarray(l1(jnp.asarray(th), t1b, t2b, t3b, x_series, y_series, jnp.asarray(h)))

    def obj_l2(th: JaxArray) -> JaxArray:
        return jnp.asarray(l2(jnp.asarray(th), t1b, t2b, t3b, x_series, y_series, jnp.asarray(h)))

    def obj_l3(th: JaxArray) -> JaxArray:
        return jnp.asarray(l3(jnp.asarray(th), t1b, t2b, t3b, x_series, y_series, jnp.asarray(h)))

    return obj_l1p, obj_l1, obj_l2, obj_l3


def _build_component_solver(  # noqa: C901, PLR0915
    kind: EstimatorKind,
    bounds: JaxBounds,
    *,
    newton_kwargs: Mapping[str, object],
    nuts_kwargs: Mapping[str, object],
    one_step_kwargs: Mapping[str, object],
) -> Callable[[JaxArray, Callable[[JaxArray], JaxArray], JaxArray], JaxArray]:
    """Return (theta0, objective, key) -> theta_hat for a component."""
    if kind == "M":
        # Inline Newton ascent to avoid passing Python callables through jit
        b_arr = _normalize_bounds(bounds)
        max_iters = _kw_int(newton_kwargs, "max_iters", 100)
        tol = _kw_float(newton_kwargs, "tol", 1e-6)
        damping = _kw_float(newton_kwargs, "damping", 0.1)
        eps = _kw_float(newton_kwargs, "eps", 1e-8)

        def run_m(
            theta0: JaxArray,
            objective: Callable[[JaxArray], JaxArray],
            _key: JaxArray,
        ) -> JaxArray:
            theta0 = jnp.asarray(theta0)
            b = b_arr.astype(theta0.dtype)
            eye = jnp.eye(theta0.shape[0], dtype=theta0.dtype)

            def obj(th: JaxArray) -> JaxArray:
                return jnp.asarray(objective(th))

            def grad_val(th: JaxArray) -> JaxArray:
                return cast("JaxArray", jax.grad(obj)(th))

            def hess_val(th: JaxArray) -> JaxArray:
                return cast("JaxArray", jax.hessian(obj)(th))

            false_flag = jnp.zeros((), dtype=jnp.bool_)

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
                th_next = jnp.clip(th - damping * delta, b[:, 0], b[:, 1])
                return th_next, it + 1, converged_now

            th_fin, _it_fin, _cv = jax.lax.while_loop(
                cond,
                body,
                (theta0, jnp.asarray(0, dtype=jnp.int32), false_flag),
            )
            return th_fin

        return run_m

    if kind == "S":
        # Inline one-step Newton-like ascent
        b_arr = _normalize_bounds(bounds)
        damping = _kw_float(one_step_kwargs, "damping", 0.1)
        eps = _kw_float(one_step_kwargs, "eps", 1e-8)

        def run_s(
            theta0: JaxArray,
            objective: Callable[[JaxArray], JaxArray],
            _key: JaxArray,
        ) -> JaxArray:
            theta0 = jnp.asarray(theta0)
            b = b_arr.astype(theta0.dtype)

            def obj(th: JaxArray) -> JaxArray:
                return jnp.asarray(objective(th))

            g = cast("JaxArray", jax.grad(obj)(theta0))
            H = cast("JaxArray", jax.hessian(obj)(theta0))
            H_sym = 0.5 * (H + H.T)
            eye = jnp.eye(theta0.shape[0], dtype=theta0.dtype)
            H_safe = H_sym - eps * eye
            delta = jnp.linalg.solve(H_safe, g)
            theta_next = theta0 - damping * delta
            return jnp.clip(theta_next, b[:, 0], b[:, 1])

        return run_s

    if kind == "B":
        num_warmup = _kw_int(nuts_kwargs, "num_warmup", 500)
        num_samples = _kw_int(nuts_kwargs, "num_samples", 1000)

        def run_b(
            theta0: JaxArray, objective: Callable[[JaxArray], JaxArray], key: JaxArray
        ) -> JaxArray:
            # Build a transition for the given objective by closing over it
            # Only pass step parameters to transition builder; sampling counts used below
            step_size = _kw_float(nuts_kwargs, "step_size", 1e-1)
            max_num_doublings = _kw_int(nuts_kwargs, "max_num_doublings", 8)
            inv_mass = _kw_jax_array_opt(nuts_kwargs, "inverse_mass_matrix")

            def logprob(th: JaxArray) -> JaxArray:
                return jnp.asarray(objective(th))

            transition = build_bayes_transition(
                logprob,
                step_size=step_size,
                max_num_doublings=max_num_doublings,
                inverse_mass_matrix=inv_mass,
            )

            key, sub = jax.random.split(key)

            def one_step(
                theta_key: tuple[JaxArray, JaxArray], _unused: JaxArray
            ) -> tuple[tuple[JaxArray, JaxArray], JaxArray]:
                th, k = theta_key
                th_new, k_new = transition(th, k)
                return (th_new, k_new), th_new

            (theta_warm_end, key_after), _ = cast(
                "tuple[tuple[JaxArray, JaxArray], JaxArray]",
                jax.lax.scan(one_step, (theta0, sub), jnp.arange(num_warmup)),
            )
            (_theta_end, _), samples = cast(
                "tuple[tuple[JaxArray, JaxArray], JaxArray]",
                jax.lax.scan(one_step, (theta_warm_end, key_after), jnp.arange(num_samples)),
            )
            # take last or mean; choose mean for stability
            return jnp.mean(samples, axis=0)

        return run_b

    msg = f"Unknown estimator kind: {kind}"
    raise ValueError(msg)


def build_seed_runner(  # noqa: PLR0915, C901
    *,
    evaluator: LikelihoodEvaluator,
    model: DegenerateDiffusionProcess,
    plan: Mapping[int, tuple[EstimatorKind, EstimatorKind, EstimatorKind]],
    config: SeedRunnerConfig,
) -> Callable[[int, ThetaTriple], tuple[ThetaTriple, ThetaTriple]]:
    """Build a JIT-able function f(seed, theta_stage0_init).

    Returns a pair ``(theta_stage0_last, theta_final_last)``. Executes ``k = 1..k0``
    according to ``plan``, mirroring the stage-0/final arrangement from the
    imperative version.
    """
    sim_kernel = build_simulator_kernel(
        model, t_max=config.t_max, h=config.h, burn_out=config.burn_out, dt=config.dt
    )

    k0 = int(max(plan))

    # Prebuild stateless evaluators for each k in 1..k0 and wrap for lax.switch
    l1p_fns = tuple(
        evaluator.make_stateless_quasi_l1_prime_evaluator(k=k) for k in range(1, k0 + 1)
    )
    l1_fns = tuple(evaluator.make_stateless_quasi_l1_evaluator(k=k) for k in range(1, k0 + 1))
    l2_fns = tuple(evaluator.make_stateless_quasi_l2_evaluator(k=k) for k in range(1, k0 + 1))
    l3_fns = tuple(evaluator.make_stateless_quasi_l3_evaluator(k=k) for k in range(1, k0 + 1))

    def _wrap_eval(
        f: Callable[
            [JaxArray, JaxArray, JaxArray, JaxArray, JaxArray, JaxArray, JaxArray],
            JaxArray,
        ],
    ) -> Callable[
        [JaxArray, JaxArray, JaxArray, JaxArray, JaxArray, JaxArray, JaxArray],
        JaxArray,
    ]:
        def wrapped(
            th: JaxArray,
            t1b: JaxArray,
            t2b: JaxArray,
            t3b: JaxArray,
            xs: JaxArray,
            ys: JaxArray,
            hh: JaxArray,
            *,
            _f: Callable[
                [JaxArray, JaxArray, JaxArray, JaxArray, JaxArray, JaxArray, JaxArray],
                JaxArray,
            ] = f,
        ) -> JaxArray:
            return _f(th, t1b, t2b, t3b, xs, ys, hh)

        return wrapped

    branches_l1p = tuple(_wrap_eval(f) for f in l1p_fns)
    branches_l1 = tuple(_wrap_eval(f) for f in l1_fns)
    branches_l2 = tuple(_wrap_eval(f) for f in l2_fns)
    branches_l3 = tuple(_wrap_eval(f) for f in l3_fns)

    # Encode plan kinds as arrays for JAX-friendly selection
    kind_code = {"M": 0, "B": 1, "S": 2}
    kinds1 = jnp.array([0] * (k0 + 2))
    kinds2 = jnp.array([0] * (k0 + 2))
    kinds3 = jnp.array([0] * (k0 + 2))
    for kk, triple in plan.items():
        k_i = int(kk)
        kinds1 = kinds1.at[k_i].set(kind_code[triple[0]])
        kinds2 = kinds2.at[k_i].set(kind_code[triple[1]])
        kinds3 = kinds3.at[k_i].set(kind_code[triple[2]])

    # Prebuild solvers for each component and kind
    solver1_M = _build_component_solver(
        "M",
        config.bounds_theta1,
        newton_kwargs=config.newton_kwargs,
        nuts_kwargs=config.nuts_kwargs,
        one_step_kwargs=config.one_step_kwargs,
    )
    solver1_B = _build_component_solver(
        "B",
        config.bounds_theta1,
        newton_kwargs=config.newton_kwargs,
        nuts_kwargs=config.nuts_kwargs,
        one_step_kwargs=config.one_step_kwargs,
    )
    solver1_S = _build_component_solver(
        "S",
        config.bounds_theta1,
        newton_kwargs=config.newton_kwargs,
        nuts_kwargs=config.nuts_kwargs,
        one_step_kwargs=config.one_step_kwargs,
    )
    solver2_M = _build_component_solver(
        "M",
        config.bounds_theta2,
        newton_kwargs=config.newton_kwargs,
        nuts_kwargs=config.nuts_kwargs,
        one_step_kwargs=config.one_step_kwargs,
    )
    solver2_B = _build_component_solver(
        "B",
        config.bounds_theta2,
        newton_kwargs=config.newton_kwargs,
        nuts_kwargs=config.nuts_kwargs,
        one_step_kwargs=config.one_step_kwargs,
    )
    solver2_S = _build_component_solver(
        "S",
        config.bounds_theta2,
        newton_kwargs=config.newton_kwargs,
        nuts_kwargs=config.nuts_kwargs,
        one_step_kwargs=config.one_step_kwargs,
    )
    solver3_M = _build_component_solver(
        "M",
        config.bounds_theta3,
        newton_kwargs=config.newton_kwargs,
        nuts_kwargs=config.nuts_kwargs,
        one_step_kwargs=config.one_step_kwargs,
    )
    solver3_B = _build_component_solver(
        "B",
        config.bounds_theta3,
        newton_kwargs=config.newton_kwargs,
        nuts_kwargs=config.nuts_kwargs,
        one_step_kwargs=config.one_step_kwargs,
    )
    solver3_S = _build_component_solver(
        "S",
        config.bounds_theta3,
        newton_kwargs=config.newton_kwargs,
        nuts_kwargs=config.nuts_kwargs,
        one_step_kwargs=config.one_step_kwargs,
    )

    def runner(  # noqa: PLR0915, C901
        seed: int, theta_stage0_init: ThetaTriple
    ) -> tuple[ThetaTriple, ThetaTriple]:
        key = jax.random.PRNGKey(seed)
        # simulate
        d_x = model.x.shape[0]
        d_y = model.y.shape[0]
        x0 = jnp.zeros((d_x,))
        y0 = jnp.zeros((d_y,))
        x_series, y_series = sim_kernel(config.true_theta, key, x0, y0)

        # initialize stage-0 parameters
        stage0_prev = theta_stage0_init

        def body(  # noqa: PLR0915, C901
            carry: tuple[ThetaTriple, JaxArray],
            k: JaxArray,
        ) -> tuple[tuple[ThetaTriple, JaxArray], tuple[ThetaTriple, ThetaTriple]]:
            (theta_bar, key_in) = carry
            # objectives for this k selected via lax.switch
            idx0 = jnp.asarray(k - 1, dtype=jnp.int32)
            h_arr = jnp.asarray(config.h)

            def obj1p(th: JaxArray) -> JaxArray:
                return cast(
                    "JaxArray",
                    jax.lax.switch(
                        idx0,
                        branches_l1p,
                        th,
                        theta_bar[0],
                        theta_bar[1],
                        theta_bar[2],
                        x_series,
                        y_series,
                        h_arr,
                    ),
                )

            def obj1(th: JaxArray) -> JaxArray:
                return cast(
                    "JaxArray",
                    jax.lax.switch(
                        idx0,
                        branches_l1,
                        th,
                        theta_bar[0],
                        theta_bar[1],
                        theta_bar[2],
                        x_series,
                        y_series,
                        h_arr,
                    ),
                )

            def obj2(th: JaxArray) -> JaxArray:
                return cast(
                    "JaxArray",
                    jax.lax.switch(
                        idx0,
                        branches_l2,
                        th,
                        theta_bar[0],
                        theta_bar[1],
                        theta_bar[2],
                        x_series,
                        y_series,
                        h_arr,
                    ),
                )

            def obj3(th: JaxArray) -> JaxArray:
                return cast(
                    "JaxArray",
                    jax.lax.switch(
                        idx0,
                        branches_l3,
                        th,
                        theta_bar[0],
                        theta_bar[1],
                        theta_bar[2],
                        x_series,
                        y_series,
                        h_arr,
                    ),
                )

            # split keys per component and final
            key_in, k1, k2, k3, kf = jax.random.split(key_in, 5)

            idx = jnp.asarray(k, dtype=jnp.int32)
            kind1 = kinds1[idx]
            kind2 = kinds2[idx]
            kind3 = kinds3[idx]

            # Build per-component branch runners capturing objectives
            def run1_M(th: JaxArray, key_run: JaxArray) -> JaxArray:
                return solver1_M(th, obj1p, key_run)

            def run1_B(th: JaxArray, key_run: JaxArray) -> JaxArray:
                return solver1_B(th, obj1p, key_run)

            def run1_S(th: JaxArray, key_run: JaxArray) -> JaxArray:
                return solver1_S(th, obj1p, key_run)

            branches1_local: tuple[BranchRunner, ...] = (run1_M, run1_B, run1_S)

            def run2_M(th: JaxArray, key_run: JaxArray) -> JaxArray:
                return solver2_M(th, obj2, key_run)

            def run2_B(th: JaxArray, key_run: JaxArray) -> JaxArray:
                return solver2_B(th, obj2, key_run)

            def run2_S(th: JaxArray, key_run: JaxArray) -> JaxArray:
                return solver2_S(th, obj2, key_run)

            branches2_local: tuple[BranchRunner, ...] = (run2_M, run2_B, run2_S)

            def run3_M(th: JaxArray, key_run: JaxArray) -> JaxArray:
                return solver3_M(th, obj3, key_run)

            def run3_B(th: JaxArray, key_run: JaxArray) -> JaxArray:
                return solver3_B(th, obj3, key_run)

            def run3_S(th: JaxArray, key_run: JaxArray) -> JaxArray:
                return solver3_S(th, obj3, key_run)

            branches3_local: tuple[BranchRunner, ...] = (run3_M, run3_B, run3_S)

            theta1_k0 = cast(
                "JaxArray",
                jax.lax.switch(kind1, branches1_local, theta_bar[0], k1),
            )
            theta2_k0 = cast(
                "JaxArray",
                jax.lax.switch(kind2, branches2_local, theta_bar[1], k2),
            )
            theta3_k0 = cast(
                "JaxArray",
                jax.lax.switch(kind3, branches3_local, theta_bar[2], k3),
            )

            stage0_now = (theta1_k0, theta2_k0, theta3_k0)

            # final stage adjustments
            kind3_next = kinds3[idx + 1]
            # If k==1 use estimator at k+1 for theta3; else keep stage0
            use_update = jnp.equal(idx, 1)

            max_idx = jnp.asarray(k0 - 1, dtype=jnp.int32)
            idx_next = jnp.minimum(idx0 + 1, max_idx)

            def obj3_next(th: JaxArray) -> JaxArray:
                return cast(
                    "JaxArray",
                    jax.lax.switch(
                        idx_next,
                        branches_l3,
                        th,
                        theta_bar[0],
                        theta_bar[1],
                        theta_bar[2],
                        x_series,
                        y_series,
                        h_arr,
                    ),
                )

            def do_update(_: tuple[JaxArray, JaxArray, JaxArray]) -> JaxArray:
                def run3n_M(th: JaxArray, key_run: JaxArray) -> JaxArray:
                    return solver3_M(th, obj3_next, key_run)

                def run3n_B(th: JaxArray, key_run: JaxArray) -> JaxArray:
                    return solver3_B(th, obj3_next, key_run)

                def run3n_S(th: JaxArray, key_run: JaxArray) -> JaxArray:
                    return solver3_S(th, obj3_next, key_run)

                branches3_next: tuple[BranchRunner, ...] = (run3n_M, run3n_B, run3n_S)
                return cast(
                    "JaxArray",
                    jax.lax.switch(kind3_next, branches3_next, stage0_now[2], kf),
                )

            def keep_stage0(_: tuple[JaxArray, JaxArray, JaxArray]) -> JaxArray:
                return stage0_now[2]

            theta3_final = cast(
                "JaxArray",
                jax.lax.cond(use_update, do_update, keep_stage0, stage0_now),
            )

            theta_bar_final = (stage0_now[0], stage0_now[1], theta3_final)
            kind1_next = kinds1[idx + 1]

            def obj1_next(th: JaxArray) -> JaxArray:
                return cast(
                    "JaxArray",
                    jax.lax.switch(
                        idx_next,
                        branches_l1,
                        th,
                        theta_bar[0],
                        theta_bar[1],
                        theta_bar[2],
                        x_series,
                        y_series,
                        h_arr,
                    ),
                )

            def run1n_M(th: JaxArray, key_run: JaxArray) -> JaxArray:
                return solver1_M(th, obj1_next, key_run)

            def run1n_B(th: JaxArray, key_run: JaxArray) -> JaxArray:
                return solver1_B(th, obj1_next, key_run)

            def run1n_S(th: JaxArray, key_run: JaxArray) -> JaxArray:
                return solver1_S(th, obj1_next, key_run)

            branches1_next: tuple[BranchRunner, ...] = (run1n_M, run1n_B, run1n_S)
            theta1_final = cast(
                "JaxArray",
                jax.lax.switch(kind1_next, branches1_next, theta_bar_final[0], kf),
            )
            theta2_final = stage0_now[1]

            next_theta_bar = (theta1_final, theta2_final, theta3_final)
            next_key = key_in
            return (next_theta_bar, next_key), (stage0_now, next_theta_bar)

        init_carry = (stage0_prev, key)
        ks = jnp.arange(1, k0 + 1, dtype=jnp.int32)
        last_carry, outs = cast(
            "tuple[tuple[ThetaTriple, JaxArray], tuple[ThetaTriple, ThetaTriple]]",
            jax.lax.scan(body, init_carry, ks),
        )
        last_stage0_all, last_final_all = outs
        # take last along time
        last_stage0 = (
            last_stage0_all[0][-1],
            last_stage0_all[1][-1],
            last_stage0_all[2][-1],
        )
        last_final = (
            last_final_all[0][-1],
            last_final_all[1][-1],
            last_final_all[2][-1],
        )
        _ = last_carry  # silence lints
        return last_stage0, last_final

    return jax.jit(runner)

"""JAX-compiled per-seed loop runner building on stateless evaluators.

This module provides a seed runner that performs:
1) Simulation with model.make_simulation_kernel
2) Alternating estimation for theta1/theta2/theta3 with M/B/S choices

It mirrors LoopEstimationAlgorithm._run_single_seed but returns a pure function
that can be jitted and later vmapped over seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import jax
import jax.numpy as jnp

from degenerate_diffusion.estimation.parameter_estimator import (
    Bounds as JaxBounds,
    JaxArray,
    build_b,
    build_m,
    build_s,
)

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from collections.abc import Callable, Mapping

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


@dataclass(frozen=True)
class ComponentSolvers:
    """Bundle of per-kind solvers for a single theta component.

    m: (theta, aux) -> theta
    s: (theta, aux) -> theta
    b: (theta, key, aux) -> theta  (Bayesian sampler returning mean)
    """

    m: Callable[[JaxArray, tuple[JaxArray, ThetaTriple, JaxArray, JaxArray, JaxArray]], JaxArray]
    s: Callable[[JaxArray, tuple[JaxArray, ThetaTriple, JaxArray, JaxArray, JaxArray]], JaxArray]
    b: Callable[
        [JaxArray, JaxArray, tuple[JaxArray, ThetaTriple, JaxArray, JaxArray, JaxArray]], JaxArray
    ]


def _build_component_solvers(
    objective_with_aux: Callable[
        [JaxArray, tuple[JaxArray, ThetaTriple, JaxArray, JaxArray, JaxArray]], JaxArray
    ],
    bounds: JaxBounds,
    *,
    newton_kwargs: Mapping[str, object],
    one_step_kwargs: Mapping[str, object],
    nuts_kwargs: Mapping[str, object],
) -> ComponentSolvers:
    """Construct M/S/B solvers for a single component from config and objective.

    Returns a ComponentSolvers bundle with .m/.s/.b callables.
    """
    m = build_m(
        objective_with_aux,
        bounds,
        max_iters=_kw_int(newton_kwargs, "max_iters", 100),
        tol=_kw_float(newton_kwargs, "tol", 1e-6),
        damping=_kw_float(newton_kwargs, "damping", 0.1),
        use_adam_fallback=bool(newton_kwargs.get("use_adam_fallback", True)),
        learning_rate=_kw_float(newton_kwargs, "learning_rate", 1e-5),
        weight_decay=_kw_float(newton_kwargs, "weight_decay", 0.0),
        clip_norm=_kw_float(newton_kwargs, "clip_norm", 1e2),
        eps=_kw_float(newton_kwargs, "eps", 1e-8),
    )
    s = build_s(
        objective_with_aux,
        bounds,
        damping=_kw_float(one_step_kwargs, "damping", 1),
        eps=_kw_float(one_step_kwargs, "eps", 1e-8),
    )
    b = build_b(
        objective_with_aux,
        bounds,
        step_size=_kw_float(nuts_kwargs, "step_size", 1e-1),
        inverse_mass_matrix=_kw_jax_array_opt(nuts_kwargs, "inverse_mass_matrix"),
        num_warmup=_kw_int(nuts_kwargs, "num_warmup", 1000),
        num_samples=_kw_int(nuts_kwargs, "num_samples", 3000),
        thin=_kw_int(nuts_kwargs, "thin", 1),
        target_acceptance_rate=_kw_float(nuts_kwargs, "target_acceptance_rate", 0.75),
    )
    return ComponentSolvers(m=m, s=s, b=b)


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


def _make_objectives_with_aux(
    branches_l1p: tuple[Callable[..., JaxArray], ...],
    branches_l1: tuple[Callable[..., JaxArray], ...],
    branches_l2: tuple[Callable[..., JaxArray], ...],
    branches_l3: tuple[Callable[..., JaxArray], ...],
) -> tuple[
    Callable[[JaxArray, tuple[JaxArray, ThetaTriple, JaxArray, JaxArray, JaxArray]], JaxArray],
    Callable[[JaxArray, tuple[JaxArray, ThetaTriple, JaxArray, JaxArray, JaxArray]], JaxArray],
    Callable[[JaxArray, tuple[JaxArray, ThetaTriple, JaxArray, JaxArray, JaxArray]], JaxArray],
    Callable[[JaxArray, tuple[JaxArray, ThetaTriple, JaxArray, JaxArray, JaxArray]], JaxArray],
]:
    """Aux-aware objectives selecting k via lax.switch.

    aux = (idx0, theta_bar, x_series, y_series, h_arr)
    """

    def wrap(
        branches: tuple[Callable[..., JaxArray], ...],
    ) -> Callable[[JaxArray, tuple[JaxArray, ThetaTriple, JaxArray, JaxArray, JaxArray]], JaxArray]:
        def objective(
            th: JaxArray,
            aux: tuple[JaxArray, ThetaTriple, JaxArray, JaxArray, JaxArray],
        ) -> JaxArray:
            idx0, theta_bar, x_series, y_series, h_arr = aux
            return cast(
                "JaxArray",
                jax.lax.switch(
                    idx0,
                    branches,
                    th,
                    theta_bar[0],
                    theta_bar[1],
                    theta_bar[2],
                    x_series,
                    y_series,
                    h_arr,
                ),
            )

        return objective

    return wrap(branches_l1p), wrap(branches_l1), wrap(branches_l2), wrap(branches_l3)


# (removed) _unused_placeholder


def _run_by_kind(
    kind_code_scalar: JaxArray,
    theta0: JaxArray,
    key: JaxArray,
    aux: tuple[JaxArray, ThetaTriple, JaxArray, JaxArray, JaxArray],
    *,
    solver_M: Callable[
        [JaxArray, tuple[JaxArray, ThetaTriple, JaxArray, JaxArray, JaxArray]], JaxArray
    ],
    trans_B: Callable[
        [JaxArray, JaxArray, tuple[JaxArray, ThetaTriple, JaxArray, JaxArray, JaxArray]], JaxArray
    ],
    solver_S: Callable[
        [JaxArray, tuple[JaxArray, ThetaTriple, JaxArray, JaxArray, JaxArray]], JaxArray
    ],
) -> JaxArray:
    """Run one component update by estimator kind.

    - kind 0: M -> solver_M(theta, aux)
    - kind 1: B -> trans_B(theta, key, aux) (sampler returning mean)
    - kind 2: S -> solver_S(theta, aux)
    """
    branches: tuple[Callable[[JaxArray, JaxArray], JaxArray], ...] = (
        lambda th, _k: solver_M(th, aux),
        lambda th, k_in: trans_B(th, k_in, aux),
        lambda th, _k: solver_S(th, aux),
    )
    return cast("JaxArray", jax.lax.switch(kind_code_scalar, branches, theta0, key))


def build_seed_runner(  # noqa: PLR0915
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
    sim_kernel = model.make_simulation_kernel(
        t_max=config.t_max,
        h=config.h,
        burn_out=config.burn_out,
        dt=config.dt,
    )

    k0 = int(max(plan)) - 1

    # Prebuild stateless evaluators for each k in 1..k0 and wrap for lax.switch
    branches_l1p = tuple(
        evaluator.make_stateless_quasi_l1_prime_evaluator(k=k) for k in range(1, k0 + 1)
    )
    branches_l1 = tuple(evaluator.make_stateless_quasi_l1_evaluator(k=k) for k in range(1, k0 + 2))
    branches_l2 = tuple(evaluator.make_stateless_quasi_l2_evaluator(k=k) for k in range(1, k0 + 1))
    branches_l3 = tuple(evaluator.make_stateless_quasi_l3_evaluator(k=k) for k in range(1, k0 + 1))

    # Encode plan kinds as arrays for JAX-friendly selection
    kind_code = {"M": 0, "B": 1, "S": 2}
    kinds1 = jnp.zeros((k0 + 2,), dtype=jnp.int32)
    kinds2 = jnp.zeros((k0 + 2,), dtype=jnp.int32)
    kinds3 = jnp.zeros((k0 + 2,), dtype=jnp.int32)
    for kk, triple in plan.items():
        k_i = int(kk)
        kinds1 = kinds1.at[k_i].set(kind_code[triple[0]])
        kinds2 = kinds2.at[k_i].set(kind_code[triple[1]])
        kinds3 = kinds3.at[k_i].set(kind_code[triple[2]])

    # Build aux-aware objectives once
    obj1p_aux, obj1_aux, obj2_aux, obj3_aux = _make_objectives_with_aux(
        branches_l1p, branches_l1, branches_l2, branches_l3
    )

    # Prebuild solver bundles per component using aux-based builders
    comps1p = _build_component_solvers(
        obj1p_aux,
        config.bounds_theta1,
        newton_kwargs=config.newton_kwargs,
        one_step_kwargs=config.one_step_kwargs,
        nuts_kwargs=config.nuts_kwargs,
    )
    comps1 = _build_component_solvers(
        obj1_aux,
        config.bounds_theta1,
        newton_kwargs=config.newton_kwargs,
        one_step_kwargs=config.one_step_kwargs,
        nuts_kwargs=config.nuts_kwargs,
    )
    comps2 = _build_component_solvers(
        obj2_aux,
        config.bounds_theta2,
        newton_kwargs=config.newton_kwargs,
        one_step_kwargs=config.one_step_kwargs,
        nuts_kwargs=config.nuts_kwargs,
    )
    comps3 = _build_component_solvers(
        obj3_aux,
        config.bounds_theta3,
        newton_kwargs=config.newton_kwargs,
        one_step_kwargs=config.one_step_kwargs,
        nuts_kwargs=config.nuts_kwargs,
    )

    def runner(
        seed: int,
        theta_stage0_init: ThetaTriple,
    ) -> tuple[JaxArray, JaxArray, JaxArray, JaxArray]:
        key = jax.random.PRNGKey(seed)
        # simulate
        d_x = model.x.shape[0]
        d_y = model.y.shape[0]
        x0 = jnp.zeros((d_x,))
        y0 = jnp.zeros((d_y,))
        x_series, y_series = sim_kernel(config.true_theta, key, x0, y0)

        # Common constants
        h_arr_const = jnp.asarray(config.h)

        def body(
            carry: tuple[ThetaTriple, JaxArray],
            k: JaxArray,
        ) -> tuple[tuple[ThetaTriple, JaxArray], tuple[JaxArray, JaxArray, JaxArray, JaxArray]]:
            (theta_bar, key_in) = carry
            # objectives for this k selected via lax.switch
            k = jnp.asarray(k, dtype=jnp.int32)
            idx = k - 1
            # aux payload passed to aux-aware objectives/solvers
            aux = (idx, theta_bar, x_series, y_series, h_arr_const)

            # split keys per component and final
            key_in, k1, k2, k3, k3f, k1f = jax.random.split(key_in, 6)

            kind1 = kinds1[k]
            kind2 = kinds2[k]
            kind3 = kinds3[k]

            # Use common runner for three components
            theta1_k0 = _run_by_kind(
                kind1,
                theta_bar[0],
                k1,
                aux,
                solver_M=comps1p.m,
                solver_S=comps1p.s,
                trans_B=comps1p.b,
            )
            theta_bar = (theta1_k0, theta_bar[1], theta_bar[2])
            aux = (idx, theta_bar, x_series, y_series, h_arr_const)
            theta2_k0 = _run_by_kind(
                kind2,
                theta_bar[1],
                k2,
                aux,
                solver_M=comps2.m,
                solver_S=comps2.s,
                trans_B=comps2.b,
            )
            theta_bar = (theta1_k0, theta2_k0, theta_bar[2])
            aux = (idx, theta_bar, x_series, y_series, h_arr_const)
            theta3_k0 = _run_by_kind(
                kind3,
                theta_bar[2],
                k3,
                aux,
                solver_M=comps3.m,
                solver_S=comps3.s,
                trans_B=comps3.b,
            )

            theta_bar = (theta1_k0, theta2_k0, theta3_k0)
            aux = (idx, theta_bar, x_series, y_series, h_arr_const)

            # final stage adjustments
            kind3_next = kinds3[k + 1]
            # If k==1 use estimator at k+1 for theta3; else keep stage0
            use_update = jnp.equal(k, 1)

            def do_update(_: tuple[JaxArray, JaxArray, JaxArray]) -> JaxArray:
                return _run_by_kind(
                    kind3_next,
                    theta_bar[2],
                    k3f,
                    aux,
                    solver_M=comps3.m,
                    solver_S=comps3.s,
                    trans_B=comps3.b,
                )

            def keep_stage0(_: tuple[JaxArray, JaxArray, JaxArray]) -> JaxArray:
                return theta_bar[2]

            theta3_k0 = cast(
                "JaxArray",
                jax.lax.cond(use_update, do_update, keep_stage0, theta_bar),
            )

            theta_bar = (theta_bar[0], theta_bar[1], theta3_k0)
            aux = (idx + 1, theta_bar, x_series, y_series, h_arr_const)

            kind1_next = kinds1[k + 1]

            theta1_final = _run_by_kind(
                kind1_next,
                theta_bar[0],
                k1f,
                aux,
                solver_M=comps1.m,
                solver_S=comps1.s,
                trans_B=comps1.b,
            )
            result = (theta_bar[0], theta1_final, theta_bar[1], theta_bar[2])
            next_key = key_in
            return (theta_bar, next_key), result

        init_carry = (theta_stage0_init, key)
        ks = jnp.arange(1, k0 + 1, dtype=jnp.int32)
        _last_carry, outs = cast(
            "tuple[tuple[ThetaTriple, JaxArray], tuple[JaxArray, JaxArray, JaxArray, JaxArray]]",
            jax.lax.scan(body, init_carry, ks),
        )
        return outs

    return jax.jit(runner)

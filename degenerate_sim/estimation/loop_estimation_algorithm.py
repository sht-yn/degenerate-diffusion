"""Iterative estimation loop driven by likelihood evaluators."""

from __future__ import annotations

import typing as _typing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
import numpy as np

from degenerate_sim.estimation.parameter_estimator import (
    Array1D,
    Bounds,
    bayes_estimate,
    newton_solve,
    one_step_estimate,
)

if TYPE_CHECKING:
    import collections.abc as cabc

    from degenerate_sim.evaluation.likelihood_evaluator_jax import LikelihoodEvaluator
else:  # pragma: no cover - runtime fallback for type-only imports
    cabc = _typing

EstimatorKind = Literal["M", "B", "S"]

THETA_COMPONENT_1 = 1
THETA_COMPONENT_2 = 2
THETA_COMPONENT_3 = 3


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for simulating observations per seed."""

    true_theta: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    t_max: float
    h: float
    burn_out: float
    dt: float = 0.001
    x0: np.ndarray | jnp.ndarray | None = None
    y0: np.ndarray | jnp.ndarray | None = None


@dataclass(frozen=True)
class Observation:
    """Observed time series generated for each seed."""

    x_series: jnp.ndarray
    y_series: jnp.ndarray
    h: float


@dataclass(frozen=True)
class IterationEstimate:
    """Intermediate and final estimates obtained in each iteration."""

    k: int
    theta_stage0: tuple[np.ndarray, np.ndarray, np.ndarray]
    theta_final: tuple[np.ndarray, np.ndarray, np.ndarray]


class LoopEstimationAlgorithm:
    """Implements the two-stage iterative estimation procedure.

    The algorithm follows the recursive construction described in the
    specification, automatically generating observations by simulating from the
    diffusion model associated with the provided :class:`LikelihoodEvaluator`.
    """

    def __init__(
        self,
        evaluator: LikelihoodEvaluator,
        simulation_config: SimulationConfig,
        bounds_theta1: Bounds,
        bounds_theta2: Bounds,
        bounds_theta3: Bounds,
        *,
        estimator_kwargs: cabc.Mapping[EstimatorKind, dict[str, object]] | None = None,
    ) -> None:
        """Store evaluator, simulation settings, parameter bounds, and options."""
        self._likelihood = evaluator
        self._model = evaluator.model
        self._simulation = simulation_config
        self._bounds = (bounds_theta1, bounds_theta2, bounds_theta3)
        default_kwargs: dict[EstimatorKind, dict[str, object]] = {
            "M": {
                "max_iters": 50,
                "tol": 1e-6,
                "damping": 1.0,
                "log_interval": None,
            },
            "B": {
                "prior_log_pdf": None,
                "num_warmup": 1_000,
                "num_samples": 2_000,
                "num_chains": 1,
                "rng_seed": 0,
            },
            "S": {},
        }
        if estimator_kwargs:
            for key, value in estimator_kwargs.items():
                default_kwargs[key] = {**default_kwargs.get(key, {}), **value}
        self._estimator_kwargs = default_kwargs

    def run(
        self,
        seeds: cabc.Iterable[int],
        plan: cabc.Mapping[int, tuple[EstimatorKind, EstimatorKind, EstimatorKind]],
        max_iteration: int,
        initial_theta_stage0: tuple[Array1D, Array1D, Array1D],
    ) -> dict[int, cabc.Sequence[IterationEstimate]]:
        """Execute the iterative estimation scheme for each seed.

        Args:
            seeds: Iterable of random seeds used for simulation and estimation.
            plan: Mapping from iteration index ``k`` to the tuple of estimator
                kinds ``(A_1^k, A_2^k, A_3^k)`` where each element is ``"M"``,
                ``"B"`` or ``"S"``.
            max_iteration: Maximum iteration index ``K`` for which estimates are
                produced.
            initial_theta_stage0: Initial measurable function
                ``hat{theta}_n^{0,0}`` used before the first iteration.

        Returns:
            A dictionary mapping each seed to a sequence of
            :class:`IterationEstimate` instances containing stage-0 and final
            estimates for every iteration.

        """
        results: dict[int, list[IterationEstimate]] = {}
        for seed in seeds:
            observation = self._simulate_observation(seed)
            stage0_prev = tuple(np.asarray(theta) for theta in initial_theta_stage0)
            iterations: list[IterationEstimate] = []
            for k in range(1, max_iteration + 1):
                if k not in plan:
                    msg = f"Plan does not contain estimator triple for k={k}."
                    raise KeyError(msg)
                theta1_k0 = self._estimate_component(
                    estimator_kind=plan[k][0],
                    component=1,
                    k_arg=k,
                    observation=observation,
                    theta_bar=stage0_prev,
                    initial_guess=stage0_prev[0],
                    use_prime_objective=True,
                )
                theta2_k0 = self._estimate_component(
                    estimator_kind=plan[k][1],
                    component=2,
                    k_arg=k,
                    observation=observation,
                    theta_bar=(theta1_k0, stage0_prev[1], stage0_prev[2]),
                    initial_guess=stage0_prev[1],
                )
                theta3_k0 = self._estimate_component(
                    estimator_kind=plan[k][2],
                    component=3,
                    k_arg=k,
                    observation=observation,
                    theta_bar=(theta1_k0, theta2_k0, stage0_prev[2]),
                    initial_guess=stage0_prev[2],
                )
                stage0_curr = tuple(np.asarray(v) for v in (theta1_k0, theta2_k0, theta3_k0))
                if (k + 1) not in plan:
                    msg = f"Plan does not contain estimator triple for k+1={k + 1}."
                    raise KeyError(msg)
                theta1_final = self._estimate_component(
                    estimator_kind=plan[k + 1][0],
                    component=1,
                    k_arg=k + 1,
                    observation=observation,
                    theta_bar=stage0_curr,
                    initial_guess=stage0_curr[0],
                    use_prime_objective=False,
                )
                theta2_final = stage0_curr[1]
                if k == 1:
                    theta3_final = self._estimate_component(
                        estimator_kind=plan[2][2],
                        component=3,
                        k_arg=1,
                        observation=observation,
                        theta_bar=stage0_curr,
                        initial_guess=stage0_curr[2],
                    )
                else:
                    theta3_final = stage0_curr[2]
                iterations.append(
                    IterationEstimate(
                        k=k,
                        theta_stage0=stage0_curr,
                        theta_final=(
                            np.asarray(theta1_final),
                            np.asarray(theta2_final),
                            np.asarray(theta3_final),
                        ),
                    )
                )
                stage0_prev = stage0_curr
            results[seed] = iterations
        return results

    def _simulate_observation(self, seed: int) -> Observation:
        cfg = self._simulation
        x_series, y_series = self._model.simulate(
            cfg.true_theta,
            t_max=cfg.t_max,
            h=cfg.h,
            burn_out=cfg.burn_out,
            seed=seed,
            x0=cfg.x0,
            y0=cfg.y0,
            dt=cfg.dt,
        )
        return Observation(
            x_series=jnp.asarray(x_series),
            y_series=jnp.asarray(y_series),
            h=cfg.h,
        )

    def _estimate_component(
        self,
        *,
        estimator_kind: EstimatorKind,
        component: int,
        k_arg: int,
        observation: Observation,
        theta_bar: tuple[Array1D, Array1D, Array1D],
        initial_guess: Array1D,
        use_prime_objective: bool = False,
    ) -> np.ndarray:
        objective = self._make_objective(
            component=component,
            k_arg=k_arg,
            observation=observation,
            theta_bar=theta_bar,
            use_prime=use_prime_objective,
        )
        bounds = self._bounds[component - 1]
        init = np.asarray(initial_guess)
        if estimator_kind == "M":
            kwargs = self._estimator_kwargs["M"]
            return np.asarray(
                newton_solve(
                    objective_function=objective,
                    search_bounds=bounds,
                    initial_guess=init,
                    **kwargs,
                )
            )
        if estimator_kind == "B":
            kwargs = self._estimator_kwargs["B"]
            return np.asarray(
                bayes_estimate(
                    objective_function=objective,
                    search_bounds=bounds,
                    initial_guess=init,
                    **kwargs,
                )
            )
        if estimator_kind == "S":
            kwargs = self._estimator_kwargs["S"]
            if kwargs:
                msg = "one_step_estimate does not accept keyword overrides."
                raise ValueError(msg)
            return np.asarray(
                one_step_estimate(
                    objective_function=objective,
                    search_bounds=bounds,
                    initial_estimator=init,
                )
            )
        msg = f"Unsupported estimator kind: {estimator_kind}"
        raise ValueError(msg)

    def _make_objective(
        self,
        *,
        component: int,
        k_arg: int,
        observation: Observation,
        theta_bar: tuple[Array1D, Array1D, Array1D],
        use_prime: bool,
    ) -> cabc.Callable[[Array1D], jnp.ndarray]:
        x_series, y_series, h = observation.x_series, observation.y_series, observation.h
        theta_bar_jnp = tuple(jnp.asarray(v) for v in theta_bar)
        if component == THETA_COMPONENT_1:
            evaluator_fn = (
                self._likelihood.make_quasi_likelihood_l1_prime_evaluator(
                    x_series, y_series, h, k_arg
                )
                if use_prime
                else self._likelihood.make_quasi_likelihood_l1_evaluator(
                    x_series, y_series, h, k_arg
                )
            )

            def objective(theta_val: Array1D) -> jnp.ndarray:
                return evaluator_fn(jnp.asarray(theta_val), *theta_bar_jnp)

            return objective
        if component == THETA_COMPONENT_2:
            evaluator_fn = self._likelihood.make_quasi_likelihood_l2_evaluator(
                x_series, y_series, h, k_arg
            )

            def objective(theta_val: Array1D) -> jnp.ndarray:
                return evaluator_fn(jnp.asarray(theta_val), *theta_bar_jnp)

            return objective
        if component == THETA_COMPONENT_3:
            evaluator_fn = self._likelihood.make_quasi_likelihood_l3_evaluator(
                x_series, y_series, h, k_arg
            )

            def objective(theta_val: Array1D) -> jnp.ndarray:
                return evaluator_fn(jnp.asarray(theta_val), *theta_bar_jnp)

            return objective
        msg = f"Unknown component index: {component}"
        raise ValueError(msg)

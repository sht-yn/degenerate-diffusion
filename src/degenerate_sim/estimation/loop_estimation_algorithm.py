"""Iterative estimation loop driven by likelihood evaluators."""

from __future__ import annotations

import csv
import os
import re
import typing as _typing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cloudpickle
import jax.numpy as jnp
import sympy as sp
from tqdm.auto import tqdm

from degenerate_sim.estimation.parameter_estimator import (
    Array1D,
    Bounds,
    bayes_estimate,
    newton_solve,
    one_step_estimate,
)

if TYPE_CHECKING:
    import collections.abc as cabc

    import numpy as np

    from degenerate_sim.evaluation.likelihood_evaluator_jax import LikelihoodEvaluator
else:  # pragma: no cover - runtime fallback for type-only imports
    cabc = _typing

EstimatorKind = Literal["M", "B", "S"]

TRUTH_PHASE = "__truth__"
META_PHASE = "__meta__"

THETA_COMPONENT_1 = 1
THETA_COMPONENT_2 = 2
THETA_COMPONENT_3 = 3


_PROCESS_POOL_STATE: dict[str, object] = {}


def _ensure_tqdm_notebook_fallback() -> None:
    """Force tqdm to fall back to text mode when notebook widgets misbehave."""
    if "TQDM_NOTEBOOK_FALLBACK" not in os.environ:
        os.environ["TQDM_NOTEBOOK_FALLBACK"] = "1"


def _build_progress_bar(
    *,
    iterable: cabc.Iterable[int] | None = None,
    total: int | None = None,
    desc: str,
    unit: str,
    position: int | None,
    leave: bool,
) -> tqdm:
    _ensure_tqdm_notebook_fallback()
    if iterable is not None:
        return tqdm(iterable, desc=desc, unit=unit, position=position, leave=leave)
    return tqdm(total=total, desc=desc, unit=unit, position=position, leave=leave)


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
    theta_stage0: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    theta_final: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]


def _process_pool_initializer(serialized_algorithm: bytes) -> None:
    _PROCESS_POOL_STATE["algorithm"] = cloudpickle.loads(serialized_algorithm)


def _process_pool_run_single_seed(
    seed: int,
    plan: dict[int, tuple[EstimatorKind, EstimatorKind, EstimatorKind]],
    k_0: int,
    initial_theta_stage0: tuple[Array1D, Array1D, Array1D],
    *,
    my_setting: bool,
) -> list[IterationEstimate]:
    algorithm: LoopEstimationAlgorithm = _PROCESS_POOL_STATE["algorithm"]
    return algorithm.run_single_seed(
        seed=seed,
        plan=plan,
        k_0=k_0,
        initial_theta_stage0=initial_theta_stage0,
        my_setting=my_setting,
    )


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
            "M": {"max_iters": 1000, "tol": 1e-6, "damping": 0.5, "log_interval": None},
            "B": {
                "prior_log_pdf": None,
                "num_warmup": 1_000,
                "num_samples": 3_000,
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
        k_0: int,
        initial_theta_stage0: tuple[Array1D, Array1D, Array1D],
        *,
        my_setting: bool = True,
        show_progress: bool = False,
        progress_desc: str | None = None,
        progress_position: int | None = None,
        num_workers: int | None = None,
    ) -> dict[int, cabc.Sequence[IterationEstimate]]:
        """Execute the iterative estimation scheme for each seed.

        Args:
            seeds: Iterable of random seeds used for simulation and estimation.
            plan: Mapping from iteration index ``k`` to the tuple of estimator
                kinds ``(A_1^k, A_2^k, A_3^k)`` where each element is ``"M"``,
                ``"B"`` or ``"S"``.
            k_0: Maximum iteration index ``k_0`` for which estimates are
                produced.
            initial_theta_stage0: Initial measurable function
                ``hat{theta}_n^{0,0}`` used before the first iteration.
            my_setting: Whether to evaluate likelihood components using the
                alternative configuration passed through to the evaluator.
            show_progress: Whether to display a progress bar over the provided
                seeds.
            progress_desc: Optional custom description shown alongside the
                progress bar.
            progress_position: Optional row position for nested progress bars.
            num_workers: When greater than one, parallelises the per-seed
                computations using a thread pool with the specified maximum
                workers. A value of ``None`` or ``1`` preserves sequential
                execution.

        Returns:
            A dictionary mapping each seed to a sequence of
            :class:`IterationEstimate` instances containing stage-0 and final
            estimates for every iteration.

        """
        seeds_sequence = list(seeds)
        if not seeds_sequence:
            return {}

        use_process_parallel = (
            num_workers is not None and num_workers > 1 and _contains_bayesian_estimators(plan)
        )

        if use_process_parallel:
            return self._run_parallel_process(
                seeds_sequence,
                plan,
                k_0,
                initial_theta_stage0,
                my_setting=my_setting,
                show_progress=show_progress,
                progress_desc=progress_desc,
                progress_position=progress_position,
                num_workers=num_workers,
            )

        use_thread_parallel = num_workers is not None and num_workers > 1
        if use_thread_parallel:
            return self._run_parallel(
                seeds_sequence,
                plan,
                k_0,
                initial_theta_stage0,
                my_setting=my_setting,
                show_progress=show_progress,
                progress_desc=progress_desc,
                progress_position=progress_position,
                num_workers=num_workers,
            )
        return self._run_sequential(
            seeds_sequence,
            plan,
            k_0,
            initial_theta_stage0,
            my_setting=my_setting,
            show_progress=show_progress,
            progress_desc=progress_desc,
            progress_position=progress_position,
        )

    def _run_sequential(
        self,
        seeds_sequence: cabc.Sequence[int],
        plan: cabc.Mapping[int, tuple[EstimatorKind, EstimatorKind, EstimatorKind]],
        k_0: int,
        initial_theta_stage0: tuple[Array1D, Array1D, Array1D],
        *,
        my_setting: bool,
        show_progress: bool,
        progress_desc: str | None,
        progress_position: int | None,
    ) -> dict[int, list[IterationEstimate]]:
        results: dict[int, list[IterationEstimate]] = {}
        iterator: cabc.Iterable[int]
        progress_bar: tqdm | None = None
        if show_progress:
            progress_bar = _build_progress_bar(
                iterable=seeds_sequence,
                desc=progress_desc or "Estimating seeds",
                unit="seed",
                position=progress_position,
                leave=False,
            )
            iterator = progress_bar
        else:
            iterator = seeds_sequence

        try:
            for seed in iterator:
                results[seed] = self._run_single_seed(
                    seed=seed,
                    plan=plan,
                    k_0=k_0,
                    initial_theta_stage0=initial_theta_stage0,
                    my_setting=my_setting,
                )
        finally:
            if progress_bar is not None:
                progress_bar.close()
        return results

    def _run_parallel(
        self,
        seeds_sequence: cabc.Sequence[int],
        plan: cabc.Mapping[int, tuple[EstimatorKind, EstimatorKind, EstimatorKind]],
        k_0: int,
        initial_theta_stage0: tuple[Array1D, Array1D, Array1D],
        *,
        my_setting: bool,
        show_progress: bool,
        progress_desc: str | None,
        progress_position: int | None,
        num_workers: int,
    ) -> dict[int, list[IterationEstimate]]:
        progress_bar: tqdm | None = None
        if show_progress:
            progress_bar = _build_progress_bar(
                total=len(seeds_sequence),
                desc=progress_desc or "Estimating seeds",
                unit="seed",
                position=progress_position,
                leave=False,
            )

        results_parallel: dict[int, list[IterationEstimate]] = {}
        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_seed = {
                    executor.submit(
                        self._run_single_seed,
                        seed=seed,
                        plan=plan,
                        k_0=k_0,
                        initial_theta_stage0=initial_theta_stage0,
                        my_setting=my_setting,
                    ): seed
                    for seed in seeds_sequence
                }
                for future in as_completed(future_to_seed):
                    seed = future_to_seed[future]
                    results_parallel[seed] = future.result()
                    if progress_bar is not None:
                        progress_bar.update(1)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        return results_parallel

    def _run_parallel_process(
        self,
        seeds_sequence: cabc.Sequence[int],
        plan: cabc.Mapping[int, tuple[EstimatorKind, EstimatorKind, EstimatorKind]],
        k_0: int,
        initial_theta_stage0: tuple[Array1D, Array1D, Array1D],
        *,
        my_setting: bool,
        show_progress: bool,
        progress_desc: str | None,
        progress_position: int | None,
        num_workers: int,
    ) -> dict[int, list[IterationEstimate]]:
        progress_bar: tqdm | None = None
        if show_progress:
            progress_bar = _build_progress_bar(
                total=len(seeds_sequence),
                desc=progress_desc or "Estimating seeds",
                unit="seed",
                position=progress_position,
                leave=False,
            )

        results_parallel: dict[int, list[IterationEstimate]] = {}

        plan_serializable = {int(k): tuple(v) for k, v in plan.items()}
        serialized_algorithm = cloudpickle.dumps(self)

        try:
            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_process_pool_initializer,
                initargs=(serialized_algorithm,),
            ) as executor:
                future_to_seed = {
                    executor.submit(
                        _process_pool_run_single_seed,
                        seed,
                        plan_serializable,
                        k_0,
                        initial_theta_stage0,
                        my_setting=my_setting,
                    ): seed
                    for seed in seeds_sequence
                }
                for future in as_completed(future_to_seed):
                    seed = future_to_seed[future]
                    results_parallel[seed] = future.result()
                    if progress_bar is not None:
                        progress_bar.update(1)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        return results_parallel

    def _run_single_seed(
        self,
        *,
        seed: int,
        plan: cabc.Mapping[int, tuple[EstimatorKind, EstimatorKind, EstimatorKind]],
        k_0: int,
        initial_theta_stage0: tuple[Array1D, Array1D, Array1D],
        my_setting: bool,
    ) -> list[IterationEstimate]:
        observation = self._simulate_observation(seed)
        stage0_prev = tuple(jnp.asarray(theta) for theta in initial_theta_stage0)
        iterations: list[IterationEstimate] = []
        for k in range(1, k_0 + 1):
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
                my_setting=my_setting,
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
            stage0_prev = tuple(jnp.asarray(v) for v in (theta1_k0, theta2_k0, theta3_k0))
            theta_stage0_snapshot = stage0_prev

            theta1_final = self._estimate_component(
                estimator_kind=plan[k + 1][0],
                component=1,
                k_arg=k + 1,
                observation=observation,
                theta_bar=stage0_prev,
                initial_guess=stage0_prev[0],
                use_prime_objective=False,
                my_setting=my_setting,
            )
            theta2_final = stage0_prev[1]
            if k_0 == 1:
                theta3_final = self._estimate_component(
                    estimator_kind=plan[k + 1][2],
                    component=3,
                    k_arg=1,
                    observation=observation,
                    theta_bar=stage0_prev,
                    initial_guess=stage0_prev[2],
                )
            else:
                theta3_final = stage0_prev[2]

            iterations.append(
                IterationEstimate(
                    k=k,
                    theta_stage0=theta_stage0_snapshot,
                    theta_final=(
                        jnp.asarray(theta1_final),
                        jnp.asarray(theta2_final),
                        jnp.asarray(theta3_final),
                    ),
                )
            )
        return iterations

    def run_single_seed(
        self,
        *,
        seed: int,
        plan: cabc.Mapping[int, tuple[EstimatorKind, EstimatorKind, EstimatorKind]],
        k_0: int,
        initial_theta_stage0: tuple[Array1D, Array1D, Array1D],
        my_setting: bool,
    ) -> list[IterationEstimate]:
        """Public wrapper around :meth:`_run_single_seed` for multiprocessing."""
        return self._run_single_seed(
            seed=seed,
            plan=plan,
            k_0=k_0,
            initial_theta_stage0=initial_theta_stage0,
            my_setting=my_setting,
        )

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
        my_setting: bool = True,
    ) -> jnp.ndarray:
        objective = self._make_objective(
            component=component,
            k_arg=k_arg,
            observation=observation,
            theta_bar=theta_bar,
            use_prime=use_prime_objective,
            my_setting=my_setting,
        )
        bounds = self._bounds[component - 1]
        init = jnp.asarray(initial_guess)
        if estimator_kind == "M":
            kwargs = self._estimator_kwargs["M"]
            return jnp.asarray(
                newton_solve(
                    objective_function=objective,
                    search_bounds=bounds,
                    initial_guess=init,
                    **kwargs,
                )
            )
        if estimator_kind == "B":
            kwargs = self._estimator_kwargs["B"]
            return jnp.asarray(
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
            return jnp.asarray(
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
        my_setting: bool = True,
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
                    x_series, y_series, h, k_arg, my_setting=my_setting
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


@dataclass(frozen=True)
class EstimateStatistics:
    """Mean and standard deviation for theta estimates across seeds."""

    mean: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    std: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]


@dataclass(frozen=True)
class PhaseComparisonData:
    """Prepared statistics for comparing multiple phases side by side."""

    phases: tuple[str, ...]
    ordered_iterations: tuple[int, ...]
    num_parameters: int
    parameter_labels: tuple[str, ...]
    per_phase_stats: tuple[dict[int, tuple[list[float], list[float]]], ...]


def summarize_loop_results(
    loop_results: cabc.Mapping[int, cabc.Sequence[IterationEstimate]],
    *,
    ddof: int = 1,
) -> dict[str, dict[int, EstimateStatistics]]:
    """Compute per-iteration statistics across seeds.

    Args:
        loop_results: Mapping from seed to the sequence of iteration estimates
            returned by :meth:`LoopEstimationAlgorithm.run`.
        ddof: Delta degrees of freedom forwarded to :func:`jax.numpy.std`.

    Returns:
        Dictionary with keys ``"stage0"`` and ``"final"`` mapping each
        iteration index ``k`` to :class:`EstimateStatistics` objects.

    """
    if not loop_results:
        return {"stage0": {}, "final": {}}

    stage0_data, final_data = _collect_iteration_estimates(loop_results)

    stage0_stats = {
        k: _compute_iteration_statistics(values, ddof) for k, values in stage0_data.items()
    }
    final_stats = {
        k: _compute_iteration_statistics(values, ddof) for k, values in final_data.items()
    }

    return {"stage0": stage0_stats, "final": final_stats}


def _collect_iteration_estimates(
    loop_results: cabc.Mapping[int, cabc.Sequence[IterationEstimate]],
) -> tuple[
    dict[int, list[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]],
    dict[int, list[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]],
]:
    stage0_data: dict[int, list[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]] = defaultdict(list)
    final_data: dict[int, list[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]] = defaultdict(list)

    for seq in loop_results.values():
        for estimate in seq:
            stage0_data[estimate.k].append(estimate.theta_stage0)
            final_data[estimate.k].append(estimate.theta_final)

    return stage0_data, final_data


def _compute_iteration_statistics(
    values: cabc.Sequence[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
    ddof: int,
) -> EstimateStatistics:
    if not values:
        msg = "Cannot compute statistics for an empty collection of estimates."
        raise ValueError(msg)

    means: list[jnp.ndarray] = []
    stds: list[jnp.ndarray] = []

    for component_values in zip(*values, strict=True):
        stacked = jnp.stack(component_values, axis=0)
        means.append(jnp.mean(stacked, axis=0))
        stds.append(jnp.std(stacked, axis=0, ddof=ddof))

    mean_tuple = tuple(means)
    std_tuple = tuple(stds)
    return EstimateStatistics(mean=mean_tuple, std=std_tuple)


def format_summary_as_latex(
    summary: dict[str, dict[int, EstimateStatistics]],
    *,
    phase: str = "final",
    true_theta: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    caption: str | None = None,
    mean_formatter: _typing.Callable[[float], str] | None = None,
    std_formatter: _typing.Callable[[float], str] | None = None,
    row_label_template: str = r"$k_0 = {k}$",
    column_labels: _typing.Sequence[str] | None = None,
    include_table_environment: bool = True,
) -> str:
    """Render summary statistics as a LaTeX table with means and standard deviations."""
    flattened_stats, template_stats = _prepare_flattened_stats(summary, phase)
    resolved_labels = _resolve_column_labels(column_labels, template_stats.mean)
    mean_fn = mean_formatter or _default_mean_formatter
    std_fn = std_formatter or _default_std_formatter

    lines = _build_latex_table(
        flattened_stats=flattened_stats,
        column_labels=resolved_labels,
        mean_formatter=mean_fn,
        std_formatter=std_fn,
        row_label_template=row_label_template,
        true_theta=true_theta,
        caption=caption,
        include_table_environment=include_table_environment,
    )
    return "\n".join(lines)


def _prepare_flattened_stats(
    summary: dict[str, dict[int, EstimateStatistics]],
    phase: str,
) -> tuple[list[tuple[int, list[float], list[float]]], EstimateStatistics]:
    phase_stats = summary.get(phase)
    if not phase_stats:
        msg = f"No statistics available for phase '{phase}'."
        raise ValueError(msg)

    ordered_items = sorted(phase_stats.items())
    flattened: list[tuple[int, list[float], list[float]]] = []
    expected_length: int | None = None

    for k, stats in ordered_items:
        means = _flatten_parameter_tuple(stats.mean)
        stds = _flatten_parameter_tuple(stats.std)
        if len(means) != len(stds):
            msg = "Mean and standard deviation lengths do not match."
            raise ValueError(msg)
        expected_length = expected_length or len(means)
        if len(means) != expected_length:
            msg = "Inconsistent number of parameters across iterations."
            raise ValueError(msg)
        flattened.append((k, means, stds))

    return flattened, ordered_items[0][1]


def _resolve_column_labels(
    provided_labels: _typing.Sequence[str] | None,
    mean_parameters: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> list[str]:
    if provided_labels is not None:
        return list(provided_labels)
    return _default_column_labels(mean_parameters)


def _default_mean_formatter(value: float) -> str:
    return f"{value:.6f}"


def _default_std_formatter(value: float) -> str:
    return f"{value:.2e}"


def _build_latex_table(
    *,
    flattened_stats: list[tuple[int, list[float], list[float]]],
    column_labels: _typing.Sequence[str],
    mean_formatter: _typing.Callable[[float], str],
    std_formatter: _typing.Callable[[float], str],
    row_label_template: str,
    true_theta: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None,
    caption: str | None,
    include_table_environment: bool,
) -> list[str]:
    num_columns = len(column_labels)
    lines: list[str] = []
    if include_table_environment:
        lines.extend(["\\begin{table}[H]", "\\centering"])
        if caption:
            lines.append(f"\\caption{{{caption}}}")
        lines.extend(["\\footnotesize", "\\setlength{\\tabcolsep}{4pt}"])

    column_spec = "|" + "|".join(["c"] * (num_columns + 1)) + "|"
    lines.append(f"\\begin{{tabular}}{{{column_spec}}}")
    lines.append("\\hline")

    header_line = " & ".join(["", *column_labels]) + " \\\\ \\hline"
    lines.append(header_line)

    if true_theta is not None:
        true_values = _flatten_parameter_tuple(true_theta)
        if len(true_values) != num_columns:
            msg = "True parameter values do not align with column labels."
            raise ValueError(msg)
        true_line = " & ".join(["True value", *[mean_formatter(val) for val in true_values]])
        lines.append(true_line + " \\\\ \\hline")

    for k, means, stds in flattened_stats:
        row_label = row_label_template.format(k=k)
        values = [
            f"{mean_formatter(mean)} ({std_formatter(std)})"
            for mean, std in zip(means, stds, strict=True)
        ]
        lines.append(" & ".join([row_label, *values]) + " \\\\ \\hline")

    lines.append("\\end{tabular}")
    if include_table_environment:
        lines.append("\\end{table}")
    return lines


def save_summary_to_csv(
    summary: dict[str, dict[int, EstimateStatistics]],
    file_path: str | Path,
    *,
    float_formatter: _typing.Callable[[float], str] | None = None,
    true_theta: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    metadata: cabc.Mapping[str, float | str] | None = None,
) -> Path:
    """Save summary statistics to ``file_path`` in CSV format.

    Args:
        summary: Aggregated loop statistics produced by
            :func:`summarize_loop_results`.
        file_path: Destination CSV path.
        float_formatter: Optional formatter applied to scalar values before
            serialization.
        true_theta: When provided, appends rows containing the true parameter
            values so that downstream tooling can reconstruct them from the CSV
            alone.
        metadata: Optional mapping of scalar metadata (for example ``h`` or
            ``nh``) to record alongside the summary values. Entries are stored
            under a dedicated phase marker and can be recovered when reading
            from the CSV.

    Returns:
        The resolved output :class:`~pathlib.Path`.

    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = _summary_rows(summary, float_formatter=float_formatter)
    if true_theta is not None:
        rows.extend(_truth_rows(true_theta, float_formatter=float_formatter))
    if metadata:
        rows.extend(_metadata_rows(metadata, float_formatter=float_formatter))
    fieldnames = ["phase", "k", "component", "mean", "std"]

    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def save_loop_results_to_csv(
    loop_results: cabc.Mapping[int, cabc.Sequence[IterationEstimate]],
    file_path: str | Path,
    *,
    float_formatter: _typing.Callable[[float], str] | None = None,
) -> Path:
    """Persist raw per-seed loop results to ``file_path`` in CSV format."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = _loop_results_rows(loop_results, float_formatter=float_formatter)
    fieldnames = ["seed", "phase", "k", "component", "values"]

    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def _summary_rows(
    summary: dict[str, dict[int, EstimateStatistics]],
    *,
    float_formatter: _typing.Callable[[float], str] | None = None,
) -> list[dict[str, object]]:
    component_names = ("theta1", "theta2", "theta3")
    rows: list[dict[str, object]] = []

    preferred_order = ["stage0", "final"]
    ordered_phases = [phase for phase in preferred_order if phase in summary]
    ordered_phases.extend(phase for phase in summary if phase not in ordered_phases)

    for phase in ordered_phases:
        stats_by_k = summary[phase]
        for k in sorted(stats_by_k):
            stats = stats_by_k[k]
            for name, mean, std in zip(
                component_names,
                stats.mean,
                stats.std,
                strict=True,
            ):
                rows.append(
                    {
                        "phase": phase,
                        "k": k,
                        "component": name,
                        "mean": _format_array(mean, float_formatter),
                        "std": _format_array(std, float_formatter),
                    }
                )

    phase_order = {phase: index for index, phase in enumerate(preferred_order)}
    rows.sort(
        key=lambda row: (
            phase_order.get(row["phase"], len(phase_order)),
            row["k"],
            row["component"],
        )
    )

    return rows


def _truth_rows(
    true_theta: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    *,
    float_formatter: _typing.Callable[[float], str] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    component_names = ("theta1", "theta2", "theta3")
    for name, component in zip(component_names, true_theta, strict=True):
        rows.append(
            {
                "phase": TRUTH_PHASE,
                "k": "",
                "component": name,
                "mean": _format_array(component, float_formatter),
                "std": "",
            }
        )
    return rows


def _metadata_rows(
    metadata: cabc.Mapping[str, float | str],
    *,
    float_formatter: _typing.Callable[[float], str] | None = None,
) -> list[dict[str, object]]:
    def _format(value: float | str) -> str:
        if isinstance(value, str):
            return value
        numeric = float(value)
        if float_formatter is not None:
            return float_formatter(numeric)
        return f"{numeric:.6f}"

    rows: list[dict[str, object]] = []
    for key, value in metadata.items():
        rows.append(
            {
                "phase": META_PHASE,
                "k": "",
                "component": str(key),
                "mean": _format(value),
                "std": "",
            }
        )
    return rows


def _loop_results_rows(
    loop_results: cabc.Mapping[int, cabc.Sequence[IterationEstimate]],
    *,
    float_formatter: _typing.Callable[[float], str] | None = None,
) -> list[dict[str, object]]:
    component_names = ("theta1", "theta2", "theta3")
    phase_order = ("stage0", "final")
    rows: list[dict[str, object]] = []

    for seed in sorted(loop_results):
        estimates = loop_results[seed]
        for estimate in estimates:
            phase_to_theta = {
                "stage0": estimate.theta_stage0,
                "final": estimate.theta_final,
            }
            for phase in phase_order:
                theta_values = phase_to_theta.get(phase)
                if theta_values is None:
                    continue
                for name, component in zip(component_names, theta_values, strict=True):
                    rows.append(
                        {
                            "seed": seed,
                            "phase": phase,
                            "k": estimate.k,
                            "component": name,
                            "values": _format_array(component, float_formatter),
                        }
                    )

    return rows


FLOAT_PATH_EPS = 1e-9


def _format_array(
    array: jnp.ndarray,
    float_formatter: _typing.Callable[[float], str] | None,
) -> str:
    flattened = jnp.asarray(array).reshape(-1)

    def _format_value(value: float) -> str:
        if float_formatter is not None:
            return float_formatter(value)
        return f"{value:.6f}"

    formatted = (_format_value(float(val)) for val in flattened.tolist())
    return " ".join(formatted)


def _format_numeric_for_path(value: float | str) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return _sanitize_folder_component(str(value))
    if abs(numeric - round(numeric)) < FLOAT_PATH_EPS:
        return str(round(numeric))
    return f"{numeric:.6f}".rstrip("0").rstrip(".")


def _sanitize_folder_component(component: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", component.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return cleaned or "output"


def build_output_directory(
    base_path: str | Path,
    model: object | str,
    *,
    h: float,
    nh: float,
) -> Path:
    if isinstance(model, str):
        model_name = model
    else:
        model_name = getattr(model, "name", None) or model.__class__.__name__
    folder_name = (
        f"{_sanitize_folder_component(str(model_name))}_"
        f"nh{_format_numeric_for_path(nh)}_h{_format_numeric_for_path(h)}"
    )
    output_dir = Path(base_path) / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _contains_bayesian_estimators(
    plan: cabc.Mapping[int, tuple[EstimatorKind, EstimatorKind, EstimatorKind]],
) -> bool:
    return any("B" in estimators for estimators in plan.values())


def save_model_structure(
    model: object,
    destination_dir: str | Path,
    *,
    filename: str = "model_structure.txt",
    encoding: str = "utf-8",
) -> Path:
    """Persist the symbolic form of ``A``, ``B``, and ``H`` for a model.

    Args:
        model: Model instance exposing symbolic artifacts ``A``, ``B``, and ``H``.
        destination_dir: Directory where the structure description will be saved.
        filename: Target file name, defaults to ``"model_structure.txt"``.
        encoding: Text encoding used when writing the file.

    Returns:
        The resolved output :class:`~pathlib.Path`.

    """
    dir_path = Path(destination_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    target_path = dir_path / filename

    model_name = getattr(model, "name", None) or model.__class__.__name__

    sections = {
        "A": getattr(model, "A", None),
        "B": getattr(model, "B", None),
        "H": getattr(model, "H", None),
    }

    lines = ["# Model structure", "", f"Model name: {model_name}", ""]

    for label, artifact in sections.items():
        if artifact is None:
            continue

        expr = getattr(artifact, "expr", artifact)
        lines.append(f"## {label}(x, y, θ)")
        try:
            pretty_expr = sp.pretty(expr)
        except (AttributeError, TypeError, ValueError):  # pragma: no cover - fallback
            pretty_expr = None

        if pretty_expr:
            lines.append("```text")
            lines.append(pretty_expr)
            lines.append("```")
            lines.append("")

    lines.append("SymPy expression:")
    lines.append("```python")
    lines.append(str(expr))
    lines.append("```")
    lines.append("")

    content = "\n".join(lines).rstrip() + "\n"
    target_path.write_text(content, encoding=encoding)
    return target_path


def _parse_numeric_sequence(value: str) -> list[float]:
    if not value:
        return []
    return [float(part) for part in value.strip().split() if part]


def _initialise_component_bucket(
    component_order: tuple[str, ...],
) -> dict[str, dict[str, list[float] | None]]:
    return {
        "mean": dict.fromkeys(component_order, None),
        "std": dict.fromkeys(component_order, None),
    }


def _parse_iteration_index(raw_value: str) -> int:
    if not raw_value:
        msg = "CSV summary row is missing iteration index."
        raise ValueError(msg)
    return int(raw_value)


def _store_component_stats(
    bucket: dict[str, dict[str, list[float] | None]],
    *,
    component: str,
    mean_values: list[float],
    std_values: list[float],
    component_order: tuple[str, ...],
) -> None:
    if component not in component_order:
        msg = f"Unknown component '{component}' detected in summary CSV."
        raise ValueError(msg)
    bucket["mean"][component] = mean_values
    bucket["std"][component] = std_values or None


def _handle_special_summary_row(
    *,
    phase: str,
    component: str,
    mean_values: list[float],
    truth_parts: dict[str, list[float]],
) -> bool:
    if phase == TRUTH_PHASE:
        truth_parts[component] = mean_values
        return True
    return phase == META_PHASE


def _load_summary_csv(
    file_path: str | Path,
) -> tuple[dict[str, dict[int, EstimateStatistics]], tuple[jnp.ndarray, ...] | None]:
    path = Path(file_path)
    component_order = ("theta1", "theta2", "theta3")
    aggregated: dict[tuple[str, int], dict[str, dict[str, list[float] | None]]] = {}
    summary: dict[str, dict[int, EstimateStatistics]] = defaultdict(dict)
    truth_parts: dict[str, list[float]] = {}

    with path.open("r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            phase = row["phase"]
            component = row["component"]
            mean_values = _parse_numeric_sequence(row.get("mean", ""))
            std_values = _parse_numeric_sequence(row.get("std", ""))

            if _handle_special_summary_row(
                phase=phase,
                component=component,
                mean_values=mean_values,
                truth_parts=truth_parts,
            ):
                continue

            k = _parse_iteration_index(row["k"])
            key = (phase, k)
            bucket = aggregated.setdefault(
                key,
                _initialise_component_bucket(component_order),
            )
            _store_component_stats(
                bucket,
                component=component,
                mean_values=mean_values,
                std_values=std_values,
                component_order=component_order,
            )

    for (phase, k), values in aggregated.items():
        means: list[jnp.ndarray] = []
        stds: list[jnp.ndarray] = []
        for component in component_order:
            mean_seq = values["mean"].get(component)
            std_seq = values["std"].get(component)
            if mean_seq is None or std_seq is None:
                msg = f"Incomplete statistics for phase '{phase}', k={k}, component '{component}'."
                raise ValueError(msg)
            means.append(jnp.asarray(mean_seq))
            if std_seq is None:
                stds.append(jnp.zeros_like(means[-1]))
            else:
                stds.append(jnp.asarray(std_seq))
        summary.setdefault(phase, {})[k] = EstimateStatistics(mean=tuple(means), std=tuple(stds))

    true_theta: tuple[jnp.ndarray, ...] | None
    if truth_parts:
        true_arrays: list[jnp.ndarray] = []
        for component in component_order:
            values = truth_parts.get(component)
            if values is None:
                msg = f"Missing true values for component '{component}' in summary CSV."
                raise ValueError(msg)
            true_arrays.append(jnp.asarray(values))
        true_theta = tuple(true_arrays)
    else:
        true_theta = None

    return summary, true_theta


def _flatten_parameter_tuple(
    parameters: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> list[float]:
    flattened: list[float] = []
    for array in parameters:
        values = jnp.asarray(array).reshape(-1).tolist()
        flattened.extend(float(val) for val in values)
    return flattened


def _default_column_labels(
    mean_parameters: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> list[str]:
    labels: list[str] = []
    for idx, array in enumerate(mean_parameters, start=1):
        flat = jnp.asarray(array).reshape(-1)
        if flat.size == 1:
            labels.append(f"\\hat{{\\theta}}_{{{idx}}}")
        else:
            labels.extend(
                f"\\hat{{\\theta}}_{{{idx}{component_idx}}}"
                for component_idx in range(1, flat.size + 1)
            )
    return labels


def format_stage_comparison_latex(
    summary: dict[str, dict[int, EstimateStatistics]] | str | Path,
    *,
    phases: _typing.Sequence[str] = ("stage0", "final"),
    true_theta: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    caption: str | None = None,
    mean_formatter: _typing.Callable[[float], str] | None = None,
    std_formatter: _typing.Callable[[float], str] | None = None,
    row_label_template: str = r"$k_0 = {k}$",
    include_table_environment: bool = True,
) -> str:
    """Render a LaTeX table comparing multiple phases side by side.

    The ``summary`` argument accepts either the in-memory statistics structure
    produced by :func:`summarize_loop_results` or the path to a CSV file that was
    emitted via :func:`save_summary_to_csv`. When a CSV path is supplied, the
    stored true parameter values (if present) are recovered automatically unless
    an explicit ``true_theta`` override is provided.
    """
    summary_data: dict[str, dict[int, EstimateStatistics]]
    resolved_true_theta = true_theta
    if isinstance(summary, (str, Path)):
        summary_data, csv_true = _load_summary_csv(summary)
        if resolved_true_theta is None and csv_true is not None:
            resolved_true_theta = csv_true
    else:
        summary_data = summary

    data = _gather_phase_comparison_data(summary_data, phases)
    mean_fn = mean_formatter or _default_mean_formatter
    std_fn = std_formatter or _default_std_formatter
    lines = _render_stage_comparison_table(
        data=data,
        mean_formatter=mean_fn,
        std_formatter=std_fn,
        row_label_template=row_label_template,
        true_theta=resolved_true_theta,
        caption=caption,
        include_table_environment=include_table_environment,
    )
    return "\n".join(lines)


def _phase_display_name(phase: str) -> str:
    normalized = phase.lower().replace(" ", "_")
    if normalized == "stage0":
        return "Stage-0"
    if normalized == "final":
        return "Final"
    return phase.replace("_", " ").title()


def _gather_phase_comparison_data(
    summary: dict[str, dict[int, EstimateStatistics]],
    phases: _typing.Sequence[str],
) -> PhaseComparisonData:
    phase_sequence = tuple(phases)
    if not phase_sequence:
        msg = "At least one phase must be provided for comparison."
        raise ValueError(msg)

    per_phase_stats: list[dict[int, tuple[list[float], list[float]]]] = []
    common_iterations: set[int] | None = None
    parameter_labels: tuple[str, ...] | None = None
    num_parameters: int | None = None

    for phase in phase_sequence:
        stats_by_k = summary.get(phase)
        if not stats_by_k:
            msg = f"No statistics available for phase '{phase}'."
            raise ValueError(msg)

        flattened_per_k = _flatten_stats_by_iteration(stats_by_k)
        per_phase_stats.append(flattened_per_k)
        common_iterations = _intersect_iterations(common_iterations, flattened_per_k)
        if not common_iterations:
            msg = "No common iteration indices across provided phases."
            raise ValueError(msg)

        num_parameters, parameter_labels = _ensure_parameter_metadata(
            stats_by_k=stats_by_k,
            flattened_per_k=flattened_per_k,
            common_iterations=common_iterations,
            num_parameters=num_parameters,
            parameter_labels=parameter_labels,
        )

    if common_iterations is None:
        msg = "No iterations detected for the provided phases."
        raise ValueError(msg)
    ordered_iterations = tuple(sorted(common_iterations))
    if num_parameters is None or parameter_labels is None:
        msg = "Unable to determine parameter structure for comparison."
        raise ValueError(msg)

    return PhaseComparisonData(
        phases=phase_sequence,
        ordered_iterations=ordered_iterations,
        num_parameters=num_parameters,
        parameter_labels=parameter_labels,
        per_phase_stats=tuple(per_phase_stats),
    )


def _render_stage_comparison_table(
    *,
    data: PhaseComparisonData,
    mean_formatter: _typing.Callable[[float], str],
    std_formatter: _typing.Callable[[float], str],
    row_label_template: str,
    true_theta: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None,
    caption: str | None,
    include_table_environment: bool,
) -> list[str]:
    column_labels = _build_comparison_column_labels(data)
    lines = _initial_table_lines(
        include_table_environment=include_table_environment,
        caption=caption,
    )
    header = " & ".join(["", *column_labels]) + " \\\\ \\hline"
    lines.extend(["\\begin{tabular}{" + _column_spec(len(column_labels)) + "}", "\\hline", header])

    if true_theta is not None:
        true_values = _flatten_parameter_tuple(true_theta)
        if len(true_values) != data.num_parameters:
            msg = "True parameter values do not align with detected labels."
            raise ValueError(msg)
        lines.append(
            _format_true_value_row(
                true_values=true_values,
                mean_formatter=mean_formatter,
                phase_count=len(data.phases),
            )
        )

    for k in data.ordered_iterations:
        lines.append(
            _format_comparison_row(
                iteration=k,
                data=data,
                mean_formatter=mean_formatter,
                std_formatter=std_formatter,
                row_label_template=row_label_template,
            )
        )

    lines.append("\\end{tabular}")
    if include_table_environment:
        lines.append("\\end{table}")
    return lines


def _flatten_stats_by_iteration(
    stats_by_k: dict[int, EstimateStatistics],
) -> dict[int, tuple[list[float], list[float]]]:
    flattened: dict[int, tuple[list[float], list[float]]] = {}
    for k, stats in stats_by_k.items():
        means = _flatten_parameter_tuple(stats.mean)
        stds = _flatten_parameter_tuple(stats.std)
        if len(means) != len(stds):
            msg = "Mean and standard deviation lengths do not match."
            raise ValueError(msg)
        flattened[k] = (means, stds)
    return flattened


def _intersect_iterations(
    existing: set[int] | None,
    flattened_per_k: dict[int, tuple[list[float], list[float]]],
) -> set[int]:
    current = set(flattened_per_k)
    if existing is None:
        return current
    return existing & current


def _ensure_parameter_metadata(
    *,
    stats_by_k: dict[int, EstimateStatistics],
    flattened_per_k: dict[int, tuple[list[float], list[float]]],
    common_iterations: set[int],
    num_parameters: int | None,
    parameter_labels: tuple[str, ...] | None,
) -> tuple[int, tuple[str, ...]]:
    if num_parameters is None or parameter_labels is None:
        first_iteration = min(flattened_per_k)
        num_parameters = len(flattened_per_k[first_iteration][0])
        parameter_labels = tuple(_default_column_labels(stats_by_k[first_iteration].mean))
        return num_parameters, parameter_labels

    inconsistent = any(len(flattened_per_k[k][0]) != num_parameters for k in common_iterations)
    if inconsistent:
        msg = "Inconsistent number of parameters across phases."
        raise ValueError(msg)
    return num_parameters, parameter_labels


def _build_comparison_column_labels(data: PhaseComparisonData) -> list[str]:
    phase_labels = [_phase_display_name(phase) for phase in data.phases]
    labels: list[str] = []
    for param_label in data.parameter_labels:
        labels.extend([f"{phase_label} ${param_label}$" for phase_label in phase_labels])
    return labels


def _initial_table_lines(
    *,
    include_table_environment: bool,
    caption: str | None,
) -> list[str]:
    lines: list[str] = []
    if include_table_environment:
        lines.extend(["\\begin{table}[H]", "\\centering"])
        if caption:
            lines.append(f"\\caption{{{caption}}}")
        lines.extend(["\\footnotesize", "\\setlength{\\tabcolsep}{4pt}"])
    return lines


def _column_spec(column_count: int) -> str:
    # +1 accounts for the leading label column
    total_columns = column_count + 1
    return "|" + "|".join(["c"] * total_columns) + "|"


def _format_true_value_row(
    *,
    true_values: list[float],
    mean_formatter: _typing.Callable[[float], str],
    phase_count: int,
) -> str:
    cells: list[str] = ["True value"]
    for value in true_values:
        formatted = mean_formatter(value)
        cells.extend([formatted] * phase_count)
    return " & ".join(cells) + " \\\\ \\hline"


def _format_comparison_row(
    *,
    iteration: int,
    data: PhaseComparisonData,
    mean_formatter: _typing.Callable[[float], str],
    std_formatter: _typing.Callable[[float], str],
    row_label_template: str,
) -> str:
    row_label = row_label_template.format(k=iteration)
    cells: list[str] = [row_label]
    for parameter_index in range(data.num_parameters):
        for phase_stats in data.per_phase_stats:
            means, stds = phase_stats[iteration]
            formatted_value = f"{mean_formatter(means[parameter_index])} ("
            formatted_value += f"{std_formatter(stds[parameter_index])})"
            cells.append(formatted_value)
    return " & ".join(cells) + " \\\\ \\hline"

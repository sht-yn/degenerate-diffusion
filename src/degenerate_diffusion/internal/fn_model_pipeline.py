"""Utilities for running the FN-model loop with the seed runner API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

import jax.numpy as jnp
from sympy import Array, symbols

from degenerate_diffusion import DegenerateDiffusionProcess, LikelihoodEvaluator
from degenerate_diffusion.estimation.loop_estimation_algorithm import (
    EstimatorKind,
    SeedRunnerConfig,
    build_seed_runner,
)

ThetaTuple = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
EstimatorPlan = dict[int, tuple[EstimatorKind, EstimatorKind, EstimatorKind]]
SeedRunnerOutput = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]


@dataclass(frozen=True)
class FnModelSetup:
    """Container for the symbolic model and seed runner configuration."""

    process: DegenerateDiffusionProcess
    evaluator: LikelihoodEvaluator
    true_theta: ThetaTuple
    bounds_theta1: tuple[tuple[float, float], ...]
    bounds_theta2: tuple[tuple[float, float], ...]
    bounds_theta3: tuple[tuple[float, float], ...]
    t_max: float
    h: float
    burn_out: float
    dt: float
    loop_plan: EstimatorPlan
    initial_theta_stage0: ThetaTuple
    model_label: str
    newton_kwargs: Mapping[str, Any]
    nuts_kwargs: Mapping[str, Any]
    one_step_kwargs: Mapping[str, Any]

    def build_seed_runner_config(self) -> SeedRunnerConfig:
        """Create a :class:`SeedRunnerConfig` instance from the stored settings."""
        return SeedRunnerConfig(
            true_theta=self.true_theta,
            t_max=self.t_max,
            h=self.h,
            burn_out=self.burn_out,
            dt=self.dt,
            bounds_theta1=self.bounds_theta1,
            bounds_theta2=self.bounds_theta2,
            bounds_theta3=self.bounds_theta3,
            newton_kwargs=self.newton_kwargs,
            nuts_kwargs=self.nuts_kwargs,
            one_step_kwargs=self.one_step_kwargs,
        )


@dataclass(frozen=True)
class FnModelSeedOutput:
    """Outputs produced by the seed runner for a single seed."""

    theta1_stage0: jnp.ndarray
    theta1_final: jnp.ndarray
    theta2_stage0: jnp.ndarray
    theta3_final: jnp.ndarray

    @classmethod
    def from_raw(cls, raw: SeedRunnerOutput) -> FnModelSeedOutput:
        """Construct from the raw tuple returned by ``build_seed_runner``."""
        theta1_stage0, theta1_final, theta2_stage0, theta3_final = raw
        return cls(
            theta1_stage0=jnp.asarray(theta1_stage0),
            theta1_final=jnp.asarray(theta1_final),
            theta2_stage0=jnp.asarray(theta2_stage0),
            theta3_final=jnp.asarray(theta3_final),
        )

    def as_theta_stage0(self, index: int) -> ThetaTuple:
        """Return the stage-0 parameter triple for iteration ``index``."""
        return (
            self.theta1_stage0[index],
            self.theta2_stage0[index],
            self.theta3_final[index],
        )

    def as_theta_final(self, index: int) -> ThetaTuple:
        """Return the final parameter triple for iteration ``index``."""
        return (
            self.theta1_final[index],
            self.theta2_stage0[index],
            self.theta3_final[index],
        )


@dataclass(frozen=True)
class FnModelSummary:
    """Simple aggregate statistics across seeds."""

    mean: FnModelSeedOutput
    std: FnModelSeedOutput


@dataclass(frozen=True)
class FnModelResults:
    """Outputs produced after executing the FN-model seed runner."""

    setup: FnModelSetup
    per_seed: dict[int, FnModelSeedOutput]
    summary: FnModelSummary
    metadata: dict[str, float]


def create_fnmodel_setup() -> FnModelSetup:
    """Instantiate the FitzHugh-Nagumo model and default runner settings."""
    x_sym, y_sym = symbols("x, y")
    theta_10, theta_20, theta_21 = symbols("theta_10 theta_20 theta_21")
    theta_30, theta_31 = symbols("theta_30 theta_31")

    x_array = Array([x_sym])
    y_array = Array([y_sym])
    theta1_array = Array([theta_10])
    theta2_array = Array([theta_20, theta_21])
    theta3_array = Array([theta_30, theta_31])

    drift = Array([theta_20 * y_sym - x_sym + theta_21])
    diffusion = Array([[theta_10]])
    observation = Array([(y_sym - y_sym**3 - x_sym + theta_31) / theta_30])

    process = DegenerateDiffusionProcess(
        x=x_array,
        y=y_array,
        theta_1=theta1_array,
        theta_2=theta2_array,
        theta_3=theta3_array,
        A=drift,
        B=diffusion,
        H=observation,
    )
    evaluator = LikelihoodEvaluator(process)

    true_theta1 = jnp.array([0.3])
    true_theta2 = jnp.array([1.5, 0.8])
    true_theta3 = jnp.array([0.1, 0.0])
    true_theta: ThetaTuple = (true_theta1, true_theta2, true_theta3)

    loop_plan: EstimatorPlan = {
        1: ("B", "B", "B"),
        2: ("M", "M", "M"),
        3: ("M", "M", "M"),
    }

    return FnModelSetup(
        process=process,
        evaluator=evaluator,
        true_theta=true_theta,
        bounds_theta1=((0.1, 0.5),),
        bounds_theta2=((0.5, 2.5), (0.5, 1.5)),
        bounds_theta3=((0.01, 0.3), (-1.0, 1.0)),
        t_max=100.0,
        h=0.05,
        burn_out=50.0,
        dt=0.001,
        loop_plan=loop_plan,
        initial_theta_stage0=(
            jnp.array([0.2]),
            jnp.array([0.5, 0.5]),
            jnp.array([0.2, 0.1]),
        ),
        model_label="FNmodel",
        newton_kwargs={},
        nuts_kwargs={},
        one_step_kwargs={},
    )


def run_fnmodel_estimation(
    *,
    seeds: Iterable[int] | None = None,
) -> FnModelResults:
    """Execute the FN-model seed runner for each seed in ``seeds``."""
    setup = create_fnmodel_setup()
    config = setup.build_seed_runner_config()
    runner = build_seed_runner(
        evaluator=setup.evaluator,
        model=setup.process,
        plan=setup.loop_plan,
        config=config,
    )

    seed_iterable = list(seeds) if seeds is not None else [0]
    per_seed: dict[int, FnModelSeedOutput] = {}
    for seed in seed_iterable:
        raw = cast("SeedRunnerOutput", runner(int(seed), setup.initial_theta_stage0))
        per_seed[int(seed)] = FnModelSeedOutput.from_raw(raw)

    summary = _compute_summary(per_seed)

    metadata = {
        "t_max": float(setup.t_max),
        "h": float(setup.h),
        "burn_out": float(setup.burn_out),
        "dt": float(setup.dt),
        "num_iterations": float(max(setup.loop_plan) - 1),
    }

    return FnModelResults(
        setup=setup,
        per_seed=per_seed,
        summary=summary,
        metadata=metadata,
    )


def _compute_summary(outputs: Mapping[int, FnModelSeedOutput]) -> FnModelSummary:
    if not outputs:
        msg = "No seed outputs provided for summary computation."
        raise ValueError(msg)

    ordered = [outputs[seed] for seed in sorted(outputs)]

    def _stack(field: str) -> jnp.ndarray:
        return jnp.stack([getattr(output, field) for output in ordered], axis=0)

    theta1_stage0_stack = _stack("theta1_stage0")
    theta1_final_stack = _stack("theta1_final")
    theta2_stage0_stack = _stack("theta2_stage0")
    theta3_final_stack = _stack("theta3_final")

    def _mean(arr: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(arr, axis=0)

    def _std(arr: jnp.ndarray) -> jnp.ndarray:
        ddof = 1 if arr.shape[0] > 1 else 0
        return jnp.std(arr, axis=0, ddof=ddof)

    mean = FnModelSeedOutput(
        theta1_stage0=_mean(theta1_stage0_stack),
        theta1_final=_mean(theta1_final_stack),
        theta2_stage0=_mean(theta2_stage0_stack),
        theta3_final=_mean(theta3_final_stack),
    )
    std = FnModelSeedOutput(
        theta1_stage0=_std(theta1_stage0_stack),
        theta1_final=_std(theta1_final_stack),
        theta2_stage0=_std(theta2_stage0_stack),
        theta3_final=_std(theta3_final_stack),
    )

    return FnModelSummary(mean=mean, std=std)

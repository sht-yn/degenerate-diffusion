"""Internal FN-model pipeline shared between tests and notebooks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
from sympy import Array, symbols

from degenerate_diffusion import DegenerateDiffusionProcess, LikelihoodEvaluator
from degenerate_diffusion.estimation import loop_estimation_algorithm as loop_algo

EstimatorPlan = dict[
    int,
    tuple[
        loop_algo.EstimatorKind,
        loop_algo.EstimatorKind,
        loop_algo.EstimatorKind,
    ],
]
ThetaTuple = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
LoopResults = dict[int, Sequence[loop_algo.IterationEstimate]]
SummaryStats = dict[str, dict[int, loop_algo.EstimateStatistics]]


@dataclass(frozen=True)
class FnModelSetup:
    """Container for the symbolic model and loop configuration."""

    process: DegenerateDiffusionProcess
    evaluator: LikelihoodEvaluator
    true_theta: ThetaTuple
    bounds_theta1: tuple[tuple[float, float], ...]
    bounds_theta2: tuple[tuple[float, float], ...]
    bounds_theta3: tuple[tuple[float, float], ...]
    simulation_config: loop_algo.SimulationConfig
    loop_plan: EstimatorPlan
    initial_theta_stage0: ThetaTuple
    model_label: str


@dataclass(frozen=True)
class FnModelResults:
    """Outputs produced after executing the FN-model estimation loop."""

    setup: FnModelSetup
    loop_results: LoopResults
    summary: SummaryStats
    metadata: dict[str, float]

    def save_artifacts(self, *, base_directory: Path | None = None) -> FnModelArtifactPaths:
        """Persist CSV summaries and the model structure to disk."""
        target_dir = base_directory or loop_algo.build_output_directory(
            Path("experiments"),
            self.setup.model_label,
            h=self.metadata["h"],
            nh=self.metadata["nh"],
        )
        summary_csv = loop_algo.save_summary_to_csv(
            self.summary,
            target_dir / "summary.csv",
            true_theta=self.setup.true_theta,
            metadata=self.metadata,
        )
        detail_csv = loop_algo.save_loop_results_to_csv(
            self.loop_results,
            target_dir / "detail.csv",
        )
        model_structure = loop_algo.save_model_structure(
            self.setup.process,
            target_dir,
            filename="model_structure.md",
        )
        return FnModelArtifactPaths(
            output_directory=target_dir,
            summary_csv=summary_csv,
            detail_csv=detail_csv,
            model_structure=model_structure,
        )


@dataclass(frozen=True)
class FnModelArtifactPaths:
    """File-system locations returned after saving the pipeline outputs."""

    output_directory: Path
    summary_csv: Path
    detail_csv: Path
    model_structure: Path


def create_fnmodel_setup() -> FnModelSetup:
    """Instantiate the FitzHugh-Nagumo model and default estimation settings."""
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

    sim_cfg = loop_algo.SimulationConfig(
        true_theta=true_theta,
        t_max=10.0,
        h=0.05,
        burn_out=50.0,
        dt=0.001,
    )

    loop_plan: EstimatorPlan = {
        1: ("B", "B", "B"),
        2: ("M", "M", "M"),
        3: ("M", "M", "M"),
        4: ("M", "M", "M"),
    }

    return FnModelSetup(
        process=process,
        evaluator=evaluator,
        true_theta=true_theta,
        bounds_theta1=((0.1, 0.5),),
        bounds_theta2=((0.5, 2.5), (0.5, 1.5)),
        bounds_theta3=((0.01, 0.3), (-1.0, 1.0)),
        simulation_config=sim_cfg,
        loop_plan=loop_plan,
        initial_theta_stage0=(
            jnp.array([0.2]),
            jnp.array([0.5, 0.5]),
            jnp.array([0.2, 0.1]),
        ),
        model_label="FNmodel_test",
    )


def run_fnmodel_estimation(
    *,
    seeds: Iterable[int] | None = None,
    show_progress: bool = False,
    progress_desc: str | None = None,
    num_workers: int | None = None,
    my_setting: bool = True,
) -> FnModelResults:
    """Execute the default FN-model loop estimation pipeline."""
    setup = create_fnmodel_setup()
    estimator = loop_algo.LoopEstimationAlgorithm(
        evaluator=setup.evaluator,
        simulation_config=setup.simulation_config,
        bounds_theta1=setup.bounds_theta1,
        bounds_theta2=setup.bounds_theta2,
        bounds_theta3=setup.bounds_theta3,
    )

    seed_iterable = seeds if seeds is not None else range(1)
    loop_results = estimator.run(
        seeds=seed_iterable,
        plan=setup.loop_plan,
        k_0=max(setup.loop_plan) - 1,
        initial_theta_stage0=setup.initial_theta_stage0,
        my_setting=my_setting,
        show_progress=show_progress,
        progress_desc=progress_desc or "Seeds",
        num_workers=num_workers,
    )

    summary = loop_algo.summarize_loop_results(loop_results, ddof=1)
    num_steps = round(setup.simulation_config.t_max / setup.simulation_config.h)
    metadata = {
        "h": float(setup.simulation_config.h),
        "nh": float(setup.simulation_config.h * num_steps),
    }

    return FnModelResults(
        setup=setup,
        loop_results=loop_results,
        summary=summary,
        metadata=metadata,
    )

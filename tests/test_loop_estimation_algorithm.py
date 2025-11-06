from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from sympy import Array, symbols

from degenerate_diffusion.estimation.loop_estimation_algorithm import (
    SeedRunnerConfig,
    build_seed_runner,
)
from degenerate_diffusion.evaluation.likelihood_evaluator import LikelihoodEvaluator
from degenerate_diffusion.internal.fn_model_pipeline import create_fnmodel_setup
from degenerate_diffusion.processes.degenerate_diffusion_process import DegenerateDiffusionProcess


def test_build_seed_runner_runs_and_returns_shapes() -> None:
    setup = create_fnmodel_setup()

    evaluator = setup.evaluator
    model = setup.process
    true_theta = setup.true_theta

    # simple plan for k=1..2
    plan = {
        1: ("B", "B", "B"),
        2: ("M", "M", "M"),
    }

    bounds = tuple(((-jnp.inf, jnp.inf),) * len(jnp.atleast_1d(true_theta[0])))
    cfg = SeedRunnerConfig(
        true_theta=true_theta,
        t_max=0.2,
        h=0.1,
        burn_out=0.0,
        dt=0.01,
        bounds_theta1=bounds,
        bounds_theta2=bounds,
        bounds_theta3=bounds,
        newton_kwargs={"max_iters": 5, "tol": 1e-5, "damping": 0.1, "use_adam_fallback": False},
        nuts_kwargs={"num_warmup": 2, "num_samples": 3, "step_size": 1e-1},
        one_step_kwargs={"damping": 0.1},
    )

    runner = build_seed_runner(
        evaluator=evaluator,
        model=model,
        plan=plan,
        config=cfg,
    )

    theta0 = tuple(jnp.asarray(t) for t in true_theta)
    outputs = runner(0, theta0)

    assert len(outputs) == 4
    theta1_stage0_seq, theta1_final_seq, theta2_stage0_seq, theta3_final_seq = outputs

    # Number of iterations equals max(plan) - 1 because k starts from 1
    expected_steps = max(plan) - 1

    expected_theta1_shape = (expected_steps, *jnp.asarray(true_theta[0]).shape)
    expected_theta2_shape = (expected_steps, *jnp.asarray(true_theta[1]).shape)
    expected_theta3_shape = (expected_steps, *jnp.asarray(true_theta[2]).shape)

    assert theta1_stage0_seq.shape == expected_theta1_shape
    assert theta1_final_seq.shape == expected_theta1_shape
    assert theta2_stage0_seq.shape == expected_theta2_shape
    assert theta3_final_seq.shape == expected_theta3_shape

    # Ensure the last step produces finite estimates
    assert jnp.isfinite(theta1_stage0_seq[-1]).all()
    assert jnp.isfinite(theta1_final_seq[-1]).all()
    assert jnp.isfinite(theta2_stage0_seq[-1]).all()
    assert jnp.isfinite(theta3_final_seq[-1]).all()


enable_x64 = True
jax.config.update("jax_enable_x64", enable_x64)
base_prng_key = jax.random.PRNGKey(42)


@dataclass
class NotebookSettings:
    """Parameter set reused by the FN-model notebook fixture."""

    model: Any
    evaluator: Any
    true_theta: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    bounds_theta1: list[tuple[float, float]]
    bounds_theta2: list[tuple[float, float]]
    bounds_theta3: list[tuple[float, float]]
    t_max: float
    h: float
    burn_out: float
    dt: float
    loop_plan: dict[int, tuple[str, str, str]]
    initial_theta_stage0: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]


x_sym, y_sym = symbols("x, y")
theta_10, theta_20, theta_21 = symbols("theta_10 theta_20 theta_21")
theta_30, theta_31 = symbols("theta_30 theta_31")

x = Array([x_sym])
y = Array([y_sym])
theta_1 = Array([theta_10])
theta_2 = Array([theta_20, theta_21])
theta_3 = Array([theta_30, theta_31])

A = Array([theta_20 * y_sym - x_sym + theta_21])
B = Array([[theta_10]])
H = Array([(y_sym - y_sym**3 - x_sym + theta_31) / theta_30])

FNmodel = DegenerateDiffusionProcess(
    x=x,
    y=y,
    theta_1=theta_1,
    theta_2=theta_2,
    theta_3=theta_3,
    A=A,
    B=B,
    H=H,
)
FN_likelihood = LikelihoodEvaluator(FNmodel)

true_theta1 = jnp.array([0.3])
true_theta2 = jnp.array([1.5, 0.8])
true_theta3 = jnp.array([0.1, 0.0])
true_theta = (true_theta1, true_theta2, true_theta3)

t_max = 100.0
burn_out = 50.0
h = 0.05
dt = 0.001

bounds_theta1 = [(0.1, 0.5)]
bounds_theta2 = [(0.5, 2.5), (0.5, 1.5)]
bounds_theta3 = [(0.01, 0.3), (-1.0, 1.0)]

loop_plan: dict[int, tuple[str, str, str]] = {
    1: ("B", "B", "B"),
    2: ("M", "M", "M"),
    3: ("M", "M", "M"),
}

initial_theta_stage0 = (
    jnp.array([0.2]),
    jnp.array([0.5, 0.5]),
    jnp.array([0.2, 0.1]),
)

settings = NotebookSettings(
    model=FNmodel,
    evaluator=FN_likelihood,
    true_theta=true_theta,
    bounds_theta1=bounds_theta1,
    bounds_theta2=bounds_theta2,
    bounds_theta3=bounds_theta3,
    t_max=t_max,
    h=h,
    burn_out=burn_out,
    dt=dt,
    loop_plan=loop_plan,
    initial_theta_stage0=initial_theta_stage0,
)


seed_runner_config = SeedRunnerConfig(
    true_theta=settings.true_theta,
    t_max=settings.t_max,
    h=settings.h,
    burn_out=settings.burn_out,
    dt=settings.dt,
    bounds_theta1=settings.bounds_theta1,
    bounds_theta2=settings.bounds_theta2,
    bounds_theta3=settings.bounds_theta3,
    newton_kwargs={},
    nuts_kwargs={},
    one_step_kwargs={},
)

seed_runner = build_seed_runner(
    evaluator=settings.evaluator,
    model=settings.model,
    plan=settings.loop_plan,
    config=seed_runner_config,
)
EXPECTED_SEED_RUNNER_OUTPUT = (
    jnp.array([[0.33719104], [0.29733259]]),
    jnp.array([[0.38148697], [0.30788488]]),
    jnp.array([[1.48917591, 0.77000132], [1.5183811, 0.78757542]]),
    jnp.array([[0.10400199, 0.00070576], [0.10151759, -0.0000813]]),
)


def test_seed_runner_matches_recorded_output() -> None:
    outputs = seed_runner(0, settings.initial_theta_stage0)

    for observed, expected in zip(outputs, EXPECTED_SEED_RUNNER_OUTPUT, strict=True):
        assert jnp.allclose(observed, expected, rtol=1e-4, atol=1e-5)

from __future__ import annotations

import jax.numpy as jnp

from degenerate_diffusion.estimation.loop_estimation_algorithm_new import (
    SeedRunnerConfig,
    build_seed_runner,
)
from degenerate_diffusion.internal.fn_model_pipeline import create_fnmodel_setup


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

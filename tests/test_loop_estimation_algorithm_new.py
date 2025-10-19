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
    stage0_last, final_last = runner(0, theta0)

    # Check shapes are consistent
    for a, b in zip(stage0_last, true_theta, strict=True):
        assert a.shape == jnp.asarray(b).shape
    for a, b in zip(final_last, true_theta, strict=True):
        assert a.shape == jnp.asarray(b).shape

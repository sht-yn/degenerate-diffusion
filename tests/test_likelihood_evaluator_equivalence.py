from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax import Array

from degenerate_diffusion.internal.fn_model_pipeline import (
    ThetaTuple,
    create_fnmodel_setup,
)

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from degenerate_diffusion.processes.degenerate_diffusion_process import (
        DegenerateDiffusionProcess,
    )


def _simulate_small_series(
    process: DegenerateDiffusionProcess, true_theta: ThetaTuple
) -> tuple[Array, Array, Array]:
    # Small and fast simulation for testing equivalence
    t_max = 0.2
    h = 0.1
    burn_out = 0.0
    dt = 0.01
    x_series, y_series = process.simulate(
        true_theta,
        t_max=t_max,
        h=h,
        burn_out=burn_out,
        seed=0,
        dt=dt,
    )
    # Ensure arrays are JAX types so mypy and runtime agree
    return jnp.asarray(x_series), jnp.asarray(y_series), jnp.asarray(h)


def test_quasi_likelihood_stateless_matches_closure() -> None:
    setup = create_fnmodel_setup()
    evaluator = setup.evaluator
    process = setup.process

    # Use small simulation to keep the test fast
    x_series, y_series, h = _simulate_small_series(process, setup.true_theta)
    h_scalar = float(h)

    # Choose k small but > 1 to activate S-corrections
    k = 2

    theta1, theta2, theta3 = setup.true_theta
    theta_1_bar, theta_2_bar, theta_3_bar = theta1, theta2, theta3

    # Closure-based evaluators
    l1p_closure = evaluator.make_quasi_likelihood_l1_prime_evaluator(
        x_series, y_series, h_scalar, k
    )
    l1_closure = evaluator.make_quasi_likelihood_l1_evaluator(x_series, y_series, h_scalar, k)
    l2_closure = evaluator.make_quasi_likelihood_l2_evaluator(x_series, y_series, h_scalar, k)
    l3_closure = evaluator.make_quasi_likelihood_l3_evaluator(x_series, y_series, h_scalar, k)

    v_l1p_closure = l1p_closure(theta1, theta_1_bar, theta_2_bar, theta_3_bar)
    v_l1_closure = l1_closure(theta1, theta_1_bar, theta_2_bar, theta_3_bar)
    v_l2_closure = l2_closure(theta2, theta_1_bar, theta_2_bar, theta_3_bar)
    v_l3_closure = l3_closure(theta3, theta_1_bar, theta_2_bar, theta_3_bar)

    # Stateless evaluators
    l1p_stateless = evaluator.make_stateless_quasi_l1_prime_evaluator(k=k)
    l1_stateless = evaluator.make_stateless_quasi_l1_evaluator(k=k)
    l2_stateless = evaluator.make_stateless_quasi_l2_evaluator(k=k)
    l3_stateless = evaluator.make_stateless_quasi_l3_evaluator(k=k)

    v_l1p_stateless = l1p_stateless(
        theta1, theta_1_bar, theta_2_bar, theta_3_bar, x_series, y_series, h
    )
    v_l1_stateless = l1_stateless(
        theta1, theta_1_bar, theta_2_bar, theta_3_bar, x_series, y_series, h
    )
    v_l2_stateless = l2_stateless(
        theta2, theta_1_bar, theta_2_bar, theta_3_bar, x_series, y_series, h
    )
    v_l3_stateless = l3_stateless(
        theta3, theta_1_bar, theta_2_bar, theta_3_bar, x_series, y_series, h
    )

    # Compare as scalars
    np.testing.assert_allclose(
        float(np.asarray(v_l1p_closure)), float(np.asarray(v_l1p_stateless)), rtol=0.0, atol=1e-7
    )
    np.testing.assert_allclose(
        float(np.asarray(v_l1_closure)), float(np.asarray(v_l1_stateless)), rtol=0.0, atol=1e-7
    )
    np.testing.assert_allclose(
        float(np.asarray(v_l2_closure)), float(np.asarray(v_l2_stateless)), rtol=0.0, atol=1e-7
    )
    np.testing.assert_allclose(
        float(np.asarray(v_l3_closure)), float(np.asarray(v_l3_stateless)), rtol=0.0, atol=1e-7
    )

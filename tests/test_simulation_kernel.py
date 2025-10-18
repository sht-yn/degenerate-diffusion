"""Unit tests for the make_simulation_kernel helper."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp

from degenerate_diffusion import DegenerateDiffusionProcess


def _build_linear_process() -> DegenerateDiffusionProcess:
    x_sym, y_sym = sp.symbols("x y")

    x = sp.Array([x_sym])
    y = sp.Array([y_sym])
    theta_1 = sp.Array([sp.symbols("theta_10")])
    theta_2 = sp.Array([sp.symbols("theta_20"), sp.symbols("theta_21")])
    theta_3 = sp.Array([sp.symbols("theta_30")])

    A_expr = sp.Array([-theta_2[0] * x_sym - theta_2[1] * y_sym])
    B_expr = sp.Array([[theta_1[0]]])
    H_expr = sp.Array([theta_3[0] * x_sym])

    return DegenerateDiffusionProcess(
        x=x,
        y=y,
        theta_1=theta_1,
        theta_2=theta_2,
        theta_3=theta_3,
        A=A_expr,
        B=B_expr,
        H=H_expr,
    )


def test_kernel_matches_simulate_output() -> None:
    """make_simulation_kernel should mirror the original simulate outputs."""
    process = _build_linear_process()

    true_theta = (
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array([1.0, 1.0], dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
    )
    t_max = 1.0
    h = 0.1
    burn_out = 0.1
    dt = 0.01
    seed = 123

    kernel = process.make_simulation_kernel(
        t_max=t_max,
        h=h,
        burn_out=burn_out,
        dt=dt,
    )

    key = jax.random.PRNGKey(seed)
    x0 = jnp.zeros((1,), dtype=jnp.float32)
    y0 = jnp.zeros((1,), dtype=jnp.float32)

    x_kernel, y_kernel = kernel(true_theta, key, x0, y0)

    x_sim, y_sim = process.simulate(
        true_theta=true_theta,
        t_max=t_max,
        h=h,
        burn_out=burn_out,
        seed=seed,
        x0=np.zeros((1,), dtype=np.float32),
        y0=np.zeros((1,), dtype=np.float32),
        dt=dt,
    )

    assert x_kernel.shape == x_sim.shape
    assert y_kernel.shape == y_sim.shape
    assert x_kernel.dtype == x_sim.dtype
    assert y_kernel.dtype == y_sim.dtype

    np.testing.assert_allclose(np.asarray(x_kernel), np.asarray(x_sim))
    np.testing.assert_allclose(np.asarray(y_kernel), np.asarray(y_sim))

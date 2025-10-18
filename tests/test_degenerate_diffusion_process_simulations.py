"""Smoke tests for ``DegenerateDiffusionProcess.simulate`` using notebook models."""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp

from degenerate_diffusion import DegenerateDiffusionProcess


def _assert_simulation_output(
    x_series: jnp.ndarray,
    y_series: jnp.ndarray,
    expected_x_shape: tuple[int, ...],
    expected_y_shape: tuple[int, ...],
) -> None:
    """Assert shapes and finite values for simulation outputs."""
    assert x_series.shape == expected_x_shape
    assert y_series.shape == expected_y_shape

    x_np = np.asarray(x_series)
    y_np = np.asarray(y_series)

    assert np.all(np.isfinite(x_np))
    assert np.all(np.isfinite(y_np))


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


def _build_fn_process() -> DegenerateDiffusionProcess:
    x_sym, y_sym = sp.symbols("x y")

    x = sp.Array([x_sym])
    y = sp.Array([y_sym])
    theta_1 = sp.Array([sp.symbols("theta_10")])
    theta_2 = sp.Array([sp.symbols("theta_20"), sp.symbols("theta_21")])
    theta_3 = sp.Array([sp.symbols("theta_30"), sp.symbols("theta_31")])

    A_expr = sp.Array([theta_2[0] * y_sym - x_sym + theta_2[1]])
    B_expr = sp.Array([[theta_1[0]]])
    H_expr = sp.Array([(y_sym - y_sym**3 - x_sym + theta_3[1]) / theta_3[0]])

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


def _build_dx3_dy2_process() -> DegenerateDiffusionProcess:
    x_symbols = sp.symbols("x_0:3")
    y_symbols = sp.symbols("y_0:2")

    x = sp.Array(x_symbols)
    y = sp.Array(y_symbols)

    sigma_symbols = sp.symbols("sigma_1 sigma_2 sigma_3")
    alpha_symbols = sp.symbols("alpha_1 alpha_2 alpha_3")
    beta_symbols = sp.symbols("beta_11 beta_12 beta_21 beta_22 beta_31 beta_32")
    gamma_symbols = sp.symbols("gamma_11 gamma_12 gamma_13 gamma_21 gamma_22 gamma_23")
    delta_sym = sp.symbols("delta")
    s_symbols = sp.symbols("s_1 s_2")

    beta_matrix = sp.Matrix(
        [
            [beta_symbols[0], beta_symbols[1]],
            [beta_symbols[2], beta_symbols[3]],
            [beta_symbols[4], beta_symbols[5]],
        ]
    )
    diag_alpha = sp.diag(*alpha_symbols)
    g_matrix = sp.Matrix(
        [
            [gamma_symbols[0], gamma_symbols[1], gamma_symbols[2]],
            [gamma_symbols[3], gamma_symbols[4], gamma_symbols[5]],
        ]
    )
    i2 = sp.eye(2)

    drift_vec = -diag_alpha * sp.Matrix(x) + beta_matrix * sp.Matrix(y)
    obs_vec = g_matrix * sp.Matrix(x) - delta_sym * i2 * sp.Matrix(y) + sp.Matrix(s_symbols)

    A_expr = sp.Array([drift_vec[i, 0] for i in range(3)])
    B_expr = sp.Array(sp.diag(*sigma_symbols))
    H_expr = sp.Array([obs_vec[i, 0] for i in range(2)])

    theta_1 = sp.Array(sigma_symbols)
    theta_2 = sp.Array((*alpha_symbols, *beta_symbols))
    theta_3 = sp.Array((*gamma_symbols, delta_sym, *s_symbols))

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


@pytest.mark.parametrize(
    (
        "builder",
        "true_theta",
        "t_max",
        "h",
        "burn_out",
        "expected_x_shape",
        "expected_y_shape",
        "dt",
    ),
    [
        (
            _build_linear_process,
            (
                jnp.array([1.0], dtype=jnp.float32),
                jnp.array([1.0, 1.0], dtype=jnp.float32),
                jnp.array([1.0], dtype=jnp.float32),
            ),
            2.0,
            0.2,
            0.2,
            (11, 1),
            (11, 1),
            0.01,
        ),
        (
            _build_fn_process,
            (
                jnp.array([0.3], dtype=jnp.float32),
                jnp.array([1.5, 0.8], dtype=jnp.float32),
                jnp.array([0.1, 0.0], dtype=jnp.float32),
            ),
            1.0,
            0.05,
            0.05,
            (21, 1),
            (21, 1),
            0.01,
        ),
        (
            _build_dx3_dy2_process,
            (
                jnp.array([0.5, 0.6, 0.7], dtype=jnp.float32),
                jnp.array([1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], dtype=jnp.float32),
                jnp.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=jnp.float32),
            ),
            1.0,
            0.05,
            0.05,
            (21, 3),
            (21, 2),
            0.01,
        ),
    ],
)
def test_simulation_runs_for_notebook_models(
    builder: Callable[[], DegenerateDiffusionProcess],
    true_theta: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    t_max: float,
    h: float,
    burn_out: float,
    expected_x_shape: tuple[int, ...],
    expected_y_shape: tuple[int, ...],
    dt: float,
) -> None:
    """Each notebook model should simulate successfully with representative parameters."""
    assert callable(builder)

    process = builder()

    x_series, y_series = process.simulate(
        true_theta=true_theta,
        t_max=t_max,
        h=h,
        burn_out=burn_out,
        seed=0,
        dt=dt,
    )

    _assert_simulation_output(x_series, y_series, expected_x_shape, expected_y_shape)

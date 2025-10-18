from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

import jax
import numpy as np
import sympy as sp
from jax import lax, numpy as jnp
from sympy import lambdify

if TYPE_CHECKING:
    from collections.abc import Callable

from degenerate_diffusion.utils.symbolic_artifact import SymbolicArtifact


@dataclass(frozen=True)
class DegenerateDiffusionProcess:
    """Container for diffusion and observation dynamics defined with SymPy.

    SymPy で定義した拡散項と観測項を JAX で扱うためのコンテナ.
    """

    x: sp.Array
    y: sp.Array
    theta_1: sp.Array
    theta_2: sp.Array
    theta_3: sp.Array
    A: sp.Array
    B: sp.Array
    H: sp.Array
    A_func: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray] = field(
        init=False,
        repr=False,
    )
    B_func: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray] = field(
        init=False,
        repr=False,
    )
    H_func: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Create JAX callables from the provided symbolic expressions.

        SymPy の式から JAX 関数を構築する.
        """
        common_args = (self.x, self.y)

        A_func = lambdify((*common_args, self.theta_2), self.A, modules="jax")
        B_func = lambdify((*common_args, self.theta_1), self.B, modules="jax")
        H_func = lambdify((*common_args, self.theta_3), self.H, modules="jax")

        object.__setattr__(
            self,
            "A",
            SymbolicArtifact(expr=self.A, func=A_func),
        )
        object.__setattr__(
            self,
            "B",
            SymbolicArtifact(expr=self.B, func=B_func),
        )
        object.__setattr__(
            self,
            "H",
            SymbolicArtifact(expr=self.H, func=H_func),
        )

        object.__setattr__(self, "A_func", A_func)
        object.__setattr__(self, "B_func", B_func)
        object.__setattr__(self, "H_func", H_func)

    @partial(jax.jit, static_argnums=(0, 5, 6, 9, 10))
    def _simulate_jax_core(
        self,  # 0: static
        theta_1_val: jnp.ndarray,  # 1
        theta_2_val: jnp.ndarray,  # 2
        theta_3_val: jnp.ndarray,  # 3
        key: jax.Array,  # 4
        t_max: float,  # 5: static
        burn_out: float,  # 6: static
        x0_val: jnp.ndarray,  # 7
        y0_val: jnp.ndarray,  # 8
        dt: float,  # 9: static
        step_stride_static: int,  # 10: static
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Run JAX based Euler Maruyama scan for the diffusion process.

        Euler Maruyama 法による離散化を JAX の scan で実行する.
        """
        # Calculate total_steps_for_scan as Python int from static args
        _total_steps_for_scan_py_float = (t_max + burn_out) / dt
        total_steps_for_scan_py = round(_total_steps_for_scan_py_float)  # Python int
        total_steps_for_scan_py = max(0, total_steps_for_scan_py)  # Ensure non-negative for length

        # Calculate start_index_py as Python int from static args
        _start_index_py_float = burn_out / dt
        start_index_py = round(_start_index_py_float)  # Python int

        # Bound start_index_py using total_steps_for_scan_py (which is also Python int)
        start_index_py = min(start_index_py, total_steps_for_scan_py)  # Python min
        start_index_py = max(0, start_index_py)  # Python max, ensure non-negative

        r = self.B.expr.shape[1]

        dt_array = jnp.asarray(dt, dtype=theta_1_val.dtype)

        def em_step(
            carry: tuple[jnp.ndarray, jnp.ndarray, jax.Array],
            _: None,
        ) -> tuple[
            tuple[jnp.ndarray, jnp.ndarray, jax.Array],
            tuple[jnp.ndarray, jnp.ndarray],
        ]:
            xt, yt, current_key = carry
            key_dW, next_key_for_loop = jax.random.split(current_key)

            A_val = self.A_func(xt, yt, theta_2_val)
            B_val = self.B_func(xt, yt, theta_1_val)
            H_val = self.H_func(xt, yt, theta_3_val)

            dW = jax.random.normal(key_dW, (r,), dtype=theta_1_val.dtype) * jnp.sqrt(dt_array)
            diffusion_term = jnp.dot(B_val, dW)

            xt_next = xt + A_val * dt_array + diffusion_term
            yt_next = yt + H_val * dt_array

            return (xt_next, yt_next, next_key_for_loop), (xt_next, yt_next)

        initial_carry = (x0_val, y0_val, key)

        # lax.scan uses the Python int total_steps_for_scan_py for its length
        _, (x_results, y_results) = lax.scan(
            em_step, initial_carry, None, length=total_steps_for_scan_py
        )

        x_all = jnp.concatenate((jnp.expand_dims(x0_val, 0), x_results), axis=0)
        y_all = jnp.concatenate((jnp.expand_dims(y0_val, 0), y_results), axis=0)

        # Slicing uses Python integers derived from static arguments
        x_series = x_all[start_index_py::step_stride_static]
        y_series = y_all[start_index_py::step_stride_static]

        return x_series, y_series

    def simulate(
        self,
        true_theta: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        t_max: float,
        h: float,
        burn_out: float,
        seed: int = 42,
        x0: np.ndarray | jnp.ndarray | None = None,
        y0: np.ndarray | jnp.ndarray | None = None,
        dt: float = 0.001,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Simulate state and observation series via Euler Maruyama.

        Euler Maruyama 法で状態系列と観測系列を生成する.
        """
        if dt <= 0:
            msg = "dt must be positive."
            raise ValueError(msg)
        if h <= 0:
            msg = "h must be positive."
            raise ValueError(msg)
        if t_max < 0:
            msg = "t_max must be non-negative."
            raise ValueError(msg)
        if burn_out < 0:
            msg = "burn_out must be non-negative."
            raise ValueError(msg)

        if not np.isclose(h % dt, 0, atol=1e-8) and not np.isclose(h % dt, dt, atol=1e-8):
            print(
                f"Warning: h ({h}) is not an integer multiple of dt ({dt}). "
                "Thinning interval might be slightly inaccurate due to rounding."
            )

        step_stride_float_py = h / dt
        step_stride_py = max(1, round(step_stride_float_py))  # Use built-in round()

        key = jax.random.PRNGKey(seed)

        d_x = self.x.shape[0]
        d_y = self.y.shape[0]

        theta_1, theta_2, theta_3 = true_theta

        theta_1_jnp = jnp.asarray(theta_1)
        theta_2_jnp = jnp.asarray(theta_2)
        theta_3_jnp = jnp.asarray(theta_3)

        target_dtype = jnp.result_type(theta_1_jnp, theta_2_jnp, theta_3_jnp)

        theta_1_jnp = theta_1_jnp.astype(target_dtype)
        theta_2_jnp = theta_2_jnp.astype(target_dtype)
        theta_3_jnp = theta_3_jnp.astype(target_dtype)

        x0_jnp = (
            jnp.zeros((d_x,), dtype=target_dtype)
            if x0 is None
            else jnp.reshape(jnp.asarray(x0, dtype=target_dtype), (d_x,))
        )
        y0_jnp = (
            jnp.zeros((d_y,), dtype=target_dtype)
            if y0 is None
            else jnp.reshape(jnp.asarray(y0, dtype=target_dtype), (d_y,))
        )

        x_series_jax, y_series_jax = self._simulate_jax_core(
            theta_1_jnp,
            theta_2_jnp,
            theta_3_jnp,
            key,
            t_max,
            burn_out,
            x0_jnp,
            y0_jnp,
            dt,
            step_stride_py,
        )
        return x_series_jax, y_series_jax

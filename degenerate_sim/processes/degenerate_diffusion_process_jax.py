# %%
# dataclassをインポート
from dataclasses import dataclass
from functools import partial

# import math # roundは組み込み関数のため不要
# jax関連のライブラリを全てインポート
import jax
import numpy as np
import sympy as sp
from jax import lax  # For jax.lax.scan
from jax import numpy as jnp
from sympy import lambdify, symbols


# %%
@dataclass(frozen=True)
class DegenerateDiffusionProcess:
    """多次元の拡散過程と観測過程を扱うためのクラス。.

    記号計算 (sympy) を用いてモデルを定義し、
    数値計算 (numpy/jax) のための関数を lambdify で生成する。
    """

    x: sp.Array
    y: sp.Array
    theta_1: sp.Array
    theta_2: sp.Array
    theta_3: sp.Array
    A: sp.Array
    B: sp.Array
    H: sp.Array

    def __post_init__(self):
        common_args = (self.x, self.y)
        try:
            object.__setattr__(
                self, "A_func", lambdify((*common_args, self.theta_2), self.A, modules="jax")
            )
            object.__setattr__(
                self, "B_func", lambdify((*common_args, self.theta_1), self.B, modules="jax")
            )
            object.__setattr__(
                self, "H_func", lambdify((*common_args, self.theta_3), self.H, modules="jax")
            )
        except Exception as e:
            print(f"Error during lambdification in __post_init__: {e}")
            raise

    @partial(jax.jit, static_argnums=(0, 5, 6, 9, 10))
    def _simulate_jax_core(
        self,  # 0: static
        theta_1_val: jnp.ndarray,  # 1
        theta_2_val: jnp.ndarray,  # 2
        theta_3_val: jnp.ndarray,  # 3
        key: jax.random.PRNGKey,  # 4
        t_max: float,  # 5: static
        burn_out: float,  # 6: static
        x0_val: jnp.ndarray,  # 7
        y0_val: jnp.ndarray,  # 8
        dt: float,  # 9: static
        step_stride_static: int,  # 10: static
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """JAX-optimized core simulation loop using Euler-Maruyama.
        lax.scan length (total_steps_for_scan_py) and slice indices (start_index_py, step_stride_static)
        are Python ints derived from static args.
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

        self.x.shape[0]
        self.y.shape[0]
        r = self.B.shape[1]

        def em_step(carry, _):
            xt, yt, current_key = carry
            key_dW, next_key_for_loop = jax.random.split(current_key)

            A_val = self.A_func(xt, yt, theta_2_val)
            B_val = self.B_func(xt, yt, theta_1_val)
            H_val = self.H_func(xt, yt, theta_3_val)

            dW = jax.random.normal(key_dW, (r,)) * jnp.sqrt(dt)
            diffusion_term = jnp.dot(B_val, dW)

            xt_next = xt + A_val * dt + diffusion_term
            yt_next = yt + H_val * dt

            return (xt_next, yt_next, next_key_for_loop), (xt_next, yt_next)

        initial_carry = (x0_val, y0_val, key)

        # lax.scan uses the Python int total_steps_for_scan_py for its length
        final_carry, (x_results, y_results) = lax.scan(
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
        true_theta: tuple[np.ndarray, np.ndarray, np.ndarray],
        t_max: float,
        h: float,
        burn_out: float,
        seed: int = 42,
        x0: np.ndarray | None = None,
        y0: np.ndarray | None = None,
        dt: float = 0.001,
        output_dtype: str = "numpy",  # numpy or jax
    ) -> tuple[np.ndarray, np.ndarray]:
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
                f"Warning: h ({h}) is not an integer multiple of dt ({dt}). Thinning interval might be slightly inaccurate due to rounding."
            )

        step_stride_float_py = h / dt
        step_stride_py = max(1, round(step_stride_float_py))  # Use built-in round()

        key = jax.random.PRNGKey(seed)

        d_x = self.x.shape[0]
        d_y = self.y.shape[0]

        theta_1_jnp, theta_2_jnp, theta_3_jnp = [
            jnp.asarray(th, dtype=jnp.float32) for th in true_theta
        ]

        _x0_np = np.zeros(d_x, dtype=np.float32) if x0 is None else np.asarray(x0, dtype=np.float32)
        _y0_np = np.zeros(d_y, dtype=np.float32) if y0 is None else np.asarray(y0, dtype=np.float32)

        x0_jnp = jnp.reshape(_x0_np, (d_x,))
        y0_jnp = jnp.reshape(_y0_np, (d_y,))

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

        if output_dtype == "jax":
            return x_series_jax, y_series_jax
        if output_dtype == "numpy":
            return np.array(x_series_jax), np.array(y_series_jax)
        return None


# %%
if __name__ == "__main__":
    x_sym = sp.Array([symbols("x_0")])
    y_sym = sp.Array([symbols("y_0")])
    theta1_sym = sp.Array([symbols("sigma")])
    theta2_sym = sp.Array([symbols("kappa")])
    theta3_sym = sp.Array([symbols("mu_y")])

    A_expr = sp.Array([-theta2_sym[0] * x_sym[0]])
    B_expr = sp.Array([[theta1_sym[0]]])
    H_expr = sp.Array([theta3_sym[0] * x_sym[0]])

    process = DegenerateDiffusionProcess(
        x=x_sym,
        y=y_sym,
        theta_1=theta1_sym,
        theta_2=theta2_sym,
        theta_3=theta3_sym,
        A=A_expr,
        B=B_expr,
        H=H_expr,
    )
    # %%
    true_sigma = np.array([0.5])
    true_kappa = np.array([1.0])
    true_mu_y = np.array([0.2])
    true_thetas = (true_sigma, true_kappa, true_mu_y)

    T_MAX = 100.0
    H_STEP = 0.1
    BURN_OUT = 100.0
    DT_SIM = 0.001

    # %%

    print("Simulating with JAX-optimized method...")
    import time

    start_time = time.time()
    for i in range(10):
        current_t_max = T_MAX
        current_h_step = H_STEP
        print(f"Running simulation with T_MAX = {current_t_max}, H_STEP = {current_h_step}")
        x_data, y_data = process.simulate(
            true_theta=true_thetas,
            t_max=current_t_max,
            h=current_h_step,
            burn_out=BURN_OUT,
            dt=DT_SIM,
            x0=np.array([0.0]),
            y0=np.array([0.0]),
            seed=1 + i,
            output_dtype="numpy",  # "jax" or "numpy"
        )
        print(
            f"Simulation finished for T_MAX = {current_t_max}, H_STEP = {current_h_step}. Generated data shapes:"
        )
        print(f"x_series shape: {x_data.shape}")
        print(f"y_series shape: {y_data.shape}")

        if i == 0:
            try:
                import matplotlib.pyplot as plt

                time_axis = np.arange(x_data.shape[0]) * current_h_step

                fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

                ax[0].plot(time_axis, x_data[:, 0], label=f"{x_sym[0]!s} (Simulated)")
                ax[0].set_ylabel(f"State {x_sym[0]!s}")
                ax[0].grid(True)
                ax[0].legend()

                ax[1].plot(time_axis, y_data[:, 0], label=f"{y_sym[0]!s} (Simulated)")
                ax[1].set_xlabel("Time")
                ax[1].set_ylabel(f"Observation {y_sym[0]!s}")
                ax[1].grid(True)
                ax[1].legend()

                plt.suptitle(
                    f"Simulated Degenerate Diffusion Process (JAX Optimized, T_MAX={current_t_max}, H_STEP={current_h_step})"
                )
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.show()
            except ImportError:
                print("\nMatplotlib not found. Skipping plot.")
            except Exception as e:
                print(f"\nError during plotting: {e}")
    end_time = time.time()
    print(f"Total simulation time for 100 runs: {end_time - start_time:.2f} seconds")
# %%

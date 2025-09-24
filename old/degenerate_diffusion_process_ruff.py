# %%
# Standard library imports
import warnings
from dataclasses import dataclass

# Third-party imports
import numpy as np
import sympy as sp
from sympy import lambdify, symbols

# Note: Original code also imported jax, jax.numpy, and functools.partial,
# but they were flagged as unused (F401).
# Original code also imported typing.Tuple, Optional, Sequence and dataclasses.dataclass,
# which are now handled by modern typing syntax and direct imports.


# %%
@dataclass(frozen=True)
class DegenerateDiffusionProcess:
    """Class for handling multidimensional diffusion and observation processes.

    This class uses symbolic computation (sympy) to define the model and
    generates functions for numerical computation (numpy) using lambdify.

    Attributes:
        x (sp.Array): State variables (sympy symbol array, shape=(d_x,)).
        y (sp.Array): Observation variables (sympy symbol array, shape=(d_y,)).
        theta_1 (sp.Array): Parameter set 1 (sympy symbol array).
        theta_2 (sp.Array): Parameter set 2 (sympy symbol array).
        theta_3 (sp.Array): Parameter set 3 (sympy symbol array).
        A (sp.Array): Drift term's x component (sympy expression array, shape=(d_x,)).
        B (sp.Array): Diffusion term (sympy expression array, shape=(d_x, r)).
        H (sp.Array): Drift term for the observation process (sympy expression array, shape=(d_y,)).
        A_func: NumPy function to calculate drift A (auto-generated).
        B_func: NumPy function to calculate diffusion term B (auto-generated).
        H_func: NumPy function to calculate observation drift H (auto-generated).

    """

    x: sp.Array
    y: sp.Array
    theta_1: sp.Array
    theta_2: sp.Array
    theta_3: sp.Array
    A: sp.Array
    B: sp.Array
    H: sp.Array

    def __post_init__(self) -> None:
        """Performs post-initialization setup.

        Calculates derived attributes and generates functions using lambdify.
        Uses object.__setattr__ due to frozen=True.
        """
        # --- Generate numerical functions using lambdify ---
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
        except Exception:
            # Consider logging the error instead of printing if this were a library
            # For a script, print might be acceptable, but Ruff advises against it.
            # Error will propagate due to the raise statement.
            # print(f"Error during lambdification in __post_init__: {e}")
            raise

    # --- Simulation function ---
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
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generates time series data for state x and observation y using Euler-Maruyama.

        Parameters
        ----------
        true_theta : tuple[np.ndarray, np.ndarray, np.ndarray]
            True values for (theta_1_val, theta_2_val, theta_3_val).
        t_max : float
            Duration of the time series data to generate (excluding burn_out).
        h : float
            Time step width of the generated data (after thinning).
        burn_out : float
            Initial period to discard to reach a stationary state.
        seed : int, optional
            Seed for the random number generator, by default 42.
        x0 : np.ndarray | None, optional
            Initial value vector for x. Zero vector if None, by default None.
        y0 : np.ndarray | None, optional
            Initial value vector for y. Zero vector if None, by default None.
        dt : float, optional
            Internal small time step for simulation, by default 0.001.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (x_series, y_series)
            x_series: Time series data for x (shape=(T, d_x)).
            y_series: Time series data for y (shape=(T, d_y)).
            Where T is an integer close to t_max / h.

        """
        np.random.seed(seed)
        theta_1_val, theta_2_val, theta_3_val = true_theta
        total_steps = int(np.round((t_max + burn_out) / dt))
        d_x = self.x.shape[0]
        d_y = self.y.shape[0]
        r = self.B.shape[1]  # Number of Wiener processes

        # Check dt and h relationship and calculate step_stride
        if not np.isclose(h % dt, 0, atol=1e-8) and not np.isclose(h % dt, dt, atol=1e-8):
            warnings.warn(
                f"Warning: h ({h}) may not be an integer multiple of dt ({dt}). "
                "Thinning interval might be slightly inaccurate due to rounding.",
                UserWarning,
                stacklevel=2,
            )
        step_stride = max(1, int(np.round(h / dt)))

        x_all = np.zeros((total_steps + 1, d_x))
        y_all = np.zeros((total_steps + 1, d_y))
        x_all[0] = x0 if x0 is not None else np.zeros(d_x)
        y_all[0] = y0 if y0 is not None else np.zeros(d_y)

        for t in range(total_steps):
            xt, yt = x_all[t], y_all[t]
            try:
                A_val = self.A_func(xt, yt, theta_2_val)
                B_val = self.B_func(xt, yt, theta_1_val)
                H_val = self.H_func(xt, yt, theta_3_val)
            except Exception:
                # Error will propagate. Original prints for debugging are removed.
                # Consider logging detailed info if needed:
                # error_msg = (
                #     f"Error evaluating A, B, or H function at step {t}: {e}. "
                #     f"xt={xt}, yt={yt}, theta_1={theta_1_val}, "
                #     f"theta_2={theta_2_val}, theta_3={theta_3_val}"
                # )
                # logging.error(error_msg) # If logging is set up
                raise

            dW = np.random.randn(r) * np.sqrt(dt)
            diffusion_term = np.dot(B_val, dW)

            if diffusion_term.shape != (d_x,):
                error_msg = (
                    f"Shape mismatch in diffusion term: expected ({d_x},), "
                    f"got {diffusion_term.shape}"
                )
                raise ValueError(error_msg)

            x_all[t + 1] = xt + A_val * dt + diffusion_term
            y_all[t + 1] = yt + H_val * dt
            # If observation noise is needed:
            # e.g., if dy = H dt + G dV, create G_func in __post_init__
            # G_val = self.G_func(xt, yt, theta_g_val)
            # dV = np.random.randn(d_y) * np.sqrt(dt)
            # y_all[t + 1] = yt + H_val * dt + np.dot(G_val, dV)

        # Apply burn-out and thinning
        start_index = int(np.round(burn_out / dt))

        x_series = x_all[start_index::step_stride]
        y_series = y_all[start_index::step_stride]

        return x_series, y_series


# %%
# --- Example usage (illustrative) ---
if __name__ == "__main__":
    # Example: 1D Ornstein-Uhlenbeck process (dx = -theta2*x dt + theta1 dW)
    # Observation y has x-dependent drift (dy = theta3*x dt)

    # Symbolic definitions
    x_sym = sp.Array([symbols("x_0")])
    y_sym = sp.Array([symbols("y_0")])  # Observation variable, can be empty if not used by A or B
    theta1_sym = sp.Array([symbols("sigma")])  # diffusion coefficient
    theta2_sym = sp.Array([symbols("kappa")])  # reversion rate
    theta3_sym = sp.Array([symbols("mu")])  # observation drift coefficient

    # Model definition
    A_expr = sp.Array([-theta2_sym[0] * x_sym[0]])
    B_expr = sp.Array([[theta1_sym[0]]])  # shape (d_x, r) = (1, 1)
    H_expr = sp.Array([theta3_sym[0] * x_sym[0]])  # shape (d_y,) = (1,)

    # Instance creation
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

    # True parameter values
    true_sigma = np.array([0.5])
    true_kappa = np.array([1.0])
    true_mu = np.array([0.2])  # Changed example for H to depend on x
    true_thetas = (true_sigma, true_kappa, true_mu)

    # Simulation parameters
    T_MAX = 1000.0
    H_STEP = 0.1  # Thinned step width
    BURN_OUT = 100.0
    DT_SIM = 0.01  # Internal simulation step width
    # %%
    print("Simulating...")
    x_data, y_data = process.simulate(
        true_theta=true_thetas,
        t_max=T_MAX,
        h=H_STEP,
        burn_out=BURN_OUT,
        dt=DT_SIM,
        x0=np.array([0.0]),
        y0=np.array([0.0]),  # Initial y value
        seed=123,
    )

    print("Simulation finished. Generated data shapes:")  # F541: removed f from f""
    print(f"x_series shape: {x_data.shape}")  # (T, d_x)
    print(f"y_series shape: {y_data.shape}")  # (T, d_y)

    # Simple plotting example (requires matplotlib)
    try:
        import matplotlib.pyplot as plt

        time = np.arange(x_data.shape[0]) * H_STEP
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        ax[0].plot(time, x_data[:, 0], label="x_0 (Simulated)")
        ax[0].set_ylabel("State x")
        ax[0].grid(visible=True)  # FBT003: Use keyword argument
        ax[0].legend()

        ax[1].plot(time, y_data[:, 0], label="y_0 (Simulated)")
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Observation y")
        ax[1].grid(visible=True)  # FBT003: Use keyword argument
        ax[1].legend()

        plt.suptitle("Simulated Degenerate Diffusion Process")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
        plt.show()
    except ImportError:
        print("\nMatplotlib not found. Skipping plot.")
# %%

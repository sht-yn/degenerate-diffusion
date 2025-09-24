# %%
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp  # JAX NumPyをインポート
import sympy as sp
from jax import lax
from sympy import (
    Array,
    Basic,
    Expr,
    Matrix,
    S,
    derive_by_array,
    factorial,
    lambdify,
    log,
    oo,
    srepr,
    tensorproduct,
)
from sympy import (
    zeros as sp_zeros,
)
from sympy.matrices.common import NonInvertibleMatrixError

from degenerate_sim.utils.einsum_sympy import einsum_sympy

if TYPE_CHECKING:
    from degenerate_sim.processes.degenerate_diffusion_process_jax import DegenerateDiffusionProcess

# --- 定数 ---
INVALID_SREPR_KEY = "invalid_sympy_srepr"


# %%
class LikelihoodEvaluator:
    """DegenerateDiffusionProcess モデルに基づき疑似尤度を計算するクラス。
    主要な記号的数式も属性として保持する。.
    """

    def __init__(self, model: "DegenerateDiffusionProcess") -> None:
        """LikelihoodEvaluator のインスタンスを初期化."""
        self.model = model
        self._L_cache: dict[tuple[int, str], Array] = {}
        self._L0_func_cache: dict[tuple[int, str], Callable] = {}
        self._S_func_cache: dict[int, tuple[Callable, ...]] = {}

        # --- モデル属性への参照 ---
        self.x = model.x
        self.y = model.y
        self.theta_1 = model.theta_1
        self.theta_2 = model.theta_2
        self.theta_3 = model.theta_3
        self.A = model.A  # sympy expression
        self.B = model.B  # sympy expression
        self.H = model.H  # sympy expression

        # --- モデルから lambdify 済み関数を取得 ---
        # これらは DegenerateDiffusionProcess クラスで modules="jax" で lambdify されていることを期待
        required_funcs = ["A_func", "B_func", "H_func"]
        if not all(
            hasattr(model, func_name) and callable(getattr(model, func_name))
            for func_name in required_funcs
        ):
            missing = [
                f
                for f in required_funcs
                if not hasattr(model, f) or not callable(getattr(model, f))
            ]
            msg = f"Provided model instance is missing required callable JAX functions: {missing}"
            raise AttributeError(msg)
        self.A_func = model.A_func
        self.B_func = model.B_func
        self.H_func = model.H_func

        common_args = (self.x, self.y)

        # --- 派生的な記号計算と lambdify (JAXへ変更) ---
        try:
            self.C_sym = einsum_sympy("ik,jk->ij", self.B, self.B)
            self.C_func = lambdify((*common_args, self.theta_1), self.C_sym, modules="jax")

            C_matrix = Matrix(self.C_sym)
            self.inv_C_expr = None
            self.log_det_C_expr = None
            try:
                self.inv_C_expr = Array(C_matrix.inv())
                self.inv_C_func = lambdify(
                    (*common_args, self.theta_1), self.inv_C_expr, modules="jax"
                )
            except NonInvertibleMatrixError:
                print(
                    "Warning: Symbolic matrix C is not invertible. inv_C_expr and inv_C_func set to None."
                )
                self.inv_C_func = None
            except Exception as e:
                print(
                    f"Warning: Error inverting C matrix or lambdifying: {e}. inv_C_expr and inv_C_func set to None."
                )
                self.inv_C_func = None

            try:
                det_C_val = C_matrix.det()
                if det_C_val.is_nonpositive:  # type: ignore
                    print(
                        "Warning: Symbolic det(C) is non-positive. log_det_C_expr and log_det_C_func set to None."
                    )
                    self.log_det_C_func = None
                elif det_C_val == 0:
                    print(
                        "Warning: Symbolic det(C) is zero. log_det_C_expr set to -oo (func to None)."
                    )
                    self.log_det_C_expr = -oo
                    self.log_det_C_func = None
                else:
                    self.log_det_C_expr = log(det_C_val)
                    self.log_det_C_func = lambdify(
                        (*common_args, self.theta_1), self.log_det_C_expr, modules="jax"
                    )
            except Exception as e:
                print(
                    f"Warning: Could not reliably determine sign of det(C) or log failed ({e}). log_det_C_expr and log_det_C_func set to None."
                )
                self.log_det_C_func = None

            self.partial_x_H_sym = derive_by_array(self.H, self.x)
            # self.partial_x_H_func = lambdify((*common_args, self.theta_3), self.partial_x_H_sym, modules="jax") # 必要に応じて

            self.V_sym = einsum_sympy(
                "ki,kl,lj->ij", self.partial_x_H_sym, self.C_sym, self.partial_x_H_sym
            )
            self.V_func = lambdify(
                (*common_args, self.theta_1, self.theta_3), self.V_sym, modules="jax"
            )

            V_matrix = Matrix(self.V_sym)
            self.inv_V_expr = None
            self.log_det_V_expr = None
            try:
                self.inv_V_expr = Array(V_matrix.inv())
                self.inv_V_func = lambdify(
                    (*common_args, self.theta_1, self.theta_3), self.inv_V_expr, modules="jax"
                )
            except NonInvertibleMatrixError:
                print(
                    "Warning: Symbolic matrix V is not invertible. inv_V_expr and inv_V_func set to None."
                )
                self.inv_V_func = None
            except Exception as e:
                print(
                    f"Warning: Error inverting V matrix or lambdifying: {e}. inv_V_expr and inv_V_func set to None."
                )
                self.inv_V_func = None

            try:
                det_V_val = V_matrix.det()
                if det_V_val.is_nonpositive:  # type: ignore
                    print(
                        "Warning: Symbolic det(V) is non-positive. log_det_V_expr and log_det_V_func set to None."
                    )
                    self.log_det_V_func = None
                elif det_V_val == 0:
                    print(
                        "Warning: Symbolic det(V) is zero. log_det_V_expr set to -oo (func to None)."
                    )
                    self.log_det_V_expr = -oo
                    self.log_det_V_func = None
                else:
                    self.log_det_V_expr = log(det_V_val)
                    self.log_det_V_func = lambdify(
                        (*common_args, self.theta_1, self.theta_3),
                        self.log_det_V_expr,
                        modules="jax",
                    )
            except Exception as e:
                print(
                    f"Warning: Could not reliably determine sign of det(V) or log failed ({e}). log_det_V_expr and log_det_V_func set to None."
                )
                self.log_det_V_func = None

            self.partial_x_H_transpose_inv_V_expr = None
            if self.inv_V_expr is not None:
                try:
                    self.partial_x_H_transpose_inv_V_expr = einsum_sympy(
                        "ji,jk->ik", self.partial_x_H_sym, self.inv_V_expr
                    )
                    self.partial_x_H_transpose_inv_V_func = lambdify(
                        (*common_args, self.theta_1, self.theta_3),
                        self.partial_x_H_transpose_inv_V_expr,
                        modules="jax",
                    )
                except Exception as e:
                    print(f"Error creating partial_x_H_transpose_inv_V_expr or func: {e}")
                    self.partial_x_H_transpose_inv_V_func = None
            else:
                self.partial_x_H_transpose_inv_V_func = None

            self.inv_S0_xx_expr = None
            self.inv_S0_xy_expr = None
            self.inv_S0_yx_expr = None
            self.inv_S0_yy_expr = None
            self.log_det_S0_expr = None

            if (
                self.inv_C_expr is not None
                and self.inv_V_expr is not None
                and self.partial_x_H_transpose_inv_V_expr is not None
            ):
                try:
                    term_for_invS0xx = einsum_sympy(
                        "ik,kj->ij", self.partial_x_H_transpose_inv_V_expr, self.partial_x_H_sym
                    )
                    self.inv_S0_xx_expr = self.inv_C_expr + 3 * term_for_invS0xx
                    self.inv_S0_xx_func = lambdify(
                        (*common_args, self.theta_1, self.theta_3),
                        self.inv_S0_xx_expr,
                        modules="jax",
                    )
                except Exception as e:
                    print(f"Error creating inv_S0_xx_expr or func: {e}")
                    self.inv_S0_xx_func = None

                try:
                    self.inv_S0_xy_expr = -6 * self.partial_x_H_transpose_inv_V_expr
                    self.inv_S0_xy_func = lambdify(
                        (*common_args, self.theta_1, self.theta_3),
                        self.inv_S0_xy_expr,
                        modules="jax",
                    )
                except Exception as e:
                    print(f"Error creating inv_S0_xy_expr or func: {e}")
                    self.inv_S0_xy_func = None

                try:
                    self.inv_S0_yx_expr = -6 * einsum_sympy(
                        "ik,kj->ij", self.inv_V_expr, self.partial_x_H_sym
                    )
                    self.inv_S0_yx_func = lambdify(
                        (*common_args, self.theta_1, self.theta_3),
                        self.inv_S0_yx_expr,
                        modules="jax",
                    )
                except Exception as e:
                    print(f"Error creating inv_S0_yx_expr or func: {e}")
                    self.inv_S0_yx_func = None

                try:
                    self.inv_S0_yy_expr = 12 * self.inv_V_expr
                    self.inv_S0_yy_func = lambdify(
                        (*common_args, self.theta_1, self.theta_3),
                        self.inv_S0_yy_expr,
                        modules="jax",
                    )
                except Exception as e:
                    print(f"Error creating inv_S0_yy_expr or func: {e}")
                    self.inv_S0_yy_func = None
            else:
                print(
                    "Warning: Could not create inv_S0_xx, inv_S0_xy, inv_S0_yx, inv_S0_yy expressions/functions due to previous errors with C_inv, V_inv or dHdx_T_V_inv."
                )
                self.inv_S0_xx_func = None
                self.inv_S0_xy_func = None
                self.inv_S0_yx_func = None
                self.inv_S0_yy_func = None

            if (
                self.log_det_C_expr is not None
                and self.log_det_V_expr is not None
                and self.log_det_C_expr != -oo
                and self.log_det_V_expr != -oo
            ):  # type: ignore
                try:
                    # d_y = self.y.shape[0] # Not used in the simplified log_det_S0 expr in the original code
                    self.log_det_S0_expr = self.log_det_C_expr + self.log_det_V_expr
                    self.log_det_S0_func = lambdify(
                        (*common_args, self.theta_1, self.theta_3),
                        self.log_det_S0_expr,
                        modules="jax",
                    )
                except Exception as e:
                    print(f"Error creating log_det_S0_expr or func: {e}")
                    self.log_det_S0_func = None
            else:
                print(
                    "Warning: Could not create log_det_S0_expr or func due to issues with log_det_C or log_det_V."
                )
                if self.log_det_C_expr == -oo or self.log_det_V_expr == -oo:  # type: ignore
                    self.log_det_S0_expr = -oo  # type: ignore
                self.log_det_S0_func = None

        except Exception as e:
            print(
                f"CRITICAL Error during symbolic calc/lambdify in LikelihoodEvaluator __init__: {e}"
            )
            self.C_func = None
            self.inv_C_func = None
            self.log_det_C_func = None
            self.V_func = None
            self.inv_V_func = None
            self.log_det_V_func = None
            self.partial_x_H_transpose_inv_V_func = None
            self.inv_S0_xx_func = None
            self.inv_S0_xy_func = None
            self.inv_S0_yx_func = None
            self.inv_S0_yy_func = None
            self.log_det_S0_func = None
            raise

    # --- キャッシュキーヘルパー (変更なし) ---
    def _get_tensor_srepr(self, tensor: Basic) -> str:
        try:
            if isinstance(tensor, Array):
                return srepr(tensor.tolist())
            return srepr(tensor)
        except Exception as e:
            print(f"Warning: srepr failed for {type(tensor)}. Using fallback. Error: {e}")
            return INVALID_SREPR_KEY

    # --- Infinitesimal Generator L (変更なし) ---
    def _L_elem_once(self, f_elem: Basic) -> Basic:
        if not isinstance(f_elem, Expr) or f_elem.is_constant(simplify=False):
            return S(0)  # sympy.S
        try:
            f_elem_sym = sp.sympify(f_elem)
            df_dx = derive_by_array(f_elem_sym, self.x)
            df_dy = derive_by_array(f_elem_sym, self.y)
            d2f_dx2 = derive_by_array(df_dx, self.x)

            C_term_sym = self.C_sym

            term1 = einsum_sympy("i,i->", self.A, df_dx)
            term2 = einsum_sympy("i,i->", self.H, df_dy)
            term3 = (S(1) / 2) * einsum_sympy("ij,ij->", C_term_sym, d2f_dx2)  # sympy.S
            return term1 + term2 + term3
        except Exception:
            print(
                f"Error applying _L_elem_once to: {type(f_elem)} with srepr: {self._get_tensor_srepr(f_elem)}"
            )
            raise

    def L(self, f_tensor: Basic, k: int) -> Basic:
        cache_key = (k, self._get_tensor_srepr(f_tensor))
        if cache_key in self._L_cache:
            return self._L_cache[cache_key]

        if k == 0:
            result = f_tensor
        elif k < 0:
            msg = "Order k must be non-negative"
            raise ValueError(msg)
        else:
            f_prev = self.L(f_tensor, k - 1)
            try:
                if isinstance(f_prev, Array) and hasattr(f_prev, "applyfunc"):
                    result = f_prev.applyfunc(self._L_elem_once)
                elif isinstance(f_prev, (Expr, Basic)):
                    result = self._L_elem_once(f_prev)
                else:
                    print(f"Warning: Applying L to non-symbolic {type(f_prev)} in L(k={k}).")
                    result = (
                        sp_zeros(*f_prev.shape) if hasattr(f_prev, "shape") else S(0)
                    )  # sympy.zeros, sympy.S
            except Exception:
                print(f"Error in L applying L once for k={k}")
                print(f"L^{k - 1}(f): {type(f_prev)}, srepr: {self._get_tensor_srepr(f_prev)}")
                raise
        self._L_cache[cache_key] = result
        return result

    def L_0(self, f_tensor: Basic, k: int) -> Basic:
        if k < 0:
            msg = "k must be non-negative"
            raise ValueError(msg)
        Lk_f = self.L(f_tensor, k)
        fact_k = factorial(k)  # sympy.factorial
        if fact_k == 0:
            msg = f"Factorial({k}) is zero? This should not happen."
            raise ValueError(msg)

        try:
            if isinstance(Lk_f, Array) and hasattr(Lk_f, "applyfunc"):
                result = Lk_f.applyfunc(lambda elem: elem / S(fact_k))  # sympy.S
            elif isinstance(Lk_f, (Expr, Basic)):
                result = Lk_f / S(fact_k)  # sympy.S
            else:
                result = Lk_f / float(fact_k)  # Fallback for non-sympy types
        except Exception:
            print(f"Error in L_0 dividing by factorial({k})")
            print(f"Lk_f type: {type(Lk_f)}, srepr: {self._get_tensor_srepr(Lk_f)}")
            raise
        return result

    def L_0_func(self, f_tensor: Basic, k: int) -> Callable:
        cache_key = (k, self._get_tensor_srepr(f_tensor))
        if cache_key in self._L0_func_cache:
            return self._L0_func_cache[cache_key]
        try:
            L0_expr_val = self.L_0(f_tensor, k)
            lambdify_args = (self.x, self.y, self.theta_1, self.theta_2, self.theta_3)

            func = lambdify(lambdify_args, L0_expr_val, modules="jax")  # Changed to JAX
            self._L0_func_cache[cache_key] = func
            return func
        except Exception:
            print(f"Error creating L_0_func for k={k}, srepr={cache_key[1]}")
            print(f"Expression was: {L0_expr_val if 'L0_expr_val' in locals() else 'undefined'}")  # type: ignore
            raise

    # --- Auxiliary functions for Quasi-Likelihood (JAXへ変更) ---
    def Dx_func(
        self,
        L0_x_funcs: tuple[Callable, ...],
        x_j: jnp.ndarray,
        x_j_1: jnp.ndarray,
        y_j_1: jnp.ndarray,
        theta_2_val: jnp.ndarray,
        theta_1_bar: jnp.ndarray,
        theta_2_bar: jnp.ndarray,
        theta_3_bar: jnp.ndarray,
        h: float,
        k_arg: int,
    ) -> jnp.ndarray:
        if h <= 0:
            return jnp.zeros_like(x_j, dtype=x_j.dtype)
        DX_SCALE = jnp.power(h, -0.5)

        D_x = DX_SCALE * (x_j - x_j_1)

        if k_arg >= 1:
            try:
                A_val = self.A_func(x_j_1, y_j_1, theta_2_val)
            except Exception as e:
                msg = f"Dx_func: A_func eval error: {e}"
                raise RuntimeError(msg) from e
            D_x -= DX_SCALE * jnp.power(h, 1.0) * A_val

        if k_arg >= 2:
            args_for_L0_bar = (x_j_1, y_j_1, theta_1_bar, theta_2_bar, theta_3_bar)
            for m_loop in range(2, k_arg + 1):
                h_pow_m = jnp.power(h, float(m_loop))
                try:
                    L0_x_m_val = L0_x_funcs[m_loop](*args_for_L0_bar)
                except IndexError as e:
                    msg = (
                        f"Dx_func: L0_x_funcs index m={m_loop} is out of range (len={len(L0_x_funcs)})."
                        f" k_arg was {k_arg}."
                    )
                    raise RuntimeError(msg) from e
                except Exception as e:
                    msg = f"Dx_func: L0_x_funcs[{m_loop}] eval error: {e}"
                    raise RuntimeError(msg) from e
                D_x -= DX_SCALE * h_pow_m * L0_x_m_val
        return D_x

    def Dy_func(
        self,
        L0_y_funcs: tuple[Callable, ...],
        y_j: jnp.ndarray,
        y_j_1: jnp.ndarray,
        x_j_1: jnp.ndarray,
        theta_3_val: jnp.ndarray,
        theta_1_bar: jnp.ndarray,
        theta_2_bar: jnp.ndarray,
        theta_3_bar: jnp.ndarray,
        h: float,
        k_arg: int,
    ) -> jnp.ndarray:
        if h <= 0:
            return jnp.zeros_like(y_j, dtype=y_j.dtype)
        DY_SCALE = jnp.power(h, -1.5)

        D_y = DY_SCALE * (y_j - y_j_1)

        if k_arg >= 1:
            try:
                H_val = self.H_func(x_j_1, y_j_1, theta_3_val)
            except Exception as e:
                msg = f"Dy_func: H_func eval error: {e}"
                raise RuntimeError(msg) from e
            D_y -= DY_SCALE * jnp.power(h, 1.0) * H_val

        if k_arg >= 2:
            args_for_L0_bar = (x_j_1, y_j_1, theta_1_bar, theta_2_bar, theta_3_bar)
            for m_loop in range(2, k_arg + 1):
                h_pow_m = jnp.power(h, float(m_loop))
                try:
                    L0_y_m_val = L0_y_funcs[m_loop](*args_for_L0_bar)
                except IndexError as e:
                    msg = (
                        f"Dy_func: L0_y_funcs index m={m_loop} is out of range (len={len(L0_y_funcs)})."
                        f" k_arg was {k_arg}."
                    )
                    raise RuntimeError(msg) from e
                except Exception as e:
                    msg = f"Dy_func: L0_y_funcs[{m_loop}] eval error: {e}"
                    raise RuntimeError(msg) from e
                D_y -= DY_SCALE * h_pow_m * L0_y_m_val
        return D_y

    # --- S 項の計算 (SymPy部分は変更なし) ---
    def S(self, k: int) -> tuple[Array, Array, Array, Array]:
        x_sym = self.x
        y_sym = self.y

        T_xx = self.L_0(tensorproduct(x_sym, x_sym), k + 1)
        T_xy = self.L_0(tensorproduct(x_sym, y_sym), k + 2)
        T_yx = self.L_0(tensorproduct(y_sym, x_sym), k + 2)
        T_yy = self.L_0(tensorproduct(y_sym, y_sym), k + 3)

        def compute_U_component(f1: Array, f2: Array, total_sum_order: int) -> Array:
            if not (isinstance(f1, Array) and f1.rank() == 1):
                msg = f"U computation requires rank-1 Array for f1, got {type(f1)}"
                raise ValueError(msg)
            if not (isinstance(f2, Array) and f2.rank() == 1):
                msg = f"U computation requires rank-1 Array for f2, got {type(f2)}"
                raise ValueError(msg)

            u_shape = (f1.shape[0], f2.shape[0])
            U_component = Array(sp_zeros(*u_shape))  # sympy.zeros

            for m1 in range(total_sum_order + 1):
                m2 = total_sum_order - m1
                try:
                    L0_f1_m1 = self.L_0(f1, m1)
                    L0_f2_m2 = self.L_0(f2, m2)

                    term = tensorproduct(L0_f1_m1, L0_f2_m2)

                    # Original code used np.prod. Since this is symbolic, explicit product.
                    term_elements = 1
                    for s_val in term.shape:
                        term_elements *= int(s_val)  # Ensure python int for product
                    u_elements = 1
                    for s_val in u_shape:
                        u_elements *= int(s_val)  # Ensure python int for product

                    if term.shape != u_shape:
                        if term_elements == u_elements:
                            print(
                                f"Warning: compute_U_component term shape mismatch ({term.shape} vs {u_shape}). Reshaping."
                            )
                            term = Array(term).reshape(*u_shape)
                        else:
                            msg = f"compute_U_component term shape mismatch ({term.shape} vs {u_shape}) and cannot reshape."
                            raise ValueError(msg)

                    U_component = U_component + term
                except Exception as e:
                    msg = (
                        f"Error in compute_U_component for f1={f1}, f2={f2}, m1={m1}, m2={m2}: {e}"
                    )
                    raise RuntimeError(msg) from e
            return Array(U_component)

        U_xx = compute_U_component(x_sym, x_sym, k + 1)
        U_xy = compute_U_component(x_sym, y_sym, k + 2)
        U_yx = compute_U_component(y_sym, x_sym, k + 2)
        U_yy = compute_U_component(y_sym, y_sym, k + 3)

        try:
            S_xx = Array(T_xx) - Array(U_xx)
            S_xy = Array(T_xy) - Array(U_xy)
            S_yx = Array(T_yx) - Array(U_yx)
            S_yy = Array(T_yy) - Array(U_yy)
        except Exception as e:
            msg = f"Error subtracting U from T for S(k={k}): {e}. Types: T_xx={type(T_xx)}, U_xx={type(U_xx)}"
            raise RuntimeError(msg) from e

        return S_xx, S_xy, S_yx, S_yy

    def S_func(self, k: int) -> tuple[Callable, Callable, Callable, Callable]:
        cache_key = k
        if cache_key in self._S_func_cache:
            return self._S_func_cache[cache_key]
        try:
            S_k_tuple = self.S(k)  # Renamed S_k to S_k_tuple to avoid confusion with k
            S_xx_expr, S_xy_expr, S_yx_expr, S_yy_expr = S_k_tuple
            lambdify_args = (self.x, self.y, self.theta_1, self.theta_2, self.theta_3)
            f_xx = lambdify(lambdify_args, S_xx_expr, modules="jax")
            f_xy = lambdify(lambdify_args, S_xy_expr, modules="jax")
            f_yx = lambdify(lambdify_args, S_yx_expr, modules="jax")
            f_yy = lambdify(lambdify_args, S_yy_expr, modules="jax")

            funcs = (f_xx, f_xy, f_yx, f_yy)
            self._S_func_cache[cache_key] = funcs
            return funcs
        except Exception as e:
            print(f"Error creating S_func for k={k}: {e}")
            raise

    # --- Quasi-Likelihood Evaluator Factories (JAXへ変更) ---
    def make_quasi_likelihood_v1_prime_evaluator(
        self, x_series: jnp.ndarray, y_series: jnp.ndarray, h: float, k: int
    ) -> Callable:
        x_series_jnp = jnp.asarray(x_series)  # Ensure JAX array
        y_series_jnp = jnp.asarray(y_series)  # Ensure JAX array

        n = x_series_jnp.shape[0]
        d_x = self.x.shape[0]  # Sympy shape, used for jnp.zeros dimension
        num_transitions = n - 1
        if num_transitions < 1 or y_series_jnp.shape[0] != n:
            msg = "Time series length must be > 1 and shapes must match for V1'."
            raise ValueError(msg)

        try:
            L0_x_funcs = tuple(self.L_0_func(self.x, m) for m in range(k))
            S_l_funcs = tuple(self.S_func(l_s) for l_s in range(1, k))
        except Exception as e:
            msg = f"V1' precalculation error: {e}"
            raise RuntimeError(msg) from e

        if self.inv_C_func is None or self.log_det_C_func is None:
            msg = "inv_C_func or log_det_C_func is missing for V1'. Check __init__ warnings."
            raise RuntimeError(msg)

        def evaluate_v1_prime(
            theta_1_val: jnp.ndarray,
            theta_1_bar: jnp.ndarray,
            theta_2_bar: jnp.ndarray,
            theta_3_bar: jnp.ndarray,
        ) -> float:
            theta_1_val_j = jnp.asarray(theta_1_val)
            theta_1_bar_j = jnp.asarray(theta_1_bar)
            theta_2_bar_j = jnp.asarray(theta_2_bar)
            theta_3_bar_j = jnp.asarray(theta_3_bar)

            result_dtype = jnp.result_type(
                theta_1_val_j.dtype, theta_1_bar_j.dtype, theta_2_bar_j.dtype, theta_3_bar_j.dtype
            )

            def scan_body(total, step_inputs):
                x_j, x_j_1, y_j_1 = step_inputs

                invC_val = self.inv_C_func(x_j_1, y_j_1, theta_1_val_j)
                logDetC_val = self.log_det_C_func(x_j_1, y_j_1, theta_1_val_j)

                Dx_val = self.Dx_func(
                    L0_x_funcs,
                    x_j,
                    x_j_1,
                    y_j_1,
                    theta_2_bar_j,
                    theta_1_bar_j,
                    theta_2_bar_j,
                    theta_3_bar_j,
                    h,
                    k - 1,
                )

                sum_Sxx_val = jnp.zeros((d_x, d_x), dtype=invC_val.dtype)
                for idx, S_funcs in enumerate(S_l_funcs, start=1):
                    Sxx_func = S_funcs[0]
                    sum_Sxx_val += (h**idx) * Sxx_func(
                        x_j_1, y_j_1, theta_1_bar_j, theta_2_bar_j, theta_3_bar_j
                    )

                term1_quad = -jnp.einsum("ij,i,j->", invC_val, Dx_val, Dx_val)
                term2_trace = jnp.einsum("ij,ji->", invC_val, sum_Sxx_val)
                term3_logdet = -logDetC_val

                step_likelihood = term1_quad + term2_trace + term3_logdet
                step_likelihood = jnp.where(
                    jnp.isfinite(step_likelihood),
                    step_likelihood,
                    jnp.array(jnp.nan, dtype=step_likelihood.dtype),
                )

                return total + step_likelihood, None

            initial_total = jnp.zeros((), dtype=result_dtype)
            scan_inputs = (x_series_jnp[1:], x_series_jnp[:-1], y_series_jnp[:-1])
            total_log_likelihood, _ = lax.scan(scan_body, initial_total, scan_inputs)

            if num_transitions > 0:
                return total_log_likelihood / (2.0 * num_transitions)
            return jnp.full_like(total_log_likelihood, jnp.nan)

        return evaluate_v1_prime

    def make_quasi_likelihood_v1_evaluator(
        self, x_series: jnp.ndarray, y_series: jnp.ndarray, h: float, k: int
    ) -> Callable:
        x_series_jnp = jnp.asarray(x_series)
        y_series_jnp = jnp.asarray(y_series)

        n = x_series_jnp.shape[0]
        d_x = self.x.shape[0]  # Sympy shape
        d_y = self.y.shape[0]  # Sympy shape
        num_transitions = n - 1
        if num_transitions < 1 or y_series_jnp.shape[0] != n:
            msg = "Time series length must be > 1 and shapes must match for V1."
            raise ValueError(msg)

        try:
            L0_x_funcs = tuple(self.L_0_func(self.x, m) for m in range(k))
            L0_y_funcs = tuple(self.L_0_func(self.y, m) for m in range(k + 1))
            S_l_funcs = tuple(self.S_func(l_s) for l_s in range(1, k))
        except Exception as e:
            msg = f"V1 precalculation error: {e}"
            raise RuntimeError(msg) from e

        s0_funcs_needed = [
            self.inv_S0_xx_func,
            self.inv_S0_xy_func,
            self.inv_S0_yx_func,
            self.inv_S0_yy_func,
            self.log_det_S0_func,
        ]
        if any(f is None for f in s0_funcs_needed):
            missing_s0_func_names = [
                name
                for name, func in zip(
                    [
                        "inv_S0_xx_func",
                        "inv_S0_xy_func",
                        "inv_S0_yx_func",
                        "inv_S0_yy_func",
                        "log_det_S0_func",
                    ],
                    s0_funcs_needed,
                    strict=False,
                )
                if func is None
            ]
            msg = f"S0 related functions {missing_s0_func_names} are missing for V1. Check __init__ warnings."
            raise RuntimeError(msg)

        def evaluate_v1(
            theta_1_val: jnp.ndarray,
            theta_1_bar: jnp.ndarray,
            theta_2_bar: jnp.ndarray,
            theta_3_bar: jnp.ndarray,
        ) -> float:
            theta_1_val_j = jnp.asarray(theta_1_val)
            theta_1_bar_j = jnp.asarray(theta_1_bar)
            theta_2_bar_j = jnp.asarray(theta_2_bar)
            theta_3_bar_j = jnp.asarray(theta_3_bar)

            result_dtype = jnp.result_type(
                theta_1_val_j.dtype,
                theta_1_bar_j.dtype,
                theta_2_bar_j.dtype,
                theta_3_bar_j.dtype,
            )

            def scan_body(total, step_inputs):
                x_j, y_j, x_j_1, y_j_1 = step_inputs

                inv_S0_xx_val = self.inv_S0_xx_func(x_j_1, y_j_1, theta_1_val_j, theta_3_bar_j)
                inv_S0_xy_val = self.inv_S0_xy_func(x_j_1, y_j_1, theta_1_val_j, theta_3_bar_j)
                inv_S0_yx_val = self.inv_S0_yx_func(x_j_1, y_j_1, theta_1_val_j, theta_3_bar_j)
                inv_S0_yy_val = self.inv_S0_yy_func(x_j_1, y_j_1, theta_1_val_j, theta_3_bar_j)
                log_det_S0_val = self.log_det_S0_func(x_j_1, y_j_1, theta_1_val_j, theta_3_bar_j)

                Dx_val = self.Dx_func(
                    L0_x_funcs,
                    x_j,
                    x_j_1,
                    y_j_1,
                    theta_2_bar_j,
                    theta_1_bar_j,
                    theta_2_bar_j,
                    theta_3_bar_j,
                    h,
                    k - 1,
                )
                Dy_val = self.Dy_func(
                    L0_y_funcs,
                    y_j,
                    y_j_1,
                    x_j_1,
                    theta_3_bar_j,
                    theta_1_bar_j,
                    theta_2_bar_j,
                    theta_3_bar_j,
                    h,
                    k - 1,
                )

                sum_S_xx = jnp.zeros((d_x, d_x), dtype=inv_S0_xx_val.dtype)
                sum_S_xy = jnp.zeros((d_x, d_y), dtype=inv_S0_xx_val.dtype)
                sum_S_yx = jnp.zeros((d_y, d_x), dtype=inv_S0_xx_val.dtype)
                sum_S_yy = jnp.zeros((d_y, d_y), dtype=inv_S0_xx_val.dtype)

                for idx, s_funcs in enumerate(S_l_funcs, start=1):
                    s_xx = s_funcs[0](x_j_1, y_j_1, theta_1_bar_j, theta_2_bar_j, theta_3_bar_j)
                    s_xy = s_funcs[1](x_j_1, y_j_1, theta_1_bar_j, theta_2_bar_j, theta_3_bar_j)
                    s_yx = s_funcs[2](x_j_1, y_j_1, theta_1_bar_j, theta_2_bar_j, theta_3_bar_j)
                    s_yy = s_funcs[3](x_j_1, y_j_1, theta_1_bar_j, theta_2_bar_j, theta_3_bar_j)

                    h_power = h**idx
                    sum_S_xx += h_power * s_xx
                    sum_S_xy += h_power * s_xy
                    sum_S_yx += h_power * s_yx
                    sum_S_yy += h_power * s_yy

                q_xx = jnp.einsum("ij,i,j->", inv_S0_xx_val, Dx_val, Dx_val)
                q_xy = jnp.einsum("ij,i,j->", inv_S0_xy_val, Dx_val, Dy_val)
                q_yx = jnp.einsum("ij,i,j->", inv_S0_yx_val, Dy_val, Dx_val)
                q_yy = jnp.einsum("ij,i,j->", inv_S0_yy_val, Dy_val, Dy_val)
                quadratic_term = -(q_xx + q_xy + q_yx + q_yy)

                tr_xx = jnp.einsum("ij,ji->", inv_S0_xx_val, sum_S_xx)
                tr_xy = jnp.einsum("ij,ji->", inv_S0_xy_val, sum_S_yx)
                tr_yx = jnp.einsum("ij,ji->", inv_S0_yx_val, sum_S_xy)
                tr_yy = jnp.einsum("ij,ji->", inv_S0_yy_val, sum_S_yy)
                trace_term = tr_xx + tr_xy + tr_yx + tr_yy

                logdet_term = -log_det_S0_val

                step_likelihood = quadratic_term + trace_term + logdet_term
                step_likelihood = jnp.where(
                    jnp.isfinite(step_likelihood),
                    step_likelihood,
                    jnp.array(jnp.nan, dtype=step_likelihood.dtype),
                )

                return total + step_likelihood, None

            initial_total = jnp.zeros((), dtype=result_dtype)
            scan_inputs = (
                x_series_jnp[1:],
                y_series_jnp[1:],
                x_series_jnp[:-1],
                y_series_jnp[:-1],
            )
            total_log_likelihood, _ = lax.scan(scan_body, initial_total, scan_inputs)

            if num_transitions > 0:
                return total_log_likelihood / (2.0 * num_transitions)
            return jnp.full_like(total_log_likelihood, jnp.nan)

        return evaluate_v1

    def make_quasi_likelihood_v2_evaluator(
        self, x_series: jnp.ndarray, y_series: jnp.ndarray, h: float, k: int
    ) -> Callable:
        x_series_jnp = jnp.asarray(x_series)
        y_series_jnp = jnp.asarray(y_series)  # y_series is used for y_j_1 in Dx_func
        n = x_series_jnp.shape[0]
        num_transitions = n - 1
        if num_transitions < 1 or y_series_jnp.shape[0] != n:  # Check y_series_jnp shape
            msg = "Time series length must be > 1 and shapes must match for V2."
            raise ValueError(msg)

        try:
            # Dx_func with k_arg = k uses L0_x_funcs up to index k. So range(k + 1) is {0, ..., k}.
            L0_x_funcs = tuple(self.L_0_func(self.x, m) for m in range(k + 1))
        except Exception as e:
            msg = f"V2 precalculation error (L0_x): {e}"
            raise RuntimeError(msg) from e

        if self.inv_C_func is None:
            msg = "inv_C_func is missing for V2. Check __init__ warnings."
            raise RuntimeError(msg)

        def evaluate_v2(
            theta_2_val: jnp.ndarray,
            theta_1_bar: jnp.ndarray,
            theta_2_bar: jnp.ndarray,
            theta_3_bar: jnp.ndarray,
        ) -> float:
            theta_2_val_j = jnp.asarray(theta_2_val)
            theta_1_bar_j = jnp.asarray(theta_1_bar)
            theta_2_bar_j = jnp.asarray(theta_2_bar)
            theta_3_bar_j = jnp.asarray(theta_3_bar)

            result_dtype = jnp.result_type(
                theta_2_val_j.dtype,
                theta_1_bar_j.dtype,
                theta_2_bar_j.dtype,
                theta_3_bar_j.dtype,
            )

            def scan_body(total, step_inputs):
                x_j, x_j_1, y_j_1 = step_inputs

                invC_val = self.inv_C_func(x_j_1, y_j_1, theta_1_bar_j)
                Dx_val = self.Dx_func(
                    L0_x_funcs,
                    x_j,
                    x_j_1,
                    y_j_1,
                    theta_2_val_j,
                    theta_1_bar_j,
                    theta_2_bar_j,
                    theta_3_bar_j,
                    h,
                    k,
                )

                term1_quad = -jnp.einsum("ij,i,j->", invC_val, Dx_val, Dx_val)
                step_likelihood = jnp.where(
                    jnp.isfinite(term1_quad),
                    term1_quad,
                    jnp.array(jnp.nan, dtype=term1_quad.dtype),
                )

                return total + step_likelihood, None

            initial_total = jnp.zeros((), dtype=result_dtype)
            scan_inputs = (x_series_jnp[1:], x_series_jnp[:-1], y_series_jnp[:-1])
            total_log_likelihood, _ = lax.scan(scan_body, initial_total, scan_inputs)

            if num_transitions > 0 and h > 0:
                return total_log_likelihood / (2.0 * h * num_transitions)
            return jnp.full_like(total_log_likelihood, jnp.nan)

        return evaluate_v2

    def make_quasi_likelihood_v3_evaluator(
        self, x_series: jnp.ndarray, y_series: jnp.ndarray, h: float, k: int
    ) -> Callable:
        x_series_jnp = jnp.asarray(x_series)
        y_series_jnp = jnp.asarray(y_series)
        n = x_series_jnp.shape[0]
        num_transitions = n - 1
        if num_transitions < 1 or y_series_jnp.shape[0] != n:
            msg = "Time series length must be > 1 and shapes must match for V3."
            raise ValueError(msg)

        try:
            # Dx_func(k_arg=k) uses L0_x_funcs up to k. So range(k + 1) i.e. {0,...,k}
            L0_x_funcs = tuple(self.L_0_func(self.x, m) for m in range(k + 1))
            # Dy_func(k_arg=k+1) uses L0_y_funcs up to k+1. So range(k + 2) i.e. {0,...,k+1}
            L0_y_funcs = tuple(self.L_0_func(self.y, m) for m in range(k + 2))
        except Exception as e:
            msg = f"V3 precalculation error (L0_x/L0_y): {e}"
            raise RuntimeError(msg) from e

        if self.inv_V_func is None or self.partial_x_H_transpose_inv_V_func is None:
            print(
                "Warning: V related functions (inv_V_func or partial_x_H_transpose_inv_V_func) are missing."
            )

            def evaluator_unavailable_v3(
                *args: Any, **kwargs: Any
            ) -> float:  # Added type hints for args/kwargs
                msg = "V3 likelihood evaluator is unavailable due to missing V-related functions (likely V is singular)."
                raise RuntimeError(msg)

            return evaluator_unavailable_v3

        def evaluate_v3(
            theta_3_val: jnp.ndarray,
            theta_1_bar: jnp.ndarray,
            theta_2_bar: jnp.ndarray,
            theta_3_bar: jnp.ndarray,
        ) -> float:
            theta_3_val_j = jnp.asarray(theta_3_val)
            theta_1_bar_j = jnp.asarray(theta_1_bar)
            theta_2_bar_j = jnp.asarray(theta_2_bar)
            theta_3_bar_j = jnp.asarray(theta_3_bar)

            result_dtype = jnp.result_type(
                theta_3_val_j.dtype,
                theta_1_bar_j.dtype,
                theta_2_bar_j.dtype,
                theta_3_bar_j.dtype,
            )

            def scan_body(total, step_inputs):
                x_j, y_j, x_j_1, y_j_1 = step_inputs

                invV_val = self.inv_V_func(x_j_1, y_j_1, theta_1_bar_j, theta_3_bar_j)
                pHTinvV_val = self.partial_x_H_transpose_inv_V_func(
                    x_j_1, y_j_1, theta_1_bar_j, theta_3_bar_j
                )

                Dx_val = self.Dx_func(
                    L0_x_funcs,
                    x_j,
                    x_j_1,
                    y_j_1,
                    theta_2_bar_j,
                    theta_1_bar_j,
                    theta_2_bar_j,
                    theta_3_bar_j,
                    h,
                    k,
                )
                Dy_val = self.Dy_func(
                    L0_y_funcs,
                    y_j,
                    y_j_1,
                    x_j_1,
                    theta_3_val_j,
                    theta_1_bar_j,
                    theta_2_bar_j,
                    theta_3_bar_j,
                    h,
                    k + 1,
                )

                term1_V3 = -jnp.einsum("ij,i,j->", invV_val, Dy_val, Dy_val)
                term2_V3 = jnp.einsum("i,ik,k->", Dx_val, pHTinvV_val, Dy_val)

                step_likelihood = term1_V3 + term2_V3
                step_likelihood = jnp.where(
                    jnp.isfinite(step_likelihood),
                    step_likelihood,
                    jnp.array(jnp.nan, dtype=step_likelihood.dtype),
                )

                return total + step_likelihood, None

            initial_total = jnp.zeros((), dtype=result_dtype)
            scan_inputs = (
                x_series_jnp[1:],
                y_series_jnp[1:],
                x_series_jnp[:-1],
                y_series_jnp[:-1],
            )
            total_log_likelihood, _ = lax.scan(scan_body, initial_total, scan_inputs)

            if num_transitions > 0 and h != 0:
                return (6.0 * h * total_log_likelihood) / num_transitions
            return jnp.full_like(total_log_likelihood, jnp.nan)

        return evaluate_v3

    # EVALUATOR_FACTORIES_END


# %%
# The DegenerateDiffusionProcess class and the __main__ block are assumed to be
# already JAX-compatible as per the user's setup (DegenerateDiffusionProcess_JAX.py)
# and the provided snippet for DegenerateDiffusionProcess.
# No changes are made to those parts here, as the request was specific to
# optimizing LikelihoodEvaluator and its numpy usages.

# Example of how DegenerateDiffusionProcess might look (from user's snippet for context)
# This is NOT part of the direct answer code block for LikelihoodEvaluator,
# but shows the expected JAX usage in the model.
"""
from dataclasses import dataclass
from functools import partial

import jax
from jax import lax
from jax import numpy as jnp_model_example  # Using a different alias for clarity
import sympy as sp_model_example
from sympy import lambdify as lambdify_model_example, symbols as symbols_model_example

@dataclass(frozen=True)
class DegenerateDiffusionProcess_Example: # Renamed for clarity
    x: sp_model_example.Array
    y: sp_model_example.Array
    theta_1: sp_model_example.Array
    theta_2: sp_model_example.Array
    theta_3: sp_model_example.Array
    A: sp_model_example.Array
    B: sp_model_example.Array
    H: sp_model_example.Array

    def __post_init__(self):
        common_args = (self.x, self.y)
        try:
            object.__setattr__(self, "A_func",
                               lambdify_model_example((*common_args, self.theta_2), self.A, modules="jax"))
            object.__setattr__(self, "B_func",
                               lambdify_model_example((*common_args, self.theta_1), self.B, modules="jax"))
            object.__setattr__(self, "H_func",
                               lambdify_model_example((*common_args, self.theta_3), self.H, modules="jax"))
        except Exception as e:
            print(f"Error during lambdification in __post_init__: {e}")
            raise
    # ... rest of the DegenerateDiffusionProcess methods ...
"""

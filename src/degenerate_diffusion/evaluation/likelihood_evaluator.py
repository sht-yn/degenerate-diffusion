from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import jax
import jax.numpy as jnp
import sympy as sp
from jax import Array as JaxArray, lax
from sympy import (
    Array,
    Basic,
    Expr,
    ImmutableDenseNDimArray,
    Matrix,
    S,
    derive_by_array,
    lambdify,
    log,
    tensorproduct,
    zeros as sp_zeros,
)
from sympy.matrices.common import NonInvertibleMatrixError

from degenerate_diffusion.utils.einsum_sympy import einsum_sympy
from degenerate_diffusion.utils.symbolic_artifact import SymbolicArtifact

if TYPE_CHECKING:
    from degenerate_diffusion.processes.degenerate_diffusion_process import (
        DegenerateDiffusionProcess,
    )


SympyTensor: TypeAlias = Array | Basic
SympyTensorKey: TypeAlias = ImmutableDenseNDimArray | Basic
JittedLikelihood: TypeAlias = Callable[[JaxArray, JaxArray, JaxArray, JaxArray], JaxArray]


def simplify_tensor(tensor: SympyTensor) -> SympyTensor:
    """Return ``tensor`` with ``sympy.simplify`` applied elementwise when applicable."""
    if isinstance(tensor, Array):
        return Array(tensor.applyfunc(sp.simplify))
    if isinstance(tensor, Basic):
        return sp.simplify(tensor)
    return tensor


class SymbolicPreparationError(RuntimeError):
    """Raise when symbolic preprocessing fails fatally.

    記号計算による前処理が致命的に失敗した際に送出される例外。
    """


@dataclass(frozen=True)
class SymbolicPrecomputation:
    """Collect symbolic preprocessing artifacts and compiled callables.

    記号的な前処理結果と対応する関数群をまとめたデータコンテナ。
    """

    A: SymbolicArtifact
    B: SymbolicArtifact
    H: SymbolicArtifact
    C: SymbolicArtifact
    inv_C: SymbolicArtifact
    log_det_C: SymbolicArtifact
    partial_x_H_sym: Array
    V: SymbolicArtifact
    inv_V: SymbolicArtifact
    log_det_V: SymbolicArtifact
    partial_x_H_transpose_inv_V: SymbolicArtifact
    inv_S0_xx: SymbolicArtifact
    inv_S0_xy: SymbolicArtifact
    inv_S0_yx: SymbolicArtifact
    inv_S0_yy: SymbolicArtifact
    log_det_S0: SymbolicArtifact


class SymbolicLikelihoodPreparer:
    """Prepare symbolic ingredients for likelihood evaluation.

    疑似尤度の評価に必要な記号的な素材を生成するヘルパー。
    """

    def __init__(self, model: DegenerateDiffusionProcess) -> None:
        """Store the diffusion model that supplies symbolic expressions.

        記号処理の対象となる拡散過程 `model` を保持する。
        """
        self.model = model

    def prepare(self) -> SymbolicPrecomputation:
        """Return symbolic expressions and lambdified functions for the model.

        モデルに対して計算した記号式と lambdify 済み関数をまとめて返す。
        """
        x_sym = self.model.x
        y_sym = self.model.y
        theta_1 = self.model.theta_1
        theta_3 = self.model.theta_3

        common_args = (x_sym, y_sym)

        C_expr = einsum_sympy("ik,jk->ij", self.model.B.expr, self.model.B.expr)
        C_func = lambdify((*common_args, theta_1), C_expr, modules="jax")

        C_artifact = SymbolicArtifact(C_expr, C_func)

        C_matrix = Matrix(C_expr)
        inv_C = self._inverse_component("C", C_matrix, (*common_args, theta_1))
        log_det_C = self._log_det_component("C", C_matrix, (*common_args, theta_1))

        partial_x_H_sym = derive_by_array(self.model.H.expr, x_sym)

        V_expr = einsum_sympy("ki,kl,lj->ij", partial_x_H_sym, C_artifact.expr, partial_x_H_sym)
        V_func = lambdify((*common_args, theta_1, theta_3), V_expr, modules="jax")

        V_artifact = SymbolicArtifact(V_expr, V_func)

        V_matrix = Matrix(V_expr)
        inv_V = self._inverse_component("V", V_matrix, (*common_args, theta_1, theta_3))
        log_det_V = self._log_det_component("V", V_matrix, (*common_args, theta_1, theta_3))

        partial_x_H_transpose_inv_V = self._partial_x_H_transpose_inv_V(
            partial_x_H_sym, inv_V, (*common_args, theta_1, theta_3)
        )

        inv_S0_xx, inv_S0_xy, inv_S0_yx, inv_S0_yy = self._build_inv_S0(
            partial_x_H_sym,
            inv_C,
            inv_V,
            partial_x_H_transpose_inv_V,
            (*common_args, theta_1, theta_3),
        )

        log_det_S0 = self._build_log_det_S0(
            log_det_C,
            log_det_V,
            (*common_args, theta_1, theta_3),
        )

        return SymbolicPrecomputation(
            A=self.model.A,
            B=self.model.B,
            H=self.model.H,
            C=C_artifact,
            inv_C=inv_C,
            log_det_C=log_det_C,
            partial_x_H_sym=partial_x_H_sym,
            V=V_artifact,
            inv_V=inv_V,
            log_det_V=log_det_V,
            partial_x_H_transpose_inv_V=partial_x_H_transpose_inv_V,
            inv_S0_xx=inv_S0_xx,
            inv_S0_xy=inv_S0_xy,
            inv_S0_yx=inv_S0_yx,
            inv_S0_yy=inv_S0_yy,
            log_det_S0=log_det_S0,
        )

    @staticmethod
    def _inverse_component(
        name: str,
        matrix: Matrix,
        lambdify_args: tuple[Any, ...],
    ) -> SymbolicArtifact:
        try:
            inv_expr = Array(matrix.inv())
        except NonInvertibleMatrixError as exc:
            msg = f"Symbolic matrix {name} is not invertible."
            raise SymbolicPreparationError(msg) from exc
        inv_func = lambdify(lambdify_args, inv_expr, modules="jax")
        return SymbolicArtifact(inv_expr, inv_func)

    @staticmethod
    def _log_det_component(
        name: str,
        matrix: Matrix,
        lambdify_args: tuple[Any, ...],
    ) -> SymbolicArtifact:
        det_val = matrix.det()

        is_nonpositive = getattr(det_val, "is_nonpositive", None)
        if det_val == 0:
            msg = f"Symbolic det({name}) is zero."
            raise SymbolicPreparationError(msg)
        if is_nonpositive is True:
            msg = f"Symbolic det({name}) is non-positive."
            raise SymbolicPreparationError(msg)

        expr = log(det_val)
        func = lambdify(lambdify_args, expr, modules="jax")
        return SymbolicArtifact(expr, func)

    @staticmethod
    def _partial_x_H_transpose_inv_V(
        partial_x_H_sym: Array,
        inv_V: SymbolicArtifact,
        lambdify_args: tuple[Any, ...],
    ) -> SymbolicArtifact:
        expr = einsum_sympy("ij,jk->ik", partial_x_H_sym, inv_V.expr)
        func = lambdify(lambdify_args, expr, modules="jax")
        return SymbolicArtifact(expr, func)

    @staticmethod
    def _build_inv_S0(
        partial_x_H_sym: Array,
        inv_C: SymbolicArtifact,
        inv_V: SymbolicArtifact,
        partial_x_H_transpose_inv_V: SymbolicArtifact,
        lambdify_args: tuple[Any, ...],
    ) -> tuple[SymbolicArtifact, SymbolicArtifact, SymbolicArtifact, SymbolicArtifact]:
        term_for_invS0xx = einsum_sympy(
            "ik,jk->ij", partial_x_H_transpose_inv_V.expr, partial_x_H_sym
        )
        inv_S0_xx_expr = inv_C.expr + 3 * term_for_invS0xx
        inv_S0_xx_func = lambdify(lambdify_args, inv_S0_xx_expr, modules="jax")
        inv_S0_xx = SymbolicArtifact(inv_S0_xx_expr, inv_S0_xx_func)

        inv_S0_xy_expr = -6 * partial_x_H_transpose_inv_V.expr
        inv_S0_xy_func = lambdify(lambdify_args, inv_S0_xy_expr, modules="jax")
        inv_S0_xy = SymbolicArtifact(inv_S0_xy_expr, inv_S0_xy_func)

        inv_S0_yx_expr = -6 * einsum_sympy("ik,jk->ij", inv_V.expr, partial_x_H_sym)
        inv_S0_yx_func = lambdify(lambdify_args, inv_S0_yx_expr, modules="jax")
        inv_S0_yx = SymbolicArtifact(inv_S0_yx_expr, inv_S0_yx_func)

        inv_S0_yy_expr = 12 * inv_V.expr
        inv_S0_yy_func = lambdify(lambdify_args, inv_S0_yy_expr, modules="jax")
        inv_S0_yy = SymbolicArtifact(inv_S0_yy_expr, inv_S0_yy_func)

        return inv_S0_xx, inv_S0_xy, inv_S0_yx, inv_S0_yy

    @staticmethod
    def _build_log_det_S0(
        log_det_C: SymbolicArtifact,
        log_det_V: SymbolicArtifact,
        lambdify_args: tuple[Any, ...],
    ) -> SymbolicArtifact:
        expr = log_det_C.expr + log_det_V.expr
        func = lambdify(lambdify_args, expr, modules="jax")
        return SymbolicArtifact(expr, func)

    # 本当は log_det_S0 = log_det_C + log_det_V - d_y log(12) だが、
    # 定数項は尤度評価に影響しないので省略


class InfinitesimalGenerator:
    """Help apply the infinitesimal generator iteratively.

    無限小生成作用素 L を繰り返し適用するための補助クラス。
    """

    def __init__(
        self, model: DegenerateDiffusionProcess, symbolics: SymbolicPrecomputation
    ) -> None:
        """Cache generator results to accelerate repeated evaluations.

        生成作用素の結果をキャッシュし、繰り返し計算を効率化する。
        """
        self.model = model
        self.symbolics = symbolics
        self._L0_cache: dict[tuple[int, Any], SymbolicArtifact] = {}

    def _normalize_tensor(self, tensor: object) -> SympyTensor:
        if isinstance(tensor, Array):
            return tensor
        if isinstance(tensor, ImmutableDenseNDimArray):
            return Array(tensor)
        if isinstance(tensor, Basic):
            return tensor
        sympified = sp.sympify(tensor)
        if isinstance(sympified, Basic):
            return sympified
        msg = f"Unsupported tensor type after sympify: {type(sympified)}"
        raise TypeError(msg)

    def _tensor_cache_key(self, tensor: SympyTensor) -> SympyTensorKey:
        if isinstance(tensor, Array):
            immutable: ImmutableDenseNDimArray = tensor.as_immutable()
            return immutable
        return tensor

    def _apply_L(self, tensor: SympyTensor) -> SympyTensor:
        if isinstance(tensor, Array):
            return tensor.applyfunc(self._L_elem_once)
        if isinstance(tensor, (Expr, Basic)):
            return self._L_elem_once(tensor)
        msg = f"Unsupported tensor type for infinitesimal generator: {type(tensor)}"
        raise TypeError(msg)

    def _L_elem_once(self, f_elem: Basic) -> Basic:
        if not isinstance(f_elem, Expr) or f_elem.is_constant(simplify=False):
            return S(0)
        f_elem_sym = sp.sympify(f_elem)
        df_dx = derive_by_array(f_elem_sym, self.model.x)
        df_dy = derive_by_array(f_elem_sym, self.model.y)
        d2f_dx2 = derive_by_array(df_dx, self.model.x)

        term1 = einsum_sympy("i,i->", self.model.A.expr, df_dx)
        term2 = einsum_sympy("i,i->", self.model.H.expr, df_dy)
        term3 = (S(1) / 2) * einsum_sympy("ij,ij->", self.symbolics.C.expr, d2f_dx2)
        return term1 + term2 + term3

    def L_0(self, f_tensor: Basic, k: int) -> SymbolicArtifact:
        """Apply the generator ``k`` times and cache the resulting artifact.

        生成作用素を ``k`` 回適用した結果をキャッシュ付きで返す。
        """
        if k < 0:
            msg = "k must be non-negative"
            raise ValueError(msg)
        normalized = self._normalize_tensor(f_tensor)
        cache_key = (k, self._tensor_cache_key(normalized))
        artifact = self._L0_cache.get(cache_key)
        if artifact is not None:
            return artifact
        if k == 0:
            expr = simplify_tensor(normalized)
            func = lambdify(
                (
                    self.model.x,
                    self.model.y,
                    self.model.theta_1,
                    self.model.theta_2,
                    self.model.theta_3,
                ),
                expr,
                modules="jax",
            )
            artifact = SymbolicArtifact(expr, func)
            self._L0_cache[cache_key] = artifact
            return artifact
        prev = self.L_0(f_tensor, k - 1)
        expr = self._apply_L(prev.expr) / S(k)
        expr = simplify_tensor(expr)
        func = lambdify(
            (
                self.model.x,
                self.model.y,
                self.model.theta_1,
                self.model.theta_2,
                self.model.theta_3,
            ),
            expr,
            modules="jax",
        )
        artifact = SymbolicArtifact(expr, func)
        self._L0_cache[cache_key] = artifact
        return artifact


class QuasiLikelihoodEvaluator:
    """Evaluate quasi-likelihoods numerically from symbolic artifacts.

    記号計算の成果物を使って疑似尤度を数値評価するためのクラス。
    """

    def __init__(
        self,
        model: DegenerateDiffusionProcess,
        symbolics: SymbolicPrecomputation,
        generator: InfinitesimalGenerator,
    ) -> None:
        """Bundle the symbolic data and generator required for quasi-likelihood scans.

        疑似尤度の走査に必要な記号データと補助クラスを束ねて保持する。
        """
        self.model = model
        self.symbolics = symbolics
        self.generator = generator
        self._S_cache: dict[
            int, tuple[SymbolicArtifact, SymbolicArtifact, SymbolicArtifact, SymbolicArtifact]
        ] = {}

    def Dx_func(
        self,
        L0_x_funcs: tuple[Callable, ...],
        x_j: JaxArray,
        x_j_1: JaxArray,
        y_j_1: JaxArray,
        theta_2_val: JaxArray,
        theta_1_bar: JaxArray,
        theta_2_bar: JaxArray,
        theta_3_bar: JaxArray,
        h: float,
        k_arg: int,
    ) -> JaxArray:
        r"""Compute the scaled \(\Delta x\) used by the quasi-likelihood.

        疑似尤度で利用する \(\Delta x\) をスケール済みで算出する。
        """
        if h <= 0:
            msg = "h must be positive in Dx_func"
            raise ValueError(msg)
        DX_SCALE = jnp.power(h, -0.5)
        D_x = DX_SCALE * (x_j - x_j_1)
        if k_arg >= 1:
            A_val = self.model.A_func(x_j_1, y_j_1, theta_2_val)
            D_x -= DX_SCALE * jnp.power(h, 1.0) * A_val
        if k_arg >= 2:
            args_for_L0_bar = (x_j_1, y_j_1, theta_1_bar, theta_2_bar, theta_3_bar)
            for m_loop in range(2, k_arg + 1):
                if m_loop >= len(L0_x_funcs):
                    msg = (
                        f"L0_x_funcs index {m_loop} unavailable; "
                        f"only {len(L0_x_funcs)} terms present."
                    )
                    raise IndexError(msg)
                h_pow_m = jnp.power(h, float(m_loop))
                L0_x_m_val = L0_x_funcs[m_loop](*args_for_L0_bar)
                D_x -= DX_SCALE * h_pow_m * L0_x_m_val
        return D_x

    def Dy_func(
        self,
        L0_y_funcs: tuple[Callable, ...],
        y_j: JaxArray,
        y_j_1: JaxArray,
        x_j_1: JaxArray,
        theta_3_val: JaxArray,
        theta_1_bar: JaxArray,
        theta_2_bar: JaxArray,
        theta_3_bar: JaxArray,
        h: float,
        k_arg: int,
    ) -> JaxArray:
        r"""Compute the scaled \(\Delta y\) used by the quasi-likelihood.

        疑似尤度で利用する \(\Delta y\) をスケール済みで算出する。
        """
        if h <= 0:
            return jnp.zeros_like(y_j, dtype=y_j.dtype)
        DY_SCALE = jnp.power(h, -1.5)
        D_y = DY_SCALE * (y_j - y_j_1)
        if k_arg >= 1:
            H_val = self.model.H_func(x_j_1, y_j_1, theta_3_val)
            args_for_L0_bar = (x_j_1, y_j_1, theta_1_bar, theta_2_bar, theta_3_val)
            L0_y_m_val = L0_y_funcs[2](*args_for_L0_bar)
            D_y -= DY_SCALE * (jnp.power(h, 1.0) * H_val + jnp.power(h, 2.0) * L0_y_m_val)
        if k_arg >= 2:
            args_for_L0_bar = (x_j_1, y_j_1, theta_1_bar, theta_2_bar, theta_3_bar)
            for m_loop in range(3, k_arg + 2):
                if m_loop >= len(L0_y_funcs):
                    msg = (
                        f"L0_y_funcs index {m_loop} unavailable; "
                        f"only {len(L0_y_funcs)} terms present."
                    )
                    raise IndexError(msg)
                h_pow_m = jnp.power(h, float(m_loop))
                L0_y_m_val = L0_y_funcs[m_loop](*args_for_L0_bar)
                D_y -= DY_SCALE * h_pow_m * L0_y_m_val
        return D_y

    def S(
        self, k: int
    ) -> tuple[SymbolicArtifact, SymbolicArtifact, SymbolicArtifact, SymbolicArtifact]:
        """Compute the order-`k` S tensors and return expression/function pairs.

        次数 `k` の `S` テンソルを計算し、式と JAX 関数のペアを返す。
        """
        if k in self._S_cache:
            return self._S_cache[k]
        x_sym = self.model.x
        y_sym = self.model.y
        T_xx = self.generator.L_0(tensorproduct(x_sym, x_sym), k + 1).expr
        T_xy = self.generator.L_0(tensorproduct(x_sym, y_sym), k + 2).expr
        T_yx = self.generator.L_0(tensorproduct(y_sym, x_sym), k + 2).expr
        T_yy = self.generator.L_0(tensorproduct(y_sym, y_sym), k + 3).expr

        def compute_U_component(f1: Array, f2: Array, total_sum_order: int) -> Array:
            U_component = Array(sp_zeros(*tensorproduct(f1, f2).shape))
            for m1 in range(total_sum_order + 1):
                m2 = total_sum_order - m1
                L0_f1_m1 = self.generator.L_0(f1, m1).expr
                L0_f2_m2 = self.generator.L_0(f2, m2).expr
                term = tensorproduct(L0_f1_m1, L0_f2_m2)
                if term.shape != U_component.shape:
                    U_component += Array(term).reshape(*U_component.shape)
                else:
                    U_component += term
            return Array(U_component)

        U_xx = compute_U_component(x_sym, x_sym, k + 1)
        U_xy = compute_U_component(x_sym, y_sym, k + 2)
        U_yx = compute_U_component(y_sym, x_sym, k + 2)
        U_yy = compute_U_component(y_sym, y_sym, k + 3)

        S_xx_expr = Array(T_xx) - Array(U_xx)
        S_xy_expr = Array(T_xy) - Array(U_xy)
        S_yx_expr = Array(T_yx) - Array(U_yx)
        S_yy_expr = Array(T_yy) - Array(U_yy)

        S_xx_expr = cast("Array", simplify_tensor(S_xx_expr))
        S_xy_expr = cast("Array", simplify_tensor(S_xy_expr))
        S_yx_expr = cast("Array", simplify_tensor(S_yx_expr))
        S_yy_expr = cast("Array", simplify_tensor(S_yy_expr))
        lambdify_args = (
            self.model.x,
            self.model.y,
            self.model.theta_1,
            self.model.theta_2,
            self.model.theta_3,
        )
        S_xx_func = lambdify(lambdify_args, S_xx_expr, modules="jax", cse=True)
        S_xy_func = lambdify(lambdify_args, S_xy_expr, modules="jax", cse=True)
        S_yx_func = lambdify(lambdify_args, S_yx_expr, modules="jax", cse=True)
        S_yy_func = lambdify(lambdify_args, S_yy_expr, modules="jax", cse=True)
        S_xx = SymbolicArtifact(S_xx_expr, S_xx_func)
        S_xy = SymbolicArtifact(S_xy_expr, S_xy_func)
        S_yx = SymbolicArtifact(S_yx_expr, S_yx_func)
        S_yy = SymbolicArtifact(S_yy_expr, S_yy_func)
        result = (S_xx, S_xy, S_yx, S_yy)
        self._S_cache[k] = result
        return result

    @staticmethod
    def a(n: int, h: float, k: int, i: int) -> float:
        """Compute the scaling factor :math:`a_n^h`.

        :math:`a_n^h` を計算する。
        """
        n_float = float(n)
        h_float = float(h)
        pow_k = math.pow(h_float, float(k))
        if i == 1:
            return math.pow(n_float, -0.5) + pow_k
        if i == 2:
            return math.pow(n_float * h_float, -0.5) + pow_k
        return math.pow(h_float / n_float, 0.5) + math.pow(h_float, float(k + 1))

    def make_quasi_likelihood_l1_prime_evaluator(
        self, x_series: JaxArray, y_series: JaxArray, h: float, k: int
    ) -> JittedLikelihood:
        """Create the simplified quasi-likelihood :math:`l_1'` evaluator without ``S_0``.

        The returned callable is wrapped with ``jax.jit``。

        返される関数には JAX の ``jit`` が適用される。
        """
        x_series_jnp = jnp.asarray(x_series)
        y_series_jnp = jnp.asarray(y_series)
        n = x_series_jnp.shape[0]
        num_transitions = n - 1
        if num_transitions < 1 or y_series_jnp.shape[0] != n:
            msg = "Time series length must be > 1 and shapes must match for l1'."
            raise ValueError(msg)

        L0_x_funcs = tuple(self.generator.L_0(self.model.x, m).func for m in range(k))
        S_xx_funcs = tuple(self.S(l_s)[0].func for l_s in range(1, k))

        invC_func = self.symbolics.inv_C.func
        logDetC_func = self.symbolics.log_det_C.func

        d_x = self.model.x.shape[0]

        def evaluate_l1_prime(
            theta_1_val: JaxArray,
            theta_1_bar: JaxArray,
            theta_2_bar: JaxArray,
            theta_3_bar: JaxArray,
        ) -> JaxArray:
            theta_1_val = jnp.asarray(theta_1_val)
            theta_1_bar = jnp.asarray(theta_1_bar)
            theta_2_bar = jnp.asarray(theta_2_bar)
            theta_3_bar = jnp.asarray(theta_3_bar)

            result_dtype = jnp.result_type(
                theta_1_val.dtype,
                theta_1_bar.dtype,
                theta_2_bar.dtype,
                theta_3_bar.dtype,
            )

            def scan_body(
                total: JaxArray,
                step_inputs: tuple[JaxArray, JaxArray, JaxArray],
            ) -> tuple[JaxArray, None]:
                x_j, x_j_1, y_j_1 = step_inputs
                invC_val = invC_func(x_j_1, y_j_1, theta_1_val)
                logDetC_val = logDetC_func(x_j_1, y_j_1, theta_1_val)
                Dx_val = self.Dx_func(
                    L0_x_funcs,
                    x_j,
                    x_j_1,
                    y_j_1,
                    theta_2_bar,
                    theta_1_bar,
                    theta_2_bar,
                    theta_3_bar,
                    h,
                    k - 1,
                )
                sum_Sxx_val = jnp.zeros((d_x, d_x), dtype=invC_val.dtype)
                for idx, Sxx_func in enumerate(S_xx_funcs, start=1):
                    sum_Sxx_val += (h**idx) * Sxx_func(
                        x_j_1, y_j_1, theta_1_bar, theta_2_bar, theta_3_bar
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
                return (
                    total_log_likelihood
                    / (2.0 * num_transitions)
                    / (self.a(num_transitions, h, k, 1) ** 2)
                )
            return jnp.full_like(total_log_likelihood, jnp.nan)

        return cast("JittedLikelihood", jax.jit(evaluate_l1_prime))

    def make_quasi_likelihood_l1_evaluator(  # noqa: PLR0915
        self,
        x_series: JaxArray,
        y_series: JaxArray,
        h: float,
        k: int,
    ) -> JittedLikelihood:
        """Create the quasi-likelihood :math:`l_1` evaluator with the ``S_0`` correction.

        The returned callable is wrapped with ``jax.jit``.

        返される関数には JAX の ``jit`` が適用される。
        """
        x_series_jnp = jnp.asarray(x_series)
        y_series_jnp = jnp.asarray(y_series)
        n = x_series_jnp.shape[0]
        num_transitions = n - 1
        if num_transitions < 1 or y_series_jnp.shape[0] != n:
            msg = "Time series length must be > 1 and shapes must match for l1."
            raise ValueError(msg)

        L0_x_funcs = tuple(self.generator.L_0(self.model.x, m).func for m in range(k))
        L0_y_funcs = tuple(self.generator.L_0(self.model.y, m).func for m in range(k + 1))
        S_funcs = tuple(
            (self.S(l_s)[0].func, self.S(l_s)[1].func, self.S(l_s)[2].func, self.S(l_s)[3].func)
            for l_s in range(1, k)
        )
        inv_S0_xx_func = self.symbolics.inv_S0_xx.func
        inv_S0_xy_func = self.symbolics.inv_S0_xy.func
        inv_S0_yx_func = self.symbolics.inv_S0_yx.func
        inv_S0_yy_func = self.symbolics.inv_S0_yy.func
        log_det_S0_func = self.symbolics.log_det_S0.func

        d_x = self.model.x.shape[0]
        d_y = self.model.y.shape[0]

        def evaluate_l1(
            theta_1_val: JaxArray,
            theta_1_bar: JaxArray,
            theta_2_bar: JaxArray,
            theta_3_bar: JaxArray,
        ) -> JaxArray:
            theta_1_val = jnp.asarray(theta_1_val)
            theta_1_bar = jnp.asarray(theta_1_bar)
            theta_2_bar = jnp.asarray(theta_2_bar)
            theta_3_bar = jnp.asarray(theta_3_bar)

            result_dtype = jnp.result_type(
                theta_1_val.dtype,
                theta_1_bar.dtype,
                theta_2_bar.dtype,
                theta_3_bar.dtype,
            )

            def scan_body(
                total: JaxArray,
                step_inputs: tuple[JaxArray, JaxArray, JaxArray, JaxArray],
            ) -> tuple[JaxArray, None]:
                x_j, y_j, x_j_1, y_j_1 = step_inputs
                inv_S0_xx_val = inv_S0_xx_func(x_j_1, y_j_1, theta_1_val, theta_3_bar)
                inv_S0_xy_val = inv_S0_xy_func(x_j_1, y_j_1, theta_1_val, theta_3_bar)
                inv_S0_yx_val = inv_S0_yx_func(x_j_1, y_j_1, theta_1_val, theta_3_bar)
                inv_S0_yy_val = inv_S0_yy_func(x_j_1, y_j_1, theta_1_val, theta_3_bar)
                log_det_S0_val = log_det_S0_func(x_j_1, y_j_1, theta_1_val, theta_3_bar)

                Dx_val = self.Dx_func(
                    L0_x_funcs,
                    x_j,
                    x_j_1,
                    y_j_1,
                    theta_2_bar,
                    theta_1_bar,
                    theta_2_bar,
                    theta_3_bar,
                    h,
                    k - 1,
                )
                Dy_val = self.Dy_func(
                    L0_y_funcs,
                    y_j,
                    y_j_1,
                    x_j_1,
                    theta_3_bar,
                    theta_1_bar,
                    theta_2_bar,
                    theta_3_bar,
                    h,
                    k - 1,
                )

                sum_S_xx = jnp.zeros((d_x, d_x), dtype=inv_S0_xx_val.dtype)
                sum_S_xy = jnp.zeros((d_x, d_y), dtype=inv_S0_xx_val.dtype)
                sum_S_yx = jnp.zeros((d_y, d_x), dtype=inv_S0_xx_val.dtype)
                sum_S_yy = jnp.zeros((d_y, d_y), dtype=inv_S0_xx_val.dtype)

                for idx, s_funcs in enumerate(S_funcs, start=1):
                    s_xx = s_funcs[0](x_j_1, y_j_1, theta_1_bar, theta_2_bar, theta_3_bar)
                    s_xy = s_funcs[1](x_j_1, y_j_1, theta_1_bar, theta_2_bar, theta_3_bar)
                    s_yx = s_funcs[2](x_j_1, y_j_1, theta_1_bar, theta_2_bar, theta_3_bar)
                    s_yy = s_funcs[3](x_j_1, y_j_1, theta_1_bar, theta_2_bar, theta_3_bar)
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

                tr_xx = jnp.einsum("ij,ij->", inv_S0_xx_val, sum_S_xx)
                tr_xy = jnp.einsum("ij,ij->", inv_S0_xy_val, sum_S_xy)
                tr_yx = jnp.einsum("ij,ij->", inv_S0_yx_val, sum_S_yx)
                tr_yy = jnp.einsum("ij,ij->", inv_S0_yy_val, sum_S_yy)
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
                return (
                    total_log_likelihood
                    / (2.0 * num_transitions)
                    / (self.a(num_transitions, h, k, 1) ** 2)
                )
            return jnp.full_like(total_log_likelihood, jnp.nan)

        return cast("JittedLikelihood", jax.jit(evaluate_l1))

    def make_quasi_likelihood_l2_evaluator(
        self, x_series: JaxArray, y_series: JaxArray, h: float, k: int
    ) -> JittedLikelihood:
        """Create the quasi-likelihood :math:`l_2` evaluator for estimating ``theta_2``.

        The returned callable is wrapped with ``jax.jit``.

        返される関数には JAX の ``jit`` が適用される。
        """
        x_series_jnp = jnp.asarray(x_series)
        y_series_jnp = jnp.asarray(y_series)
        n = x_series_jnp.shape[0]
        num_transitions = n - 1
        if num_transitions < 1 or y_series_jnp.shape[0] != n:
            msg = "Time series length must be > 1 and shapes must match for l2."
            raise ValueError(msg)

        L0_x_funcs = tuple(self.generator.L_0(self.model.x, m).func for m in range(k + 1))
        invC_func = self.symbolics.inv_C.func

        def evaluate_l2(
            theta_2_val: JaxArray,
            theta_1_bar: JaxArray,
            theta_2_bar: JaxArray,
            theta_3_bar: JaxArray,
        ) -> JaxArray:
            theta_2_val = jnp.asarray(theta_2_val)
            theta_1_bar = jnp.asarray(theta_1_bar)
            theta_2_bar = jnp.asarray(theta_2_bar)
            theta_3_bar = jnp.asarray(theta_3_bar)

            result_dtype = jnp.result_type(
                theta_2_val.dtype,
                theta_1_bar.dtype,
                theta_2_bar.dtype,
                theta_3_bar.dtype,
            )

            def scan_body(
                total: JaxArray,
                step_inputs: tuple[JaxArray, JaxArray, JaxArray],
            ) -> tuple[JaxArray, None]:
                x_j, x_j_1, y_j_1 = step_inputs
                invC_val = invC_func(x_j_1, y_j_1, theta_1_bar)
                Dx_val = self.Dx_func(
                    L0_x_funcs,
                    x_j,
                    x_j_1,
                    y_j_1,
                    theta_2_val,
                    theta_1_bar,
                    theta_2_bar,
                    theta_3_bar,
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
                return (
                    total_log_likelihood
                    / (2.0 * h * num_transitions)
                    / (self.a(num_transitions, h, k, 2) ** 2)
                )
            return jnp.full_like(total_log_likelihood, jnp.nan)

        return cast("JittedLikelihood", jax.jit(evaluate_l2))

    def make_quasi_likelihood_l3_evaluator(
        self, x_series: JaxArray, y_series: JaxArray, h: float, k: int
    ) -> JittedLikelihood:
        """Create the quasi-likelihood :math:`l_3` evaluator for estimating ``theta_3``.

        The returned callable is wrapped with ``jax.jit``.

        返される関数には JAX の ``jit`` が適用される。
        """
        x_series_jnp = jnp.asarray(x_series)
        y_series_jnp = jnp.asarray(y_series)
        n = x_series_jnp.shape[0]
        num_transitions = n - 1
        if num_transitions < 1 or y_series_jnp.shape[0] != n:
            msg = "Time series length must be > 1 and shapes must match for l3."
            raise ValueError(msg)

        L0_x_funcs = tuple(self.generator.L_0(self.model.x, m).func for m in range(k + 1))
        L0_y_funcs = tuple(self.generator.L_0(self.model.y, m).func for m in range(k + 2))

        invV_func = self.symbolics.inv_V.func
        pHTinvV_func = self.symbolics.partial_x_H_transpose_inv_V.func

        def evaluate_l3(
            theta_3_val: JaxArray,
            theta_1_bar: JaxArray,
            theta_2_bar: JaxArray,
            theta_3_bar: JaxArray,
        ) -> JaxArray:
            theta_3_val = jnp.asarray(theta_3_val)
            theta_1_bar = jnp.asarray(theta_1_bar)
            theta_2_bar = jnp.asarray(theta_2_bar)
            theta_3_bar = jnp.asarray(theta_3_bar)

            result_dtype = jnp.result_type(
                theta_3_val.dtype,
                theta_1_bar.dtype,
                theta_2_bar.dtype,
                theta_3_bar.dtype,
            )

            def scan_body(
                total: JaxArray,
                step_inputs: tuple[JaxArray, JaxArray, JaxArray, JaxArray],
            ) -> tuple[JaxArray, None]:
                x_j, y_j, x_j_1, y_j_1 = step_inputs
                invV_val = invV_func(x_j_1, y_j_1, theta_1_bar, theta_3_bar)
                pHTinvV_val = pHTinvV_func(x_j_1, y_j_1, theta_1_bar, theta_3_bar)
                Dx_val = self.Dx_func(
                    L0_x_funcs,
                    x_j,
                    x_j_1,
                    y_j_1,
                    theta_2_bar,
                    theta_1_bar,
                    theta_2_bar,
                    theta_3_bar,
                    h,
                    k,
                )
                Dy_val = self.Dy_func(
                    L0_y_funcs,
                    y_j,
                    y_j_1,
                    x_j_1,
                    theta_3_val,
                    theta_1_bar,
                    theta_2_bar,
                    theta_3_bar,
                    h,
                    k,
                )

                term1_l3 = -jnp.einsum("ij,i,j->", invV_val, Dy_val, Dy_val)
                term2_l3 = jnp.einsum("i,ik,k->", Dx_val, pHTinvV_val, Dy_val)
                step_likelihood = term1_l3 + term2_l3
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
                return (
                    (6.0 * h * total_log_likelihood)
                    / num_transitions
                    / (self.a(num_transitions, h, k, 3) ** 2)
                )
            return jnp.full_like(total_log_likelihood, jnp.nan)

        return cast("JittedLikelihood", jax.jit(evaluate_l3))


class LikelihoodEvaluator:
    """Facade that pairs symbolic preparation with quasi-likelihood evaluation.

    記号準備と疑似尤度評価をまとめて扱うファサードクラス。
    """

    def __init__(
        self,
        model: DegenerateDiffusionProcess,
        *,
        precomputed: SymbolicPrecomputation | None = None,
    ) -> None:
        """Build helper objects around ``model`` and reuse precomputation when supplied.

        ``model`` を中心に補助オブジェクトを構築し、必要なら既存の前処理結果を再利用する。
        """
        self.model = model
        self.A = model.A
        self.B = model.B
        self.H = model.H
        self.symbolics = precomputed or SymbolicLikelihoodPreparer(model).prepare()
        self.generator = InfinitesimalGenerator(model, self.symbolics)
        self.quasi = QuasiLikelihoodEvaluator(model, self.symbolics, self.generator)

    def L_0(self, f_tensor: Basic, k: int) -> Basic:
        r"""Delegate to :class:`InfinitesimalGenerator` to evaluate \(L_0\).

        内部の :class:`InfinitesimalGenerator` に委譲して \(L_0\) を計算する。
        """
        return self.generator.L_0(f_tensor, k)

    def Dx_func(
        self,
        L0_x_funcs: tuple[Callable, ...],
        x_j: JaxArray,
        x_j_1: JaxArray,
        y_j_1: JaxArray,
        theta_2_val: JaxArray,
        theta_1_bar: JaxArray,
        theta_2_bar: JaxArray,
        theta_3_bar: JaxArray,
        h: float,
        k_arg: int,
    ) -> JaxArray:
        """Forward to :meth:`QuasiLikelihoodEvaluator.Dx_func` with explicit arguments.

        :class:`QuasiLikelihoodEvaluator` の ``Dx_func`` を同じ引数で呼び出す。
        """
        return self.quasi.Dx_func(
            L0_x_funcs,
            x_j,
            x_j_1,
            y_j_1,
            theta_2_val,
            theta_1_bar,
            theta_2_bar,
            theta_3_bar,
            h,
            k_arg,
        )

    def Dy_func(
        self,
        L0_y_funcs: tuple[Callable, ...],
        y_j: JaxArray,
        y_j_1: JaxArray,
        x_j_1: JaxArray,
        theta_3_val: JaxArray,
        theta_1_bar: JaxArray,
        theta_2_bar: JaxArray,
        theta_3_bar: JaxArray,
        h: float,
        k_arg: int,
    ) -> JaxArray:
        """Forward to :meth:`QuasiLikelihoodEvaluator.Dy_func` with explicit arguments.

        :class:`QuasiLikelihoodEvaluator` の ``Dy_func`` を同じ引数で呼び出す。
        """
        return self.quasi.Dy_func(
            L0_y_funcs,
            y_j,
            y_j_1,
            x_j_1,
            theta_3_val,
            theta_1_bar,
            theta_2_bar,
            theta_3_bar,
            h,
            k_arg,
        )

    def S(
        self, k: int
    ) -> tuple[SymbolicArtifact, SymbolicArtifact, SymbolicArtifact, SymbolicArtifact]:
        """Return the symbolic and JAX artifacts for the ``S`` tensors.

        ``S`` テンソルの記号式と JAX 評価関数をまとめた成果物を返す。
        """
        return self.quasi.S(k)

    def make_quasi_likelihood_l1_prime_evaluator(
        self, x_series: JaxArray, y_series: JaxArray, h: float, k: int
    ) -> JittedLikelihood:
        """Create the simplified quasi-likelihood :math:`l_1'` evaluator.

        The returned callable is wrapped with ``jax.jit``.

        簡略版疑似尤度 :math:`l_1'` の評価関数を生成して返す。
        """
        return self.quasi.make_quasi_likelihood_l1_prime_evaluator(x_series, y_series, h, k)

    def make_quasi_likelihood_l1_evaluator(
        self,
        x_series: JaxArray,
        y_series: JaxArray,
        h: float,
        k: int,
    ) -> JittedLikelihood:
        """Create the quasi-likelihood :math:`l_1` evaluator with optional correction.

        The returned callable is wrapped with ``jax.jit``.

        補正込みの疑似尤度 :math:`l_1` の評価関数を生成して返す。
        """
        return self.quasi.make_quasi_likelihood_l1_evaluator(x_series, y_series, h, k)

    def make_quasi_likelihood_l2_evaluator(
        self, x_series: JaxArray, y_series: JaxArray, h: float, k: int
    ) -> JittedLikelihood:
        """Create the quasi-likelihood :math:`l_2` evaluator specialized for ``theta_2``.

        The returned callable is wrapped with ``jax.jit``.

        パラメータ ``theta_2`` に特化した疑似尤度 :math:`l_2` を評価する関数を生成する。
        """
        return self.quasi.make_quasi_likelihood_l2_evaluator(x_series, y_series, h, k)

    def make_quasi_likelihood_l3_evaluator(
        self, x_series: JaxArray, y_series: JaxArray, h: float, k: int
    ) -> JittedLikelihood:
        """Create the quasi-likelihood :math:`l_3` evaluator specialized for ``theta_3``.

        The returned callable is wrapped with ``jax.jit``.

        パラメータ ``theta_3`` に特化した疑似尤度 :math:`l_3` を評価する関数を生成する。
        """
        return self.quasi.make_quasi_likelihood_l3_evaluator(x_series, y_series, h, k)

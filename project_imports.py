# project_imports.py

# --- SymPy ---
import sympy as sp
from sympy import (
    Array, symbols, factorial, tensorproduct, log, det, Matrix,
    Expr, Basic, S, zeros, srepr, oo
)
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.tensor.array import derive_by_array, ImmutableDenseNDimArray
from sympy.utilities.lambdify import lambdify

# --- NumPy ---
import numpy as np
# `LikelihoodEvaluator.py` で直接使われているNumPy関数を再エクスポート
einsum = np.einsum
power = np.power
isfinite = np.isfinite
nan = np.nan

# --- Typing (型ヒント) ---
from typing import Dict, Callable, Tuple, Any, Optional, List, TYPE_CHECKING, Sequence # <--- Sequence を追加

# --- SciPy ---
from scipy import optimize

# --- Bayesian Estimation (PyMC関連) ---
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
import arviz as az

# --- 標準ライブラリ ---
import logging
from collections import defaultdict # einsum_sympy.py で使用
from itertools import product     # einsum_sympy.py で使用
from dataclasses import dataclass # ここで dataclass をインポート

# --- Interactive Widgets (UI/Notebook用、コアロジックに必須でなければ分離も検討) ---
import ipywidgets as widgets
from ipywidgets import (
    interact, fixed, FloatSlider, IntSlider, FloatText, IntText
)
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class SymbolicArtifact:
    """Container for a SymPy expression and its callable evaluator.

    Japanese: SymPy の式と JAX で評価可能な関数をひとまとめに管理する入れ物です.
    """

    expr: Any
    func: Callable[..., Any]

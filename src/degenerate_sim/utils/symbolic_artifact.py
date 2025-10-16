from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SymbolicArtifact:
    """SymPy の式と JAX で評価可能な関数をひとまとまりに管理する入れ物。"""

    expr: Any
    func: Callable

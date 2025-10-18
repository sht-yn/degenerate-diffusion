from collections import defaultdict
from itertools import product

import sympy as sp
from sympy import Expr, ImmutableDenseNDimArray


def _zero_expr() -> Expr:
    """Return the additive identity for SymPy expressions."""
    return sp.Integer(0)


def einsum_sympy(subscripts: str, *operands: ImmutableDenseNDimArray) -> ImmutableDenseNDimArray:
    """Compute Einstein summation for SymPy tensors.

    Japanese: SymPy 配列を使ってアインシュタイン縮約を計算します。
    """
    input_str, output_str = subscripts.replace(" ", "").split("->")
    input_subs = input_str.split(",")

    shapes: list[tuple[int, ...]] = [op.shape for op in operands]
    dim_map: dict[str, int] = {}
    dim_order: list[str] = []
    for subs, shape in zip(input_subs, shapes, strict=False):
        if len(subs) != len(shape):
            msg = f"Subscript {subs} does not match shape {shape}"
            raise ValueError(msg)
        for index_symbol, dimension in zip(subs, shape, strict=False):
            if index_symbol in dim_map:
                if dim_map[index_symbol] != dimension:
                    msg = (
                        f"Inconsistent dimension for subscript '{index_symbol}': "
                        f"{dim_map[index_symbol]} vs {dimension}"
                    )
                    raise ValueError(msg)
            else:
                dim_map[index_symbol] = dimension
                dim_order.append(index_symbol)

    full_indices = [range(dim_map[symbol]) for symbol in dim_order]
    index_pos: dict[str, int] = {symbol: position for position, symbol in enumerate(dim_order)}

    output_shape: tuple[int, ...] = tuple(dim_map[symbol] for symbol in output_str)

    result_dict: defaultdict[tuple[int, ...], Expr] = defaultdict(_zero_expr)
    for full_idx in product(*full_indices):
        term: Expr = sp.Integer(1)
        for subs, operand in zip(input_subs, operands, strict=False):
            operand_index = tuple(full_idx[index_pos[symbol]] for symbol in subs)
            term *= operand[operand_index]
        output_index = tuple(full_idx[index_pos[symbol]] for symbol in output_str)
        result_dict[output_index] += term

    output_ranges = [range(dimension) for dimension in output_shape]
    flat_data = [sp.simplify(result_dict[output_index]) for output_index in product(*output_ranges)]
    return ImmutableDenseNDimArray(flat_data, output_shape)

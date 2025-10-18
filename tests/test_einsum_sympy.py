import sympy as sp
from sympy import ImmutableDenseNDimArray

from degenerate_diffusion.utils.einsum_sympy import einsum_sympy


def test_einsum_vector_outer_product() -> None:
    x0, x1 = sp.symbols("x0 x1")
    y0, y1, y2 = sp.symbols("y0 y1 y2")
    u = ImmutableDenseNDimArray([x0, x1], (2,))
    v = ImmutableDenseNDimArray([y0, y1, y2], (3,))

    result = einsum_sympy("i,j->ij", u, v)

    expected = ImmutableDenseNDimArray(
        [
            x0 * y0,
            x0 * y1,
            x0 * y2,
            x1 * y0,
            x1 * y1,
            x1 * y2,
        ],
        (2, 3),
    )

    assert result == expected


def test_einsum_matrix_vector_product() -> None:
    a00, a01, a10, a11 = sp.symbols("a00 a01 a10 a11")
    b0, b1 = sp.symbols("b0 b1")
    mat = ImmutableDenseNDimArray([a00, a01, a10, a11], (2, 2))
    vec = ImmutableDenseNDimArray([b0, b1], (2,))

    result = einsum_sympy("ij,j->i", mat, vec)

    expected = ImmutableDenseNDimArray(
        [a00 * b0 + a01 * b1, a10 * b0 + a11 * b1],
        (2,),
    )

    assert result == expected


def test_einsum_rank_three_contraction() -> None:
    a0, a1, a2, a3, a4, a5, a6, a7 = sp.symbols("a0:8")
    b0, b1 = sp.symbols("b0 b1")
    tensor = ImmutableDenseNDimArray([a0, a1, a2, a3, a4, a5, a6, a7], (2, 2, 2))
    vec = ImmutableDenseNDimArray([b0, b1], (2,))

    result = einsum_sympy("ijk,k->ij", tensor, vec)

    expected = ImmutableDenseNDimArray(
        [
            a0 * b0 + a1 * b1,
            a2 * b0 + a3 * b1,
            a4 * b0 + a5 * b1,
            a6 * b0 + a7 * b1,
        ],
        (2, 2),
    )

    assert result == expected

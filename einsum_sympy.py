from project_imports import ImmutableDenseNDimArray, defaultdict, product, sp  # Sも追加


def einsum_sympy(subscripts, *operands):
    """SymPy版 einsum: einsum_sympy("ij,kij->k", A, B)."""
    input_str, output_str = subscripts.replace(" ", "").split("->")
    input_subs = input_str.split(",")

    # 入力テンソルのshapeと添字の対応を構築
    shapes = [op.shape for op in operands]
    dim_map = {}
    dim_order = []
    for subs, shape in zip(input_subs, shapes, strict=False):
        assert len(subs) == len(shape), f"Subscript {subs} does not match shape {shape}"
        for s, d in zip(subs, shape, strict=False):
            if s in dim_map:
                if dim_map[s] != d:
                    msg = f"Inconsistent dimension for subscript '{s}': {dim_map[s]} vs {d}"
                    raise ValueError(msg)
            else:
                dim_map[s] = d
                dim_order.append(s)

    # 全次元の順列を生成
    full_indices = [range(dim_map[s]) for s in dim_order]
    index_pos = {s: i for i, s in enumerate(dim_order)}

    # 出力のshape決定
    output_shape = tuple(dim_map[s] for s in output_str)

    # 結果を計算
    result_dict = defaultdict(lambda: 0)
    for full_idx in product(*full_indices):
        term = 1
        for subs, op in zip(input_subs, operands, strict=False):
            op_idx = tuple(full_idx[index_pos[s]] for s in subs)
            term *= op[op_idx]
        out_idx = tuple(full_idx[index_pos[s]] for s in output_str)
        result_dict[out_idx] += term

    # 結果を ImmutableDenseNDimArray に変換
    flat_data = []
    for out_idx in product(*[range(d) for d in output_shape]):
        flat_data.append(sp.simplify(result_dict[out_idx]))
    return ImmutableDenseNDimArray(flat_data, output_shape)


# テスト：ベクトルのテンソル積 "i,j->ij"
x0, x1 = sp.symbols("x0 x1")
y0, y1, y2 = sp.symbols("y0 y1 y2")
u = ImmutableDenseNDimArray([x0, x1])  # shape (2,)
v = ImmutableDenseNDimArray([y0, y1, y2])  # shape (3,)

# outer product
tensor_product_result = einsum_sympy("i,j->ij", u, v)
tensor_product_result.shape, tensor_product_result

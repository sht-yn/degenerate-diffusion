# JAX の制御フロー: scan / while_loop / fori_loop の使い分け

このノートでは、JAX の代表的なループ演算 `lax.scan` / `lax.while_loop` / `lax.fori_loop` の違いを、等価な擬似 Python コードとともに整理します。最後に本リポジトリの `parameter_estimator_new.py` に実装した Newton 法（最大化）での `while_loop` 例を解説します。

## 共通事項
- いずれも「carry（状態）を受け取り、次ステップの carry を返す」スタイル。
- JIT / 自動微分に対応（ループ本体は JAX 演算を用いる）。
- carry の構造（PyTree）および各葉の shape/dtype は各ステップで不変である必要があります。

---

## lax.scan
- 固定長の反復。系列入力 `xs` を要素ごとに処理し、各ステップの出力を自動でスタックして返します。
- 署名（概念図）: `(carry, ys) = lax.scan(f, init, xs)`
- 返り値: `final_carry, stacked_outputs`

### 等価な擬似 Python コード
```python
def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:                # 固定長で最後まで回る
        carry, y = f(carry, x)  # f: (carry, x) -> (carry, y)
        ys.append(y)
    return carry, np.stack(ys)
```

### 特徴
- 早期停止はできません（必要なら done フラグで更新をマスク）。
- 逆伝播が効率的に実装されており、長い系列タスクに向いています。
- 履歴（各ステップの出力）が必要な場合に第一候補。

---

## lax.while_loop
- 収束判定など、反復回数が実行時に決まるループ。
- 署名（概念図）: `carry = lax.while_loop(cond, body, init_carry)`
- 返り値: `final_carry`（履歴が必要なら固定長バッファを carry に入れて自前で保持）。

### 等価な擬似 Python コード
```python
def while_loop(cond, body, init_carry):
    carry = init_carry
    while cond(carry):        # cond: carry -> bool (JAXでは jnp.bool_)
        carry = body(carry)   # body: carry -> carry
    return carry
```

### 特徴
- 早期停止が可能（収束判定など）。
- `xs` のような系列入力引数は無いので、各ステップの入力を変えたい場合は carry 側に入れて自分で管理します（例: `i` と `xs` を持ち、`x_i = xs[i]` を body で取り出す）。
- 履歴を取りたい場合は、固定長の配列を carry に入れ、`dynamic_update_slice` や `.at[i].set` で埋める。

---

## lax.fori_loop
- 固定回数の軽量な for ループ（履歴は返さない）。
- 署名（概念図）: `carry = lax.fori_loop(start, stop, body, init_carry)`
- 返り値: `final_carry`

### 等価な擬似 Python コード
```python
def fori_loop(start, stop, body, init_carry):
    carry = init_carry
    for i in range(start, stop):  # 固定回数
        carry = body(i, carry)     # body: (i, carry) -> carry
    return carry
```

### 特徴
- 履歴が不要かつ固定回数で済む処理に向きます。
- 逆伝播は可能ですが、履歴が必要な場合は `scan` を使うのが通常です。

---

## 本リポジトリの例: Newton 法（最大化）での while_loop
ファイル: `src/degenerate_diffusion/estimation/parameter_estimator_new.py`

ニュートン上昇（最大化）において、勾配ノルム `||grad|| <= tol` で停止させるため `jax.lax.while_loop` を使っています。

### 要約
- carry = `(theta, it, converged)`
  - `theta`: 現在の推定パラメータ（JAX Array）
  - `it`: 反復カウント（jnp.int32）
  - `converged`: 収束フラグ（jnp.bool_）
- cond: `it < max_iters and not converged`
- body: 勾配・ヘッセを計算し、最大化用の安定化ヘッセ `H_safe = H_sym - eps * I` でニュートン上昇更新 → ボックス制約に射影

### 擬似コード（本実装の骨子）
```python
import jax
import jax.numpy as jnp

max_iters = 10_000
Tol = 1e-7
Damp = 0.1
Eps = 1e-8
bounds = ...  # (d, 2)

def objective(theta, aux):
    return ...  # スカラー（最大化）

@jax.jit
def newton_solve(theta0, aux):
    theta0 = jnp.asarray(theta0)
    b = bounds.astype(theta0.dtype)
    false = jnp.zeros((), dtype=jnp.bool_)

    def obj(th):
        return jnp.asarray(objective(th, aux))

    def grad(th):
        return jax.grad(obj)(th)

    def hess(th):
        return jax.hessian(obj)(th)

    def cond(carry):
        th, it, conv = carry
        return jnp.logical_and(it < max_iters, jnp.logical_not(conv))

    def body(carry):
        th, it, _ = carry
        g = grad(th)
        H = hess(th)
        H_sym = 0.5 * (H + H.T)
        eye = jnp.eye(th.shape[0], dtype=th.dtype)
        H_safe = H_sym - Eps * eye  # 最大化: 負定方向に安定化
        delta = jnp.linalg.solve(H_safe, g)
        th_next = th - Damp * delta
        th_next = jnp.clip(th_next, b[:, 0], b[:, 1])
        conv = jnp.linalg.norm(g) <= Tol
        return th_next, it + 1, conv

    th_fin, _, _ = jax.lax.while_loop(
        cond,
        body,
        (theta0, jnp.asarray(0, jnp.int32), false),
    )
    return th_fin
```

---

## 使い分けのまとめ
- 収束や条件付き停止が必要 → `while_loop`
- 固定長・履歴（系列出力）が必要 → `scan`
- 固定長・履歴不要 → `fori_loop`

いずれの場合も、
- carry の構造・各葉の shape/dtype を不変にする
- 分岐は `lax.cond`、繰り返しは `scan/while_loop/fori_loop`
- 早期停止を固定長でエミュレートしたい場合は mask を用意する
といった JAX のお作法を守ると、JIT/自動微分と相性よく動作します。

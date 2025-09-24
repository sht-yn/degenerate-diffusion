# JAX 利用メモ

## 1. `degenerate_sim.processes.degenerate_diffusion_process_jax`

### SymPy から JAX への変換
- `degenerate_sim/processes/degenerate_diffusion_process_jax.py` では、クラス `DegenerateDiffusionProcess` が `A`, `B`, `H` を SymPy で記述し、`lambdify(..., modules="jax")` で JAX 対応関数 (`A_func`, `B_func`, `H_func`) に変換する。

### `_simulate_jax_core` と `static_argnums`
```python
@partial(jax.jit, static_argnums=(0, 5, 6, 9, 10))
def _simulate_jax_core(self, theta_1_val, theta_2_val, theta_3_val,
                      key, t_max, burn_out, x0_val, y0_val, dt, step_stride_static):
    ...
```
- **静的引数 (static argnums)**: `self`, `t_max`, `burn_out`, `dt`, `step_stride_static` を JIT コンパイル時に固定値として扱う。これにより `round(...)` や `range(...)` のような Python 側演算でトレーサ（JAX が追跡するシンボリック値）が混ざるのを防ぎ、コンパイル済みの XLA 実装をそのまま再利用できる。
- **動的引数**: `theta*_val`, `key`, `x0_val`, `y0_val` などの実行時に変わる値は JIT 後の実行フェーズで流し込むだけ。これらを変えても再コンパイルは不要で、既存の XLA コードにデータだけ差し替えて評価する。
- **重要**: `lax.scan` の `length` など、制御フローを決める値はコンパイル時に既知（静的）である必要がある。ここを動的にすると XLA がループ展開を確定できずエラーになる／再コンパイルが頻発する。

> **覚えておくこと**: 「ループ回数やスライス幅など制御構造に影響する値 ⇒ static」「パラメータや乱数キーのように毎回変わるが演算本体は同じ ⇒ dynamic」。これを守ると一度の JIT で高速な実行が繰り返せる。

### `lax.scan` の利用
```python
def em_step(carry, _):
    xt, yt, current_key = carry
    key_dW, next_key = jax.random.split(current_key)
    A_val = self.A_func(xt, yt, theta_2_val)
    B_val = self.B_func(xt, yt, theta_1_val)
    H_val = self.H_func(xt, yt, theta_3_val)
    dW = jax.random.normal(key_dW, (r,)) * jnp.sqrt(dt)
    diffusion_term = jnp.dot(B_val, dW)
    xt_next = xt + A_val * dt + diffusion_term
    yt_next = yt + H_val * dt
    return (xt_next, yt_next, next_key), (xt_next, yt_next)

initial_carry = (x0_val, y0_val, key)
final_carry, (x_results, y_results) = lax.scan(
    em_step, initial_carry, None, length=total_steps_for_scan_py)

x_all = jnp.concatenate((jnp.expand_dims(x0_val, 0), x_results), axis=0)
y_all = jnp.concatenate((jnp.expand_dims(y0_val, 0), y_results), axis=0)
```
- `scan` はステップごとの出力を `(x_results, y_results)` のように自動で積み上げる。
- 初期値を `jnp.concatenate` で先頭に付けてパスを作る。
  - `jnp.expand_dims(x0_val, 0)` によって `(d_x,) → (1, d_x)` に形を揃え、`x_results` (`(N, d_x)`) の先頭に連結し、初期値を含む `(N+1, d_x)` の軌道を得る。
- **擬似コードのイメージ**
  ```python
  def scan(f, init, xs, length=None):
      if xs is None:
          xs = [None] * length
      carry = init
      ys = []
      for x in xs:
          carry, y = f(carry, x)
          ys.append(y)
      return carry, np.stack(ys)
  ```
  上記のイメージを XLA で効率良くまとめたものが `jax.lax.scan`。PyTree 構造を保ったままステップ結果を自動でスタックしてくれる。

### 乱数キー管理と numpy 変換
- 乱数キーは `jax.random.split` で逐次更新し、純粋関数性を保つ。
- `simulate(...)` は numpy 入出力に対応し、`output_dtype` で制御できる。

### 64bit 精度
- 高次の評価器を使う場合はする `config.update("jax_enable_x64", True)` を推奨。

---

## 2. Heston モデルの JAX 実装

### 模型
\[
\begin{aligned}
 dS_t &= \mu S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^{(1)},\\
 dv_t &= \kappa(\theta - v_t)\,dt + \xi \sqrt{v_t}\,dW_t^{(2)},\\
 \mathbb{E}\,[dW_t^{(1)} dW_t^{(2)}] &= \rho\,dt.
\end{aligned}
\]

### ステップ関数と `lax.scan`
```python
import jax
import jax.numpy as jnp
from jax import lax, random

def simulate_heston(s0, v0, mu, kappa, theta, xi, rho, dt, n_steps, seed=0):
    key = random.PRNGKey(seed)
    sqrt_dt = jnp.sqrt(dt)
    chol = jnp.array([[1.0, 0.0], [rho, jnp.sqrt(1.0 - rho**2)]])

    def step(carry, _):
        S_t, v_t, key = carry
        key, subkey = random.split(key)
        z = random.normal(subkey, shape=(2,))
        dW1, dW2 = chol @ z
        dW1 *= sqrt_dt
        dW2 *= sqrt_dt

        v_sqrt = jnp.sqrt(jnp.maximum(v_t, 0.0))
        v_next = v_t + kappa * (theta - v_t) * dt + xi * v_sqrt * dW2
        v_next = jnp.maximum(v_next, 0.0)
        S_next = S_t + mu * S_t * dt + v_sqrt * S_t * dW1

        return (S_next, v_next, key), (S_next, v_next)

    carry0 = (jnp.array(s0), jnp.array(v0), key)
    _, paths = lax.scan(step, carry0, None, length=n_steps)
    S_path = jnp.concatenate([jnp.array([s0]), paths[0]])
    v_path = jnp.concatenate([jnp.array([v0]), paths[1]])
    return S_path, v_path
```
- 結果タプル `(S_next, v_next)` を return すると、`scan` が `(S_results, v_results)` を自動で積み上げる。
- 初期値を付け加えて最終的なパスを得る。

### パフォーマンスのポイント
- `jax.jit(simulate_heston)` すれば繰り返し使用時にコンパイルが使い回せる。
- パスを大量に生成するなら `jax.vmap` でバッチ化すると高速。
- 64bit が必要なら `config.update("jax_enable_x64", True)` をインポート前に設定。

---

### 参考
- JAX ドキュメント: https://jax.readthedocs.io/
- NumPyro ドキュメント: https://num.pyro.ai/

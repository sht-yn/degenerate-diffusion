# BlackJAX を使った NUTS サンプラー構築ガイド

このドキュメントでは、`src/degenerate_diffusion/estimation/parameter_estimator_new.py` に定義されている `build_b` 関数を例に、BlackJAX の NUTS サンプラーをどのように組み立てているかを解説します。

## 概要

`build_b` は以下のような関数を返します。

```python
(theta0: jax.Array, key: jax.Array, aux: Any) -> jax.Array
```

- `theta0`: サンプリング開始点（パラメータ初期値）
- `key`: `jax.random.PRNGKey`（乱数生成の元）
- `aux`: ログ確率計算で必要な補助データ（PyTree）

返り値は、ブラックボックス化された NUTS サンプリングから得た事後分布サンプルの平均です。関数全体は `jax.jit` で JIT コンパイルされ、JAX 互換の純関数として利用できます。

## BlackJAX カーネルの構築

```python
nuts = blackjax.nuts(logprob, step_size=step_size, inverse_mass_matrix=inv_mass)
state0 = nuts.init(theta0)
```

1. `logprob`: `theta -> log p(theta | aux)` を返す関数。`build_b` 内では、ユーザーが渡した `logprob_fn(theta, aux)` を `jnp.asarray` で包み、JAX の自動微分が扱える形に整えます。
2. `step_size`: NUTS のステップサイズ（リープフロッグ1ステップあたりの長さ）。
3. `inverse_mass_matrix`: ハミルトニアンモンテカルロにおける逆質量行列。未指定なら `theta0` と同形状の `1` を使い、各次元を同じスケールとして扱います。各パラメータのスケール調整や相関の取り込みが必要な場合は、ユーザーが適切な値を渡します。
4. `nuts.init(theta0)`: 初期状態を構築。ここで `theta0` を出発点とするハミルトニアン軌道計算の準備が整います。

## 1 ステップの更新 (`one_step`)

```python
def one_step(state, _):
    st, k_local = state
    k_local, k_use = jax.random.split(k_local)
    st_new, _info = nuts.step(k_use, st)
    return (st_new, k_local), st_new.position
```

- 乱数キーを `split` し、一度きりの乱数を抽出。
- `nuts.step` は次の NUTS 状態 (`st_new`) とメタ情報（サンプル採択率など）を返します。
- `st_new.position` が新しいパラメータサンプルです。

## ウォームアップ (`jax.lax.scan`)

```python
(state_warm_end, key_after), _ = jax.lax.scan(
    one_step, (state0, key), jnp.arange(num_warmup_i)
)
```

- ウォームアップ回数 (`num_warmup`) 分だけ `one_step` を繰り返し、
  サンプラーを事後分布の高密度領域へ移動させます。
- `scan` を使うことで、JAX がループを逐次処理するのではなく、
  計算グラフとして最適化・並列化できます。

## サンプリングとシンニング

```python
def sample_body(carry, _):
    st, k_local = carry

    def do_thin(_, c):
        st_cur, k_cur = c
        k_cur, k_use = jax.random.split(k_cur)
        st_new, _ = nuts.step(k_use, st_cur)
        return st_new, k_cur

    st_thin, k_thin = jax.lax.fori_loop(0, thin_i - 1, do_thin, (st, k_local))
    (st_new, k_new), sample = one_step((st_thin, k_thin), jnp.asarray(0))
    return (st_new, k_new), sample

(_, _), samples = jax.lax.scan(
    sample_body, (state_warm_end, key_after), jnp.arange(num_samples_i)
)
```

- `thin > 1` の場合、`fori_loop` で `thin-1` 回だけサンプラーを動かし、
  連続サンプル間の自己相関を下げます。
- `one_step` を呼んで実際の保存サンプルを取得。
- `scan` で `num_samples` 個のサンプル列を構築します。

## 返り値

```python
return jnp.mean(samples, axis=0)
```

- 得られたサンプルの算術平均を返します。必要に応じて、中央値や分散など別の統計量を計算するように変更できます。

## 使い方の例

```python
from jax import random
import jax.numpy as jnp

# ログ確率関数の例（多変量正規）
def logprob(theta, aux):
    mean, cov_inv = aux
    diff = theta - mean
    return -0.5 * diff.T @ cov_inv @ diff

key = random.key(0)
theta0 = jnp.zeros(3)
mean = jnp.array([1.0, 0.0, -1.0])
cov_inv = jnp.eye(3)
aux = (mean, cov_inv)

sampler = build_b(logprob, step_size=0.1, num_warmup=200, num_samples=1000)
theta_mean = sampler(theta0, key, aux)
```

- ここでは単純な正規分布を例にしましたが、`logprob` の中身を差し替えるだけで複雑なモデルにも適用できます。
- 実行環境が GPU/TPU の場合も `jax.jit` によって同じコードで高速に動作します。

## チューニングのヒント

- **`step_size`**: 大きすぎると受容率が下がり、小さすぎると探索効率が落ちます。`build_b` は固定値を受け取る設計ですが、外部で適応アルゴリズム（例: step size tuning）を行って結果を渡すこともできます。
- **`inverse_mass_matrix`**: 事前に最適化や過去のサンプルから推定した共分散情報を使うと、収束が大幅に改善することがあります。BlackJAX は対角行列と一般行列のどちらにも対応しています。
- **`thin`**: 自己相関が気になる場合のみ大きめに設定し、基本は `1`（間引きなし）が推奨です。
- **`num_warmup` と `num_samples`**: モデルの複雑さに応じて調整してください。ウォームアップが不十分だと収束せず、サンプル数が少ないと事後平均の推定が不安定になります。

---

`build_b` は BlackJAX の新しい API（>=1.0）に合わせて実装されており、Step実行時に追加の引数を渡す必要がありません。サンプラの挙動をカスタマイズしたい場合は、このガイドの各ステップを参考にコードを修正してください。

# 推定ユーティリティの利用ガイド（現行API）

このドキュメントは、現行の JAX ネイティブ推定 API に対応しています。

含まれる機能（抜粋）
- ニュートン上昇の1ステップ/収束ソルバ（境界投影つき）
- BlackJAX NUTS による事後更新の1ステップ（unary logprob）

```python
from degenerate_diffusion.estimation.parameter_estimator_new import (
    build_one_step_ascent,
    build_newton_ascent_solver,
    build_bayes_transition,
)
```

## M推定（最大化）

目的関数: `objective(theta, aux) -> scalar`（最大化）。

```python
import jax
import jax.numpy as jnp

objective = lambda th, aux: -0.5 * jnp.sum((th - aux["mu"]) ** 2)
bounds = [(-jnp.inf, jnp.inf)] * 3

step = build_one_step_ascent(objective, bounds, damping=0.5)
solve = build_newton_ascent_solver(objective, bounds, tol=1e-8, damping=1.0)

theta0 = jnp.zeros(3)
aux = {"mu": jnp.ones(3)}

theta1 = step(theta0, aux)
thetahat = solve(theta0, aux)
```

ポイント
- ヘッセ安定化は最大化のため `-eps * I` を使用
- AdamW フォールバックは最大化なので `-grad` を最適化
- 境界は `clip` により射影

## B推定（NUTS 一歩）

BlackJAX NUTS の1ステップ遷移をビルダーで構築します。logprob は **unary**（`theta` のみ）です。aux を使う場合は、事前に閉じてから渡します。

```python
import jax
import jax.numpy as jnp

def logprob(theta: jax.Array, aux) -> jax.Array:
    d = theta - aux["mean"]
    return -0.5 * (d @ (aux["inv_cov"] @ d))

aux_fixed = {"mean": jnp.array([1.0]), "inv_cov": jnp.eye(1)}
closed = lambda th: logprob(th, aux_fixed)

step = build_bayes_transition(closed, step_size=0.2, max_num_doublings=6)

key = jax.random.PRNGKey(0)
th0 = jnp.array([0.0])
th1, key = step(th0, key)
```

注意
- `num_integration_steps` は使わず、NUTS の木の深さは `max_num_doublings` を指定
- `inverse_mass_matrix` を省略すると `ones_like(theta)` がデフォルト
- 内部で API 差分に対応（古い BlackJAX でも動作）

## 交互推定（M/B）

`lax.scan`・`lax.fori_loop` で M/B を交互に呼び分けられます。スケジュールやブロック更新の詳細は設計ドキュメントを参照してください。

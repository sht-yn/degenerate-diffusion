# パラメータ推定ユーティリティの使い方

`degenerate_sim/estimation/parameter_estimator.py` では、勾配ベースの推定器を複数提供しています。ここでは `m_estimate` と `m_estimate_jax` の概要と利用方法をまとめます。

## 共通の前提

- 目的関数 `objective_function(theta)` は JAX の `jnp.ndarray` を受け取り、実数スカラーを返す必要があります。
- `search_bounds` は各パラメータに対する `(low, high)` のタプル列で、`None` を指定すると非制限境界になります。
- `initial_guess` は推定を開始する初期値ベクトルです。境界外の成分は自動的にクリップされます。

```python
from degenerate_sim.estimation.parameter_estimator import (
    m_estimate,
    m_estimate_jax,
)
```

## `m_estimate`

NumPy ベースの簡易勾配法です。小規模問題や試行錯誤に向いています。

```python
theta_hat = m_estimate(
    objective_function=my_objective,
    search_bounds=[(0.0, 1.0), (None, None)],
    initial_guess=[0.2, 0.0],
    learning_rate=1e-3,
    max_iters=2_000,
    tol=1e-6,
    log_interval=100,
)
```

- `learning_rate` は固定ステップ幅の勾配上昇です。
- `log_interval` を設定すると標準出力に進捗が表示されます。
- 返り値は `numpy.ndarray` です。

## `m_estimate_jax`

JAXopt の `ProjectedGradient` による推定で、JIT コンパイルや大規模計算と相性が良いアプローチです。JAXopt がインストールされている必要があります。

```python
theta_hat = m_estimate_jax(
    objective_function=my_objective,
    search_bounds=[(0.0, 1.0), (None, None)],
    initial_guess=[0.2, 0.0],
    learning_rate=1e-2,
    max_iters=1_000,
    tol=1e-7,
    log_interval=50,
)
```

- `learning_rate` (`stepsize`) は `ProjectedGradient` のステップ幅です。
- `log_interval` を設定すると `jax.debug.print` 経由でホストにログが出力され、JIT 後も利用できます。
- 返り値は `numpy.ndarray`。推定後に境界内へ再クリップされます。

## よくある注意点

- 目的関数の値が最大化対象であることを確認してください。最小化の場合は符号を反転させてください。
- 勾配計算が不安定な場合は、`search_bounds` の設定や初期値、ステップ幅を調整すると収束しやすくなります。
- JAXopt 版を使う際は `pip install jaxopt` などで事前にライブラリを導入してください。

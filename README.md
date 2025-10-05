# 退化拡散過程シミュレーションツールキット

このリポジトリは退化拡散過程（degenerate diffusion process）に対する擬似尤度推定の実験環境です。SymPy で記述した記号モデルを JAX へ変換し、高速なシミュレーションと NumPyro ベースの推定を行います。研究用ノートブックを主眼としていますが、`degenerate_sim/` 配下のモジュールは軽量なツールキットとしても利用できます。

## 特徴
- SymPy で定義したドリフト・拡散・観測方程式を `lambdify` で JAX 関数へ自動変換。
- `jax.lax.scan` を用いた GPU/CPU 互換のオイラー–丸山法シミュレータ。
- 擬似尤度評価器（`V1`／`V1'`／`V2`／`V3`）を JAX で実装し、自動微分やベクトル化を活用可能。
- JAX の勾配・ヘッシアンと NumPyro の NUTS を利用した推定ヘルパ（M 推定、ワンステップ、ベイズ推定）。
- SymPy 配列向けユーティリティ（`ImmutableDenseNDimArray` 用 `einsum` など）。

## ディレクトリ構成
- `degenerate_sim/`
  - `processes/degenerate_diffusion_process_jax.py`: 記号モデルと JAX シミュレータ。
  - `evaluation/likelihood_evaluator_jax.py`: 擬似尤度評価器を生成するファクトリ関数。
  - `estimation/parameter_estimator.py`: `m_estimate`、`one_step_estimate`、`bayes_estimate` を提供。
  - `utils/einsum_sympy.py`: SymPy 向け `einsum` 実装（インポート時に簡易セルフテスト実行）。
  - `__init__.py`: 主要クラス／関数の再エクスポート。
- `FNmodel.ipynb`, `jax_study.ipynb`: モデル構築・シミュレーション・推定のノート。
- `old/`: リファクタ以前の互換コード。歴史的ノートを再現するために残しています。
- `pyproject.toml`, `uv.lock`: [uv](https://github.com/astral-sh/uv) を利用する場合のメタデータ。

## 動作環境
- Python 3.11 で動作確認。
- 主な依存パッケージ: `jax`、`jaxlib`、`numpy`、`sympy`、`numpyro`、`matplotlib`（ノートブック用）。
- 高次（`k > 3`）の擬似尤度を扱う際は 64bit 浮動小数点を有効にすることを推奨します。
  ```python
  from jax import config
  config.update("jax_enable_x64", True)  # JAX をインポートする前に実行
  ```
  もしくは `JAX_ENABLE_X64=1` を環境変数として設定してください。

## セットアップ手順
### uv を使う場合
```bash
uv init .               # 初回のみ（pyproject が未整備なら）
uv add jax jaxlib numpy sympy numpyro matplotlib
uv sync
source .venv/bin/activate
```

### pip を使う場合
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install jax jaxlib numpy sympy numpyro matplotlib
```

## クイックスタート
```python
import sympy as sp
import jax.numpy as jnp
from degenerate_sim import DegenerateDiffusionProcess, LikelihoodEvaluator
from degenerate_sim.estimation import m_estimate, bayes_estimate

# 1. SymPy でモデルを定義
x0, y0 = sp.symbols("x0 y0")
x = sp.Array([x0])
y = sp.Array([y0])
theta_1 = sp.Array([sp.symbols("sigma")])
theta_2 = sp.Array([sp.symbols("kappa")])
theta_3 = sp.Array([sp.symbols("mu")])
A_expr = sp.Array([-theta_2[0] * x[0]])
B_expr = sp.Array([[theta_1[0]]])
H_expr = sp.Array([theta_3[0] * x[0]])
process = DegenerateDiffusionProcess(x, y, theta_1, theta_2, theta_3, A_expr, B_expr, H_expr)

# 2. シミュレーション
true_theta = (
    jnp.array([0.4], dtype=jnp.float64),
    jnp.array([1.0], dtype=jnp.float64),
    jnp.array([0.2], dtype=jnp.float64),
)
x_series, y_series = process.simulate(true_theta, t_max=100.0, h=0.1, burn_out=50.0, dt=1e-3)

# 3. 擬似尤度評価器を作成
likelihood = LikelihoodEvaluator(process)
l1 = likelihood.make_quasi_likelihood_l1_evaluator(x_series, y_series, h=0.1, k=3)

# 4a. M 推定（勾配上昇）
def objective(theta_vec: jnp.ndarray) -> float:
    return v1(theta_vec, true_theta[0], true_theta[1], true_theta[2])

m_result = m_estimate(
    objective_function=objective,
    search_bounds=[(0.1, 0.7)],
    initial_guess=jnp.array([0.3]),
)

# 4b. NumPyro (NUTS) によるベイズ推定
bayes_result = bayes_estimate(
    objective_function=objective,
    search_bounds=[(0.1, 0.7)],
    initial_guess=jnp.array([0.3]),
    num_warmup=500,
    num_samples=2000,
    rng_seed=0,
)
```

### `bayes_estimate` の補足
- 境界の情報から一様／指数／正規の事前分布を自動で選択します。`prior_log_pdf` で任意の事前項を追加できます。
- NumPyro の NUTS を使用するため、初回実行時は JIT コンパイルとウォームアップに時間がかかります。
- `k` を大きくすると数値不安定が生じやすいため、64bit 精度の有効化を推奨します。

## 推定 API まとめ
- `m_estimate(objective_function, search_bounds, initial_guess, *, learning_rate=1e-2, max_iters=1000, tol=1e-6)`
  - JAX の勾配で境界付き勾配上昇。各ステップで境界内にクリップ。
- `one_step_estimate(objective_function, search_bounds, initial_estimator)`
  - 勾配とヘッシアンを使ったニュートン 1 ステップ更新。
- `bayes_estimate(objective_function, search_bounds, initial_guess, *, prior_log_pdf=None, num_warmup=1000, num_samples=2000, num_chains=1, rng_seed=0)`
  - NumPyro の `MCMC(NUTS)` を呼び出し、事後サンプルの平均を返却。

## ユーティリティとノート
- `degenerate_sim.utils.einsum_sympy`: SymPy 配列向け `einsum` 実装。インポート時に自己テストが走ります。
- `FNmodel.ipynb`: Fowler–Nordheim 型モデルを題材にした例示ノート。
- `jax_study.ipynb`: JAX の挙動を検証したスケッチ。
- `old/`: 過去のモジュール構成。古いノートを開く用途で残存。

## 既知の課題 / TODO
- `einsum_sympy.py` がインポート時にデモ計算を実行するため、副作用をテストへ移したい。
- 擬似尤度クラスの警告出力が `print` に依存している。ロガーや例外クラスへの置換が課題。
- 一部ノートブックは `old/` 配下の互換モジュールを参照している。`degenerate_sim` への移行を進める。
- 高次擬似尤度（`k > 3`）は数値的に不安定なため、64bit 精度やパラメータスケーリングの検討が必要。

改善提案や機能追加があれば Issue / Pull Request で共有してください。

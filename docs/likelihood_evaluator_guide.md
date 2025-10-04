# LikelihoodEvaluator 利用ガイド

このドキュメントでは、`degenerate_sim` パッケージに含まれる `LikelihoodEvaluator` と関連コンポーネントの構造、利用手順、注意点をまとめます。2024 年リファクタ後の新しい設計に基づいています。

## 全体構成

`LikelihoodEvaluator` は以下 3 つのコンポーネントを束ねるファサードです。

- `SymbolicLikelihoodPreparer`
  - SymPy を使って行列 `C`, `V`, その逆行列や行列式などを導出し、JAX で実行可能な関数へ `lambdify` します。
  - 計算結果は `SymbolicPrecomputation` データクラス（後述）に格納されます。
- `InfinitesimalGenerator`
  - 無限小生成作用素 \(L\) とその正規化 \(L_0\) を扱います。
  - SymPy の式を再利用するためのキャッシュを持ち、`L`, `L_0`, `L_0_func` を外部に提供します。
- `QuasiLikelihoodEvaluator`
  - JAX 上で疑似尤度を計算する各種ファクトリ (`make_quasi_likelihood_v*_evaluator`) を提供します。
  - `SymbolicPrecomputation` を参照し、利用可能なコンポーネントを検査した上で evaluators を生成します。

### 補助クラス

- `SymbolicArtifact`
  - 記号式 (`expr`) と対応する JAX 関数 (`func`) をまとめたデータコンテナ。
- `SymbolicPrecomputation`
  - 前処理済みの記号式と関数をまとめた不変データクラス。
- 例外クラス
  - `SymbolicPreparationError`: 記号前処理で致命的エラーが発生した場合に送出。

## 使い方の流れ

1. **モデルの準備**
   - `DegenerateDiffusionProcess` など、`A_func`, `B_func`, `H_func` を JAX 互換で提供するモデルインスタンスを用意します。
   - `LikelihoodEvaluator` の初期化時にこれらの関数が存在しないと `AttributeError` となります。

2. **`LikelihoodEvaluator` を初期化**
   ```python
   likelihood = LikelihoodEvaluator(model)
   ```
   - 既に前処理結果がある場合は `precomputed=SymbolicPrecomputation(...)` を渡して再利用可能です。

3. **初期化エラーの確認**
   ```python
   from degenerate_sim.evaluation.likelihood_evaluator_jax import SymbolicPreparationError

   try:
       likelihood = LikelihoodEvaluator(model)
   except SymbolicPreparationError as exc:
       raise RuntimeError("前処理に失敗しました") from exc
   ```
   - C や V が特異な場合などは初期化時に `SymbolicPreparationError` が送出されるため、モデル式やパラメータを調整してください。

4. **疑似尤度の evaluator を生成**
   - `make_quasi_likelihood_v1_evaluator`, `make_quasi_likelihood_v1_prime_evaluator`, `make_quasi_likelihood_v2_evaluator`, `make_quasi_likelihood_v3_evaluator` を用途に応じて呼び出します。
  - 必須成分が欠けている場合は初期化時点で `SymbolicPreparationError` が送出されるため、モデル式やパラメータを調整してください。

5. **最適化ルーチンへ渡す**
   - evaluator は callable (JAX 関数) を返します。`parameter_estimator.m_estimate` や `newton_solve` などと組み合わせてパラメータ推定を実施します。

## 各 evaluator の依存関係

| evaluator | 目的 | 要求される主なコンポーネント |
|-----------|------|------------------------------|
| `v1` (`make_quasi_likelihood_v1_evaluator`) | フル疑似尤度 | `inv_S0_*` 系、`log_det_S0` |
| `v1_prime` (`make_quasi_likelihood_v1_prime_evaluator`) | 一部簡略化版 | `inv_C`, `log_det_C` |
| `v2` (`make_quasi_likelihood_v2_evaluator`) |  \(\theta_2\) 向け | `inv_C` |
| `v3` (`make_quasi_likelihood_v3_evaluator`) |  \(\theta_3\) 向け | `inv_V`, `partial_x_H^T V^{-1}` |

`v3` は特に `V` が特異な場合に初期化段階で失敗するため、`LikelihoodEvaluator` の生成時に `SymbolicPreparationError` を捕捉し、必要に応じてモデルの記号式を修正してください。

## `L` / `L_0` の利用

- 記号式の入力に対して \(L^k f\) や \(L_0^k f\) を計算できます。
- `L_0_func` は (x, y, θ₁, θ₂, θ₃) を引数に取る JAX 互換の関数を返すため、疑似尤度計算内部でも利用されています。
- 内部キャッシュは SymPy の式オブジェクトを不変化してキーにする方式へ変更されています。旧来の `srepr` ベースのキャッシュより頑健です。

## 例: `FNmodel.ipynb` の抜粋

```python
likelihood = LikelihoodEvaluator(FNmodel)

v1 = likelihood.make_quasi_likelihood_v1_evaluator(
    x_series=x_series, y_series=y_series, h=h, k=3
)

v3 = likelihood.make_quasi_likelihood_v3_evaluator(
    x_series=x_series, y_series=y_series, h=h, k=3
)
```

## 発生しうる例外

- `SymbolicPreparationError`
  - 記号計算の過程で逆行列が計算できないなど致命的エラーが発生した場合に送出。
  - モデルの定義（`B`, `H` など）を再確認してください。
- `ValueError`
  - `make_quasi_*` が時系列長チェックや `k` の非負チェックで利用。
- `IndexError`
  - `Dx_func`/`Dy_func` で要求される L₀ の次数が未生成の場合に明示的に発生させます。`k` と L₀ の生成範囲が一致するように注意してください。

## 旧バージョンとの違い

- 初期化時に記号計算が失敗した場合は `SymbolicPreparationError` を送出する設計に統一しました。
- 大量のロジックが 1 クラスに集約されていた構造から、役割ごとの専用クラスへ分割されています。
- キャッシュキーが `srepr` 文字列から、SymPy 式の不変化データへ変更されました。
- 初期化で記号計算に失敗した場合は例外で即座に伝播するよう統一し、欠損した成果物を残さない設計にしました。
- `DegenerateDiffusionProcess` 側でも `A`/`B`/`H` を `SymbolicArtifact` として保持し、疑似尤度パイプライン全体で同じ取り扱いになっています。

## 設計メモ
- SymPy の `derive_by_array` は通常の微分係数がそのまま式の前に掛かる形で返ってくるため、係数が増える箇所を想定したうえで後段の処理を組む。
- `SymbolicPreparationError` を捕捉することで、記号計算の失敗原因を早期に知ることができる。

- 組み込みの行列 `C_sym` や `C_func` は必ず構築される前提なので、ラッパークラスを設けず素のまま保持しています。
- 一方で逆行列や log det などは特異で計算できないケースがあるため `SymbolicArtifact` で値と状態を一緒に持たせ、欠損が判明した時点で例外を送出して原因を明示しています。
- `SymbolicPrecomputation` を `frozen=True` にしているのは、生成後に内部状態を差し替えず（再 lambdify せず）安全に参照共用するためです。更新が必要になった場合は再生成します。

## トラブルシューティング

- **`SymbolicPreparationError` が投げられる**
  - `C` や `V` の記号計算が失敗した場合、初期化時に `SymbolicPreparationError` が生じます。モデル側の式やパラメータを調整し、再度初期化してください。
- **疑似尤度が `nan` を返す**
  - JAX の計算中で非有界値が出た可能性があります。入力データのスケールや `theta` の初期値を確認してください。
- **`IndexError` が `Dx_func` から出る**
  - `make_quasi_*` に渡した `k` と L₀ 関数の確保数が一致しているか確認してください。標準提供のファクトリでは `k` の値に合わせて十分な L₀ を生成します。

## 参考情報

- 推定ルーチン: `degenerate_sim/estimation/parameter_estimator.py`
- シミュレーション例: `degenerate_sim/processes/degenerate_diffusion_process_jax.py`
- Notebook サンプル: `FNmodel.ipynb`

以上を参考に、必要に応じて Notebook やスクリプトに組み込んでください。

# Optax の最小ルールと本リポジトリでの使い方

このノートは Optax の「最小限の文法と設計ルール」を、手前味噌にならないレベルで短く整理します。最初に「Optax は黒箱ではなく組み立て式」「state と updates が肝」という要点を先に押さえ、そのうえで使い方と実装パターンを示します。

## 0. まず押さえる結論（黒箱ではない／update は1ステップ）
- Optax は SciPy の minimize のような“手法を選ぶだけの黒箱”ではありません。前処理や更新則などの「変換（transform）」を組み合わせ、あなたが最適化ループを組み立てます。
- `optimizer.update(grads, state, params)` は「その1回分の更新量（updates）」と「次の内部状態（state）」を返すだけです。複数回の更新をまとめて行うものではありません（複数回は fori_loop/scan/while_loop で回します）。

## 1. 用語（最初にここだけ）
- Transform（変換）: `optax.adamw(...)` や `optax.clip_by_global_norm(...)` が返す「最適化処理の部品」。`init / update` を持つ。
- Optimizer State（最適化器の状態）: モメンタムや二乗平均など、変換が内部で保持する統計の PyTree。`state = optimizer.init(params)` で作る。
- Updates（更新量）: `optimizer.update(...)` が返す、パラメタと同形の差分 PyTree。`optax.apply_updates(params, updates)` で適用（本質的に params + updates）。
- Params（パラメタ）: 学習対象（PyTree）。`init / update` の形状の基準。
- Chain（連結）: `optax.chain(t1, t2, ...)` で変換を左から順に適用（例: クリップ → AdamW）。

## 2. 基本モデル（init / update / apply）
- 変換（Transform）は次の3役割を持ちます。
  1) `init(params)` → Optimizer State を作る
  2) `update(grads, state, params)` → (updates, next_state) を返す
  3) `optax.apply_updates(params, updates)` → 新しい params を得る（要素ごとの加算）

これが Optax の根幹です。

## 3. Optimizer State を詳しく（AdamW を例に）
Optimizer State は、各 transform が「更新を適応的に決めるために必要な内部統計」を保持する PyTree です。Adam/AdamW の典型は以下のとおりです。

- m（一次モーメント）: 勾配の移動平均（モメンタム）
- v（二次モーメント）: 勾配二乗の移動平均
- count: 反復カウント（バイアス補正に使う）

最小化規約での更新の概念式:
```text
m_t = β1 * m_{t-1} + (1-β1) * g_t
v_t = β2 * v_{t-1} + (1-β2) * (g_t ⊙ g_t)
m̂_t = m_t / (1 - β1^t)
v̂_t = v_t / (1 - β2^t)
updates = - lr * m̂_t / (sqrt(v̂_t) + ε)
# AdamW は decoupled weight decay により、勾配とは独立に params を減衰
params_next = params + updates  -  lr * weight_decay * params
```
実際には、これらは `optimizer.update(grads, state, params)` の内部で計算され、戻り値の `updates` と次の `state` が返ってきます。

チェーン時の State:
- `optax.chain(clip, adamw)` のように複数変換を連結すると、各変換の state が束ねられた PyTree になります（クリッピングは無状態のことが多い）。
- パラメタの PyTree 形状や dtype を変えた場合は、`state = optimizer.init(new_params)` で再初期化するのが安全です。チェックポイントは `(params, state, config)` をまとめて保存・復元します。

## 4. chain で前処理→最適化の順に組む
- `optax.chain(t1, t2, t3, ...)` は、左から順に変換を適用します。
- 例: `optax.chain(optax.clip_by_global_norm(clip_norm), optax.adamw(...))`
  - grads に対してまず「グローバルノルムでクリップ」し、その後に「AdamW のモメンタム・重み減衰」を適用する、という順序を構成します。

ポイント: クリッピングなどの「勾配前処理」は AdamW より前に置くのが定石です。

### 補足（FAQ）: chain は optimizer を“書き換える”の？
- いいえ。`optax.chain` は「既存オプティマイザの `.update` を後から書き換える（モンキーパッチする）」のではありません。
- 複数の Transform を合成して「新しい GradientTransformation（= `init/update` のペア）」を作る関数です。
- 内部のイメージ（擬似コード）:

```python
def chain(*transforms):
  def init(params):
    return tuple(t.init(params) for t in transforms)

  def update(updates, state, params=None):
    # 最初の updates は grads（= 勾配）を想定
    new_states = []
    for t, s in zip(transforms, state):
      updates, s = t.update(updates, s, params)  # 左から順に前処理→最適化…
      new_states.append(s)
    return updates, tuple(new_states)

  return GradientTransformation(init=init, update=update)
```

- したがって「optimizer をカスタマイズ」というより、「1ステップの update 規則を部品の連結で“定義する”」イメージです。
- なお、`update` は常に“1ステップ分”だけ計算します。複数ステップ進めるには `fori_loop/scan/while_loop` などでループを回します（§6 を参照）。

## 5. Adam と AdamW の違い（decoupled weight decay）
- Adam: 通常の Adam は L2 正則化を勾配に項として足す形になりがちです。
- AdamW: 重み減衰（weight decay）を勾配から切り離して適用（decoupled）。これによりスケーリング挙動が安定しやすい。
- Optax の `adamw(learning_rate, weight_decay, ...)` はこの AdamW を返します。

## 6. 高速に複数ステップ回す最小例（JIT + fori_loop / scan）
複数ステップの最適化を JAX で高速に回すには、ループを `jax.jit` で丸ごとコンパイルします。固定回数なら `lax.fori_loop` か `lax.scan` が便利です。

### 4.1 fori_loop 版（履歴不要・固定回数）
```python
import jax, jax.numpy as jnp
import optax

opt = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)

def loss(p):
  return jnp.sum(p**2)  # 最小化の例（最大化なら -objective を loss に）

@jax.jit
def train_n_steps(params, state, n_steps: int):
  def body(i, carry):
    params, state = carry
    grads = jax.grad(loss)(params)          # 最大化なら grads = -jax.grad(objective)(params)
    updates, state = opt.update(grads, state, params)
    params = optax.apply_updates(params, updates)
    return (params, state)

  return jax.lax.fori_loop(0, n_steps, body, (params, state))

params = jnp.zeros((3,))
state = opt.init(params)
params, state = train_n_steps(params, state, n_steps=1000)
```

### 4.2 scan 版（履歴が必要・固定回数）
```python
import jax, jax.numpy as jnp
import optax

opt = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)

def loss(p):
  return jnp.sum(p**2)

@jax.jit
def train_scan(params, state, steps: int):
  def step_fn(carry, _):
    params, state = carry
    grads = jax.grad(loss)(params)
    updates, state = opt.update(grads, state, params)
    params = optax.apply_updates(params, updates)
    return (params, state), params  # 履歴として params を収集（不要なら None）

  (params, state), traj = jax.lax.scan(step_fn, (params, state), xs=None, length=steps)
  return params, state, traj  # traj.shape = (steps, *params.shape)

params = jnp.zeros((3,))
state = opt.init(params)
params, state, traj = train_scan(params, state, steps=1000)
```

注意
- 上の例は「最小化」規約です。目的が最大化なら `loss = lambda p: -objective(p)` とするか、`grads = -jax.grad(objective)(params)` として符号を反転してください。
- 反復回数が実行時に決まる（収束で停止）場合は `lax.while_loop` が適切です。

補足: `optax.apply_updates(params, updates)` は本質的に「要素ごとの加算」です（PyTree 同形の `params + updates`）。
updates には学習率・モメンタム・重み減衰・クリッピングなどの効果がすでに反映されているため、ユーザ側でさらに学習率を掛ける必要はありません。差分を確認したい場合は `updates` 自体をログすれば OK です。

## 7. clip_by_global_norm と併用（chain）
```python
opt = optax.chain(
    optax.clip_by_global_norm(clip_norm=1.0),
    optax.adamw(learning_rate=1e-3, weight_decay=1e-4),
)
state = opt.init(params)
updates, state = opt.update(grads, state, params)
params = optax.apply_updates(params, updates)
```
- `clip_by_global_norm` は勾配ベクトル全体の L2 ノルムを上限 `clip_norm` に収める前処理。
- その後に AdamW がモメンタムと減衰を加味した更新を計算します。

## 8. 本リポジトリでの運用（parameter_estimator_new.py）
- 背景: 「最大化」問題。ヘッセが負定でない場合は AdamW へフォールバック。
- 構成:
```python
adam = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
optimizer = optax.chain(optax.clip_by_global_norm(clip_norm), adam)
state0 = optimizer.init(theta0)
```
- 反復内の AdamW 分岐:
```python
# g: objective の勾配（上昇方向へ動きたい）
# Maximization のため、最小化規約の Optax には -g を渡すのが自然
updates, state_next = optimizer.update(-g, state, theta)
theta_next = optax.apply_updates(theta, updates)
```
- これにより「AdamW は最小化器」という規約を保ったまま、目的は最大化（上昇）に整合します。

補足:
- `optimizer.update(grads, state, params)` の第3引数 `params` は一部の変換（weight decay など）で必要です。常に渡しておくのが無難です。
- learning rate / decay / clip の値は問題に応じてチューニングします。Newton 分岐が安定なら AdamW 分岐は稀にしか使われない想定でも、数値的安全性のためクリップは有効です。

## 9. よくある落とし穴
- 最大化で `optimizer.update(g, ...)` としてしまい、実は「降下」している。
  - 対策: `optimizer.update(-g, ...)` にするか、loss を `-objective` にする。
- chain の順序ミス
  - 例: AdamW の後に clip を置くと、モメンタム計算後の更新がクリップされ、挙動が変わる。
- state の取り回し
  - `state = optimizer.init(params)` を忘れたり、PyTree 形状が合わないとエラー。パラメタの dtype 変更時にも注意。

## 10. 参考パターン（最大化問題のひな型）
```python
opt = optax.chain(
    optax.clip_by_global_norm(clip_norm),
    optax.adamw(learning_rate=lr, weight_decay=wd),
)

@jax.jit
def ascend_step(theta, aux, state):
    def obj(th):
        return jnp.asarray(objective_fn(th, aux))  # maximize
    g = jax.grad(obj)(theta)
    updates, state = opt.update(-g, state, theta)  # 最大化なので -g
    theta = optax.apply_updates(theta, updates)
    return theta, state
```

この最小ルールを押さえておけば、`parameter_estimator_new.py` の Optimizer 構築と while_loop 内での運用意図を読み解けるはずです。

---

## 付録A. 用語集（Glossary）
- Transform（変換）
  - `optax.adamw(...)` や `optax.clip_by_global_norm(...)` が返す「最適化処理の部品」。
  - `init/ update` を提供し、連結（`optax.chain`）してパイプラインを構築できる。
- Optimizer State（最適化器の状態）
  - モメンタム、二乗平均、重み減衰の内部統計などを保持する PyTree。
  - `state = optimizer.init(params)` で作成し、`updates, state = optimizer.update(...)` で毎ステップ更新。
- Updates（更新量の PyTree）
  - `update` が返す「パラメタと同形の差分」。`optax.apply_updates(params, updates)` で適用。
  - 変換の組み合わせ（clip→adamw など）を経た最終的な更新値。
- Params（パラメタ）
  - 学習対象の PyTree。`init/ update` 双方に形状が必要。
- Chain（連結）
  - `optax.chain(t1, t2, ...)` で変換を左から順に適用。前処理→最適化の順を守るのが定石。
- Learning rate（学習率）
  - 更新幅のスケーリング。スケジュール（`optax.cosine_decay_schedule` 等）と組み合わせ可。
- Weight decay（重み減衰）
  - パラメタに直接の減衰を与える正則化。AdamW は勾配から独立に適用（decoupled）。
- Clipping（クリッピング）
  - 勾配/更新のノルムを上限に収める前処理。`clip_by_global_norm` は全体 L2 ノルム基準。

---

## 付録B. 「update は1ステップだけ」の注意点（AdamWは反復法だが黒箱ではない）
- `optimizer.update(grads, state, params)` は、その呼び出し1回ぶんの「更新量（updates）」と「次の状態（state）」を返します。複数回ぶんをまとめて適用するものではありません。
- Adam/AdamW は反復法ですが、各反復での“1ステップの更新規則”を定めています。よって、`update` の1回呼び出し = 1ステップ更新の計算です。
- 複数ステップ進めたい場合は、ループ（`lax.fori_loop` / `lax.scan` / `lax.while_loop`）で複数回 `update → apply_updates` を回してください（§4 参照）。
- `optax.chain(...)` は「前処理→最適化」の合成を“1ステップの中で順に適用”しているだけで、複数回の反復がまとめて実行されるわけではありません。

---

## 付録C. Optimizer State の詳細式（参考）
Optimizer State は、各 transform が「更新を適応的に決めるために必要な内部統計」を保持する PyTree です。AdamW の場合、代表的には以下が含まれます。

- m（一次モーメント）: 勾配の移動平均（モメンタム）
- v（二次モーメント）: 勾配二乗の移動平均
- count: 反復カウント（バイアス補正に使う）

典型的な更新の流れ（最小化規約）:
```text
m_t = β1 * m_{t-1} + (1-β1) * g_t
v_t = β2 * v_{t-1} + (1-β2) * (g_t ⊙ g_t)
m̂_t = m_t / (1 - β1^t)
v̂_t = v_t / (1 - β2^t)
updates = - lr * m̂_t / (sqrt(v̂_t) + ε)   # 最小化の降下方向
# AdamW の decoupled weight decay は勾配とは独立に params に減衰を適用
params_next = params + updates  -  lr * weight_decay * params
```
Optax 実装では、これらの処理が `optimizer.update(grads, state, params)` の中で行われ、戻り値の `updates` と次の `state` が返されます。

チェーン時の State:
- `optax.chain(clip, adamw)` のように複数変換を連結すると、各変換の state が束ねられた PyTree になります。
- 例えば `clip_by_global_norm` は「無状態」のことが多く、一方で `adamw` は `m/v/count` 等を持ちます。

再初期化・リストアの指針:
- パラメタの PyTree 形状や dtype を変えた場合は、`state = opt.init(new_params)` で再初期化するのが安全です。
- チェックポイントを取る場合、`(params, state, opt_config)` をセットで保存・復元します。

JAX 的な不変性と best practices:
- state はイミュータブルに扱い、毎ステップ `updates, state = opt.update(...); params = apply_updates(...)` で新しい値を受け取ります。
- while_loop/scan 内の carry として `(params, state)` を回すと JIT で高速化できます（§4 参照）。
- 最大化では `grads` の符号を反転して渡す（`updates` 自体は“降下”規約で生成されるため）。

補足: Stateless な transform
- クリッピングや単純なスケーリングは state を持たないことがあります。この場合でも `chain` の中で他の変換の state と一緒に扱われます。

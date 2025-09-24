# uv と仮想環境のセットアップ手順メモ

## 背景
- プロジェクトは `/Users/yanoshouta/dev/simulation`（Python 3.11）。
- 以前 `/Users/yanoshouta/simulation/.venv` に旧環境が存在し、そこを使い続けていた。
- `uv` は "カレントプロジェクト直下の .venv" を前提にするため、環境がずれていると警告が出る。

## 1. プロジェクト直下に `.venv` を作成
```bash
cd /Users/yanoshouta/dev/simulation
uv sync            # .venv が作成され、依存関係がインストールされる
source .venv/bin/activate
```

※ Windows 用の `activate.bat` を macOS で実行する必要はない。`source .venv/bin/activate` で十分。

## 2. プロジェクトを "import 可能" にする
### なぜ `uv add .` や `uv pip install -e .` ではダメなのか
- `uv add .` は "プロジェクト自身 (`simulation`) を依存ライブラリとして追加" しようとするため、自己依存で弾かれる。
- `uv pip install -e .` は editable install を試みるが、ビルドで PyPI から `setuptools` / `wheel` を取得しようとし、ネットワーク制限で失敗。

### 解決策（`.pth` ファイル）
`site-packages` にパスを追加する `.pth` ファイルを置く。一度作成すればその `.venv` が生きている限り有効。
```bash
# プロジェクト直下で実行
echo "/Users/yanoshouta/dev/simulation" > .venv/lib/python3.11/site-packages/simulation.pth
```

これで `import degenerate_sim` がどこからでも通るようになる（Ruff の「絶対インポート推奨」にも対応）。

### 補足
- `.pth` は仮想環境を再作成したり削除したときに消えるので、`uv sync` で `.venv` を作り直した場合は再度作成する。
- `pyproject.toml` では以下を指定し、`degenerate_sim` のみをパッケージ対象にするよう調整済み。
  ```toml
  [tool.setuptools.packages.find]
  include = ["degenerate_sim*"]
  exclude = ["old*", "__pycache__"]
  
  [build-system]
  requires = ["setuptools>=65", "wheel"]
  build-backend = "setuptools.build_meta"
  ```

## 3. 依存関係の更新（参考）
- 依存を追加したい場合は `uv add <package>`（`<package>` は PyPI の名前）。
- 自分のプロジェクトをあえて依存として入れる必要はない。
- 仮想環境圧縮や削除は `rm -rf .venv` で行い、再構築は `uv sync`。

## 4. 結局どう使うか
```bash
cd /Users/yanoshouta/dev/simulation
uv sync                         # まだなら仮想環境を作る
source .venv/bin/activate
# ↑ 以降はこのシェルで作業

# 初回だけ .pth を作る
echo "/Users/yanoshouta/dev/simulation" > .venv/lib/python3.11/site-packages/simulation.pth

# 以降、ノートブックやスクリプトでは
python -c "import degenerate_sim; print(degenerate_sim.__file__)"
```

これで JAX コードを絶対インポートで利用でき、Ruff の警告（相対インポート）も回避できる。ネットワーク制限下でも編集内容が即時反映され、開発体験がシンプルになる。

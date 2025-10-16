# uv と仮想環境のセットアップ手順メモ

## 背景
- プロジェクトは `/Users/yanoshouta/dev/degenerate-diffusion`（Python 3.11）。
- 以前 `/Users/yanoshouta/degenerate-diffusion/.venv` に旧環境が存在し、そこを使い続けていた。
- `uv` は "カレントプロジェクト直下の .venv" を前提にするため、環境がずれていると警告が出る。

## 1. プロジェクト直下に `.venv` を作成
```bash
cd /Users/yanoshouta/dev/degenerate-diffusion
uv sync            # .venv が作成され、依存関係がインストールされる
source .venv/bin/activate
```

※ Windows 用の `activate.bat` を macOS で実行する必要はない。`source .venv/bin/activate` で十分。

## 2. プロジェクトを "import 可能" にする
### なぜ `uv add .` や `uv pip install -e .` を使わないのか
- `uv add .` は "プロジェクト自身 (`degenerate-diffusion`) を依存ライブラリとして追加" しようとするため、自己依存で弾かれる。
- `uv pip install -e .` は editable install を試みるが、ネットワーク制限環境では失敗することがある。

### 現状の解決策
`src` レイアウトと `pyproject.toml` の設定により、`uv sync` でローカルパッケージが自動的にインストールされる。このため追加の `.pth` ファイルは不要。

参考として `pyproject.toml` の指定は以下の通り。
```toml
[tool.setuptools.packages.find]
where = ["src"]
include = ["degenerate_diffusion*"]
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
cd /Users/yanoshouta/dev/degenerate-diffusion
uv sync                         # まだなら仮想環境を作る
source .venv/bin/activate
# ↑ 以降はこのシェルで作業

# 動作確認
python -c "import degenerate_diffusion; print(degenerate_diffusion.__file__)"
```

これで JAX コードを絶対インポートで利用でき、Ruff の警告（相対インポート）も回避できる。ネットワーク制限下でも編集内容が即時反映され、開発体験がシンプルになる。

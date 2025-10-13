# mutual_concept_loss

共有サブスペースX仮説を検証するためのミニマルな実験基盤です。Phase 0/1では、`uv`を用いたPython環境整備と、8×8グリッド上で動作する合成タスク生成器を実装しました。

## セットアップ
1. Python 3.11系を`uv`で用意します。
   ```bash
   uv python install 3.11
   ```
2. 仮想環境を作成します。
   ```bash
   uv venv .venv
   ```
3. 依存パッケージを同期します。
   ```bash
   uv pip sync pyproject.toml --extra dev
   ```
   - このリポジトリではネットワーク制限環境で生成したため`uv.lock`はプレースホルダです。実行環境で以下を実行してロックファイルを再生成してください。
     ```bash
     uv pip compile pyproject.toml -o uv.lock
     ```
4. 環境を有効化し、テストを実行します。
   ```bash
   source .venv/bin/activate
   uv run pytest
   ```

## ディレクトリ構成
```
src/mutual_concept_loss/
  ├── data/
  │   ├── __init__.py
  │   ├── generator.py
  │   ├── primitives.py
  │   ├── sampler.py
  │   └── transforms.py
  └── utils/
      └── seed.py
```
- `data/`配下に合成データ生成のためのプリミティブ、データセット、バッチサンプラを実装しています。
- `utils/seed.py`ではPython/NumPy/PyTorchのシード固定を一元化しています。
- `tests/`にはプリミティブ、データセット、サンプラ向けの単体テストを配置しました。

## 今後の予定
- Phase 2以降でモデル本体、損失関数、学習スクリプトを順次追加します。
- 生成タスクの統計やログ収集の整備を行います。


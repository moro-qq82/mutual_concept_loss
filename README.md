# mutual_concept_loss

共有サブスペースX仮説を検証するためのミニマルな実験基盤です。Phase 0/1では`uv`を用いたPython環境整備と合成タスク生成器を整備し、
Phase 2ではベースラインモデルと損失群（共有正則化・疎AE・補助ヘッド）を実装しました。Phase 3では学習ループとロギング、チェックポイント保存を備えたトレーナーを追加し、フェーズ4以降の評価整備に向けた基盤を揃えています。

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
  ├── losses/
  │   ├── __init__.py
  │   ├── manager.py
  │   ├── share.py
  │   ├── sparse_autoencoder.py
  │   └── task.py
  ├── models/
  │   ├── __init__.py
  │   ├── adapters.py
  │   ├── bottleneck.py
  │   ├── decoder.py
  │   ├── encoder.py
  │   ├── shared_autoencoder.py
  │   └── sparse_autoencoder.py
  ├── training/
  │   ├── __init__.py
  │   ├── evaluation.py
  │   ├── loop.py
  │   └── scheduler.py
  └── utils/
      └── seed.py
```
- `data/`配下に合成データ生成のためのプリミティブ、データセット、バッチサンプラを実装しています。
- `models/`配下では共有ボトルネック付きのベースラインモデルと疎AE、補助ヘッドを提供します。
- `losses/`配下ではタスク損失、共有正則化、疎AE損失、および重みスケジューラをまとめた`LossManager`を提供します。
- `training/`配下ではTensorBoard/CSVロギングとチェックポイント保存を備えた`Trainer`と学習率スケジューラを提供します。
- `utils/seed.py`ではPython/NumPy/PyTorchのシード固定を一元化しています。
- `tests/`にはPhase 2で追加されたモデル・損失の単体テストも含まれます。

## 今後の予定
- Phase 4でゼロショット評価とfew-shot適応スクリプトを整備します。
- 生成タスクの統計やログ収集の整備を行います。


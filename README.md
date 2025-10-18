# mutual_concept_loss

共有サブスペースX仮説を検証するためのミニマルな実験基盤です。Phase 0/1では`uv`を用いたPython環境整備と合成タスク生成器を整備し、
Phase 2ではベースラインモデルと損失群（共有正則化・疎AE・補助ヘッド）を実装しました。Phase 3では学習ループとロギング、チェックポイント保存を備えたトレーナーを追加し、Phase 4ではゼロショット評価とfew-shot適応のユーティリティ／スクリプトを整備しました。Phase 5ではCKA・Grassmann距離・疎コード統計を自動化する表現解析モジュールを追加しています。

## セットアップ
1. Python 3.11系の仮想環境を`uv`で用意します。
   ```bash
   uv venv --python 3.11
   ```

2. 依存パッケージを同期します。
      - 簡単（pyprojectの依存＋devを直接インストール）
     ```bash
     uv sync --extra dev
     ```
   - ロックファイルを明示的に再生成してから同期
     ```bash
     # 既存のプレースホルダ uv.lock がある場合は上書き
     uv pip compile pyproject.toml --extra dev -o uv.lock
     uv pip sync uv.lock
     ```

3. テストを実行します。
   ```bash
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
  ├── analysis/
  │   ├── __init__.py
  │   ├── cka.py
  │   ├── collection.py
  │   ├── grassmann.py
  │   └── sparse.py
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
- `analysis/`配下では表現解析ユーティリティを提供し、CKA/Grassmann距離や疎コード統計を計算できます。
- `utils/seed.py`ではPython/NumPy/PyTorchのシード固定を一元化しています。
- `tests/`にはPhase 2で追加されたモデル・損失の単体テストも含まれます。

## 評価とfew-shot適応
- `mutual_concept_loss.training.EvaluationDataConfig`と`build_zero_shot_dataloader`を利用することで、未見合成（例：合成長3）に対するゼロショット評価データローダを構築できます。
- `mutual_concept_loss.training.FewShotDataConfig`と`build_few_shot_loaders`は、サポート集合とクエリ集合を分けたfew-shotデータ分割を生成します。
- `mutual_concept_loss.training.run_few_shot_adaptation`は、ボトルネック表現直後に挿入される線形アダプタのみを学習対象とした微調整ループを提供します。
- CLIスクリプトとして`scripts/eval_zero_shot.py`および`scripts/finetune_few_shot.py`を追加しており、チェックポイントを指定することで再現可能な評価・適応を実行できます。

### 使用例
```bash
# ゼロショット評価（合成長3）
python scripts/eval_zero_shot.py checkpoints/best.pt --device cuda --num-samples 1024

# few-shot適応（サポート64例・クエリ256例）
python scripts/finetune_few_shot.py checkpoints/best.pt --device cuda --support-samples 64 --query-samples 256
```

## 表現解析
Phase 5で追加された`scripts/analyze_representation.py`により、以下の処理を一括して実行できます。

```bash
python scripts/analyze_representation.py checkpoints/best.pt \
  --device cuda \
  --num-samples 1024 \
  --heatmap-dir outputs/analysis/figures \
  --output outputs/analysis/summary.json
```

- `--feature-keys`で指定したモデル出力間の線形CKAを算出し、数値行列とヒートマップ（Matplotlib利用時）を生成します。
- `--grassmann-feature`はGrassmann距離の計算対象となる表現を指定し、合成長ごとにPCAを実施します。
- 疎コード統計は`summarize_sparse_codes`を用いて平均活性・稼働率・プリミティブ相関を出力します。
- `--evaluate`を指定すると同じ分割での評価指標も再計算し、JSONへ同梱します。

## 今後の予定
- Phase 6として解析レポートと図表の文書化、再現手順の整備を進めます。


# Phase 6 図表生成・文書化の実験手順

本書は、Phase 6で要求される図表生成および文書化プロセスを再現可能な形でまとめた手順書です。解析済みチェックポイントを前提に、ゼロショット評価・few-shot適応・表現解析の各スクリプトを用いた再現実験と成果物整理の流れを記載します。

## 1. 前提条件
- Python 3.11系の仮想環境を`uv`で構築し、`uv sync --extra dev`で依存関係を同期していること。
- Phase 3〜5で生成した学習済みチェックポイント（例: `checkpoints/best.pt`）が利用可能であること。
- 実験結果を保存するための作業ディレクトリ（例: `outputs/phase6/`）を用意しておくこと。

## 2. ディレクトリ準備
1. 作業用ディレクトリを作成します。
   ```bash
   mkdir -p outputs/phase6/{zero_shot,few_shot,analysis}
   ```
2. チェックポイントを`checkpoints/`直下に配置します。
3. 合成タスクを固定した分割として利用する場合は、以下のコマンドでPhase 2仕様のデータセットを生成し、`outputs/datasets/`などに保存しておきます（オンザフライ生成でも問題ありません）。
   ```bash
   uv run python scripts/generate_experiment_datasets.py outputs/datasets \
     --train-samples 50000 --val-samples 5000 --test-samples 5000 \
     --train-composition 1 2 --evaluation-composition 3 3
   ```

## 3. ゼロショット評価
未見合成長（既定では3）に対する性能を記録し、表形式で整理します。

1. 評価を実行します。
   ```bash
   uv run python scripts/eval_zero_shot.py checkpoints/best.pt \
     --device cuda \
     --num-samples 2048 \
     --batch-size 128 \
     --output outputs/phase6/zero_shot/metrics.json
   ```
   - GPUが利用できない場合は`--device cpu`に変更し、`--num-samples`を512程度まで減らして所要時間を調整してください。
2. `metrics.json`に含まれる指標（IoU、精度、損失など）を抽出し、`docs/phase6_結果サマリー.md`へ追記します。

## 4. Few-shot適応
共有サブスペースXの効果をfew-shotで検証するため、サポート/クエリ分割を固定した微調整を行います。

1. 適応スクリプトを実行します。
   ```bash
   uv run python scripts/finetune_few_shot.py checkpoints/best.pt \
     --device cuda \
     --support-samples 64 \
     --query-samples 256 \
     --max-steps 200 \
     --output outputs/phase6/few_shot/metrics.csv
   ```
   - CPU実行時は`--device cpu`に変更し、`--max-steps`を64程度まで短縮してください。
2. 実行後に生成される`metrics.csv`を集計し、サポートサンプル数と性能の推移を折れ線グラフとしてまとめます。Matplotlibが利用できる環境では以下のようなスクリプトを活用します。
   ```bash
   uv run python -m mutual_concept_loss.analysis.plot_few_shot \
     --input outputs/phase6/few_shot/metrics.csv \
     --output outputs/phase6/few_shot/curve.png
   ```
   - `plot_few_shot`モジュールはPhase 6で新規追加予定の可視化ユーティリティです。未実装の場合は、CSVを手動で解析してください。
3. 得られた指標とグラフを`docs/phase6_結果サマリー.md`へ追記します。

## 5. 表現解析
CKA、Grassmann距離、疎コード統計の算出と可視化をまとめます。

1. 解析コマンドを実行します。
   ```bash
   uv run python scripts/analyze_representation.py checkpoints/best.pt \
     --device cuda \
     --num-samples 1024 \
     --batch-size 128 \
     --heatmap-dir outputs/phase6/analysis/figures \
     --output outputs/phase6/analysis/summary.json \
     --evaluate
   ```
   - CPU環境の場合は`--device cpu`とし、`--num-samples`を256程度に抑えてください。
2. `summary.json`に記録されたCKA行列、Grassmann距離、疎コード統計を読み取り、主要な所見（例: 高CKAを示す層、共有度合いの変化、アクティベーションの疎性）を文章化します。
3. `--heatmap-dir`で出力された画像ファイル（CKAヒートマップ、Grassmann距離マトリクスなど）を`docs/phase6_結果サマリー.md`へ引用し、図版として添付します。

## 6. 結果サマリー文書の更新
1. `docs/phase6_結果サマリー.md`を作成または更新し、以下の要素を含めます。
   - 実験条件の一覧（チェックポイント情報、データ分割、主要ハイパーパラメータ）。
   - ゼロショット評価の結果表。
   - Few-shot適応の性能曲線と解釈。
   - 表現解析の要約と図表。
2. READMEの「今後の予定」セクションを更新し、Phase 6の進捗と成果物リンクを追記します。

## 7. 再現性チェックリスト
- [ ] 乱数シードが`config.yaml`等に記録されている。
- [ ] `outputs/phase6/`配下の成果物が日付付きサブディレクトリに整理されている。
- [ ] 実験ログを`docs/実験ログ.md`に追記済み。
- [ ] 解析で用いたスクリプトのバージョン（コミットハッシュ）が記録されている。

## 8. トラブルシューティング
- **チェックポイントが存在しない場合**：Phase 3/4の学習スクリプトで再学習を行い、最新の`best.pt`を生成してください。
- **Matplotlibがインストールされていない場合**：`uv sync --extra viz`で追加依存を導入するか、数値行列のみで解析を進めてください。
- **CUDAが利用できない場合**：全スクリプトを`--device cpu`で実行し、サンプル数やステップ数を縮小して時間を調整してください。

以上の手順を順守することで、Phase 6の図表生成および文書化タスクを一貫して再現できます。

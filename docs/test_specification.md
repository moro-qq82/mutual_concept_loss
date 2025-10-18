# テスト仕様

本ドキュメントは`tests`ディレクトリ配下の各テストファイルが検証している主なふるまいを整理したものです。

## tests/analysis/test_cka.py
- `linear_cka`が同一表現に対して1に近い類似度を返すことを確認。
- `pairwise_linear_cka`が辞書入力から対称行列とラベル一覧を返すことを確認。
- サンプル数が一致しない場合に`linear_cka`が`ValueError`を送出することを確認。

## tests/analysis/test_collection.py
- `collect_representations`が基本的なデータ収集を行い、各特徴量・ラベルが全サンプル分揃うことを確認。
- バッチ数制限と`flatten=False`指定時に元のテンソル形状を保持したまま収集が止まること、推論後に学習モードへ戻ることを確認。
- モデル出力に指定キーがない場合に`collect_representations`が`KeyError`を送出することを確認。
- `RepresentationCollection.to`が各テンソルを指定デバイスへコピーすることを確認。
- `group_by_label`がラベルごとにテンソルを分割できることを確認。

## tests/analysis/test_grassmann.py
- `compute_principal_components`が直交基底を返すことを確認。
- 同一基底間の`grassmann_distance`が0になることを確認。
- `pairwise_grassmann_distances`が対角が0の距離行列を返すことを確認。

## tests/analysis/test_sparse.py
- `summarize_sparse_codes`が平均活性・活性率・相関行列の形状を正しく返すことを確認。

## tests/data/test_generator.py
- `SyntheticTaskDataset`が入力・出力グリッドとラベルを妥当な形状で提供すること、画素がワンホットであることを確認。
- キャッシュ経由のサンプルが複製（クローン）として返ることを確認。

## tests/data/test_primitives.py
- 回転操作がワンホット表現を維持することを確認。
- 色交換操作が指定チャネルを入れ替えることを確認。
- 塗りつぶし操作が最大成分を別色に置き換えることを確認。
- シフト操作がグリッド形状を保つことを確認。

## tests/data/test_sampler.py
- `GroupedBatchSampler`が同一ラベルのサンプルを同じバッチにまとめることを確認。
- サンプル総数から算出されるバッチ数が`__len__`の戻り値と一致することを確認。

## tests/data/test_transforms.py
- `indices_to_one_hot`と`one_hot_to_indices`の往復変換が正しく行われることを確認。
- 入力次元やクラス数が不正な場合に`indices_to_one_hot`が例外を送出することを確認。
- `one_hot_to_indices`が非二値テンソルを拒否することを確認。
- `ensure_one_hot`が最大チャネルを選択してワンホット化することを確認。
- `clamp_coordinates`が座標をグリッド境界内に収めることを確認。

## tests/losses/test_losses.py
- `TaskLoss`がクロスエントロピー損失とIoUメトリクスを返すことを確認。
- `SharedSubspaceLoss`がスカラー損失とメトリクスを返すことを確認。
- `SparseAutoencoderLoss`が再構成損失とスパース項を組み合わせることを確認。
- `LossManager`が各損失をスケジューラ付きで合計し、メトリクスを統合することを確認。

## tests/models/test_adapters.py
- `LinearAdapter`が残差付き変換を行い、指定アクティベーションを使用することを確認。
- 入力形状が不正な場合に`LinearAdapter`が例外を送出することを確認。
- コンフィグでゼロ以下の次元を指定した場合に初期化が失敗することを確認。

## tests/models/test_shared_autoencoder.py
- 共有オートエンコーダの各出力が期待する形状であることを確認。
- アダプタの取り付け・取り外しとパラメータ公開フローが動作することを確認。

## tests/training/test_few_shot.py
- 数ショット適応ループがCPU上で少ステップ完走し、履歴・ベスト指標を記録することを確認。

## tests/training/test_loop.py
- `Trainer.fit`が小規模設定で学習と評価を実行し、チェックポイントを保存することを確認。

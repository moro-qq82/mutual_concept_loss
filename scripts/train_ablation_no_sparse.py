"""疎AE損失を無効化したアブレーション設定で学習するスクリプト。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mutual_concept_loss.training.cli import add_common_run_arguments, run_training


def parse_args() -> argparse.Namespace:
    """疎AEアブレーションの引数を解析する。"""

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_run_arguments(parser, default_output_dir=Path("outputs/ablation_no_sparse"))
    return parser.parse_args()


def main() -> None:
    """疎AE無しの構成で学習ループを実行する。"""

    args = parse_args()
    metrics = run_training(args, alpha=1.0, beta=1.0, gamma=0.0)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


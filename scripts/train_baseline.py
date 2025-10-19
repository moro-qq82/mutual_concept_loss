"""タスク損失のみを学習する比較用ベースラインスクリプト。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mutual_concept_loss.training.cli import add_common_run_arguments, run_training


def parse_args() -> argparse.Namespace:
    """ベースライン学習の引数を解析する。"""

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_run_arguments(parser, default_output_dir=Path("outputs/baseline_task_only"))
    return parser.parse_args()


def main() -> None:
    """タスク損失のみの構成で学習ループを実行する。"""

    args = parse_args()
    metrics = run_training(args, alpha=1.0, beta=0.0, gamma=0.0)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


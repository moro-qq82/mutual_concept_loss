"""共有サブスペース正則化を無効化したアブレーション学習スクリプト。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mutual_concept_loss.training.cli import add_common_run_arguments, run_training


def parse_args() -> argparse.Namespace:
    """共有サブスペースアブレーションの引数を解析する。"""

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_run_arguments(parser, default_output_dir=Path("outputs/ablation_no_shared"))
    return parser.parse_args()


def main() -> None:
    """共有サブスペース無しの構成で学習ループを実行する。"""

    args = parse_args()
    metrics = run_training(args, alpha=1.0, beta=0.0, gamma=1.0)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


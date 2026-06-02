from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config
from ranking import run_ranking_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run listwise reranking.")
    parser.add_argument("--config", required=True, help="Path to a YAML or JSON config.")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of dataset samples to rank.")
    parser.add_argument("--permutations", type=int, default=None, help="Number of permutations per ranked sample.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.num_samples is not None:
        cfg.ranking_num_samples = args.num_samples
    if args.permutations is not None:
        cfg.eval_num_permutations = args.permutations
    records = run_ranking_pipeline(cfg)
    print(f"Wrote {len(records)} ranked-list records.")


if __name__ == "__main__":
    main()

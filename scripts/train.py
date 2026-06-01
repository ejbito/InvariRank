from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invarirank.config import load_config
from invarirank.training import run_training_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an InvariRank/LFT reranker.")
    parser.add_argument("--config", required=True, help="Path to a YAML or JSON config.")
    args = parser.parse_args()
    result = run_training_pipeline(load_config(args.config))
    print(result)


if __name__ == "__main__":
    main()

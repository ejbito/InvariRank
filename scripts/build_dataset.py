from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config
from datasets import build_dataset_splits, write_dataset_splits
from datasets.utils import cfg_get


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/val/test JSONL datasets.")
    parser.add_argument("--config", required=True, help="Path to a YAML or JSON dataset config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train, val, test = build_dataset_splits(cfg)
    output_dir = cfg_get(cfg, "paths.output_dir", cfg_get(cfg, "output_dir", None))
    if not output_dir:
        raise ValueError("Config must define paths.output_dir or output_dir.")
    write_dataset_splits(train, val, test, output_dir)


if __name__ == "__main__":
    main()

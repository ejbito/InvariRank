from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.data.datasets import prepare_amazon, prepare_movielens
from experiments.scripts.common import load_dataset_settings, processed_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a configured dataset for retrieval experiments.")
    parser.add_argument("--config", default="experiments/configs/datasets.yaml")
    parser.add_argument("--dataset", default="movielens")
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--reviews-file", default=None)
    parser.add_argument("--metadata-file", default=None)
    parser.add_argument("--min-rating", type=float, default=None)
    parser.add_argument("--min-user-interactions", type=int, default=None)
    parser.add_argument("--split-strategy", choices=["temporal_ratio", "leave_one_out"], default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dataset_settings(args.config, args.dataset)
    dataset_name = config.get("name")
    output_dir = processed_dir(config, args.processed_dir)
    common_kwargs = {
        "raw_dir": args.raw_dir or config["raw_dir"],
        "processed_dir": output_dir,
        "min_rating": args.min_rating if args.min_rating is not None else config["min_rating"],
        "min_user_interactions": (
            args.min_user_interactions
            if args.min_user_interactions is not None
            else config["min_user_interactions"]
        ),
        "show_progress": not args.no_progress,
        "split_strategy": args.split_strategy or config.get("split", "temporal_ratio"),
        "train_ratio": args.train_ratio if args.train_ratio is not None else float(config.get("train_ratio", 0.7)),
        "val_ratio": args.val_ratio if args.val_ratio is not None else float(config.get("val_ratio", 0.1)),
    }

    if dataset_name == "movielens":
        stats = prepare_movielens(**common_kwargs)
    elif dataset_name == "amazon":
        stats = prepare_amazon(
            **common_kwargs,
            reviews_file=args.reviews_file or config.get("reviews_file"),
            metadata_file=args.metadata_file or config.get("metadata_file"),
        )
    else:
        raise ValueError(
            f"Unsupported dataset name '{dataset_name}' for configured dataset '{args.dataset}'."
        )

    print(f"Prepared {args.dataset} at {output_dir}")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

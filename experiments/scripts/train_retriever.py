from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.data.interactions import load_interactions
from experiments.retrieval.registry import get_retriever_class
from experiments.scripts.common import load_dataset_settings, processed_dir
from experiments.utils.io import ensure_dir, read_yaml, write_json
from experiments.utils.progress import progress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a retrieval model.")
    parser.add_argument("--dataset", default="movielens")
    parser.add_argument("--dataset-config", default="experiments/configs/datasets.yaml")
    parser.add_argument("--retriever-config", default="experiments/configs/retrievers.yaml")
    parser.add_argument("--retriever", required=True)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--artifact-dir", default="artifacts/retrievers")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = load_dataset_settings(args.dataset_config, args.dataset)
    retriever_config = read_yaml(args.retriever_config)["retrievers"]
    if args.retriever not in retriever_config:
        raise ValueError(f"No config found for retriever '{args.retriever}'.")

    state = {}
    steps = ["load train split", "fit retriever", "save artifact"]
    for step in progress(
        steps,
        desc=f"Training {args.retriever}",
        total=len(steps),
        enabled=not args.no_progress,
    ):
        if step == "load train split":
            data_dir = processed_dir(dataset_config, args.processed_dir)
            train_path = data_dir / "retriever_train.csv"
            if not train_path.exists():
                train_path = data_dir / "train.csv"
            train = load_interactions(train_path)
            min_rating = float(dataset_config.get("min_rating", 4.0))
            state["train"] = train[train["rating"] >= min_rating].copy()
        elif step == "fit retriever":
            retriever_cls = get_retriever_class(args.retriever)
            params = dict(retriever_config[args.retriever].get("params", {}))
            params.setdefault("show_progress", not args.no_progress)
            state["retriever"] = retriever_cls(**params).fit(state["train"])
        elif step == "save artifact":
            output_dir = ensure_dir(Path(args.artifact_dir) / args.dataset / args.retriever)
            state["retriever"].save(output_dir)
            write_json(state["retriever"].training_stats_, output_dir / "training_stats.json")
    print(f"Saved {args.retriever} retriever to {output_dir}")
    print(f"Saved training stats to {output_dir / 'training_stats.json'}")
    loss_history = state["retriever"].training_stats_.get("loss_history", [])
    if loss_history:
        first_loss = loss_history[0]["loss"]
        final_loss = loss_history[-1]["loss"]
        print(f"loss: {first_loss:.6f} -> {final_loss:.6f}")


if __name__ == "__main__":
    main()

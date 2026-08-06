from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.data.interactions import load_interactions
from experiments.evaluation.metrics import evaluate_at_k
from experiments.evaluation.sampling import select_user_ids
from experiments.scripts.common import (
    ground_truth_from_interactions,
    extend_retriever_seen_items,
    load_dataset_settings,
    load_trained_retriever,
    print_metrics,
    processed_dir,
    retrieval_metrics_path,
)
from experiments.utils.io import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained retrieval model.")
    parser.add_argument("--dataset", default="movielens")
    parser.add_argument("--dataset-config", default="experiments/configs/datasets.yaml")
    parser.add_argument("--retriever", required=True)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--artifact-dir", default="artifacts/retrievers")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--sample-users", action="store_true")
    parser.add_argument("--user-sample-seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = load_dataset_settings(args.dataset_config, args.dataset)
    data_dir = processed_dir(dataset_config, args.processed_dir)
    split_path = data_dir / ("train_queries.csv" if args.split == "train" else f"{args.split}.csv")
    eval_interactions = load_interactions(split_path)
    eval_interactions = eval_interactions[
        eval_interactions["rating"] >= float(dataset_config.get("min_rating", 4.0))
    ].copy()
    all_user_ids = sorted(int(user_id) for user_id in eval_interactions["user_id"].unique())
    user_ids = select_user_ids(
        all_user_ids,
        max_users=args.max_users,
        sample=args.sample_users,
        seed=args.user_sample_seed,
    )

    retriever = load_trained_retriever(
        args.retriever,
        dataset=args.dataset,
        artifact_dir=args.artifact_dir,
        show_progress=not args.no_progress,
    )
    extend_retriever_seen_items(retriever, data_dir, args.split)
    recommendations = retriever.recommend(user_ids, k=args.k, exclude_seen=True)

    ground_truth = ground_truth_from_interactions(eval_interactions)
    selected_ground_truth = {user_id: ground_truth[user_id] for user_id in user_ids if user_id in ground_truth}
    metrics = evaluate_at_k(recommendations, selected_ground_truth, args.k)
    payload = {
        "stage": "retrieval",
        "dataset": args.dataset,
        "retriever": args.retriever,
        "split": args.split,
        "k": args.k,
        "num_users": len(user_ids),
        **metrics,
    }
    print(f"num_users: {len(user_ids)}")
    print_metrics(metrics)
    output = Path(args.output) if args.output else retrieval_metrics_path(args.dataset, args.retriever, args.split, args.k)
    ensure_dir(output.parent)
    write_json(payload, output)
    print(f"Saved retrieval metrics to {output}")


if __name__ == "__main__":
    main()

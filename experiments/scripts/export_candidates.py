from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.data.candidates import format_llm_candidates, load_item_metadata
from experiments.data.interactions import load_interactions
from experiments.evaluation.sampling import order_user_ids, select_user_ids
from experiments.scripts.common import (
    candidate_output_path,
    extend_retriever_seen_items,
    ground_truth_from_interactions,
    load_dataset_settings,
    load_trained_retriever,
    processed_dir,
)
from experiments.utils.io import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ranked retrieval candidates with metadata.")
    parser.add_argument("--dataset", default="movielens")
    parser.add_argument("--dataset-config", default="experiments/configs/datasets.yaml")
    parser.add_argument("--retriever", required=True)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--artifact-dir", default="artifacts/retrievers")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--sample-users", action="store_true")
    parser.add_argument("--user-sample-seed", type=int, default=42)
    parser.add_argument("--candidate-batch-size", type=int, default=1000)
    parser.add_argument("--output", default=None)
    ground_truth_group = parser.add_mutually_exclusive_group()
    ground_truth_group.add_argument(
        "--require-ground-truth-in-candidates",
        dest="require_ground_truth_in_candidates",
        action="store_true",
        default=True,
        help="Export only users whose candidate list contains at least one held-out ground-truth item.",
    )
    ground_truth_group.add_argument(
        "--allow-missing-ground-truth",
        dest="require_ground_truth_in_candidates",
        action="store_false",
        help="Include users even when retrieval did not place ground truth in the candidate list.",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _ground_truth_recommendations(
    retriever,
    ordered_user_ids: list[int],
    ground_truth: dict[int, list[int]],
    k: int,
    max_users: int | None,
    batch_size: int,
) -> tuple[dict[int, list[int]], list[int], int]:
    if batch_size < 1:
        raise ValueError("--candidate-batch-size must be at least 1.")

    kept_recommendations = {}
    scanned_users = 0
    target_users = max_users or len(ordered_user_ids)

    for start in range(0, len(ordered_user_ids), batch_size):
        batch_user_ids = ordered_user_ids[start : start + batch_size]
        scanned_users += len(batch_user_ids)
        batch_recommendations = retriever.recommend(batch_user_ids, k=k, exclude_seen=True)
        for user_id in batch_user_ids:
            candidate_items = set(int(item_id) for item_id in batch_recommendations.get(user_id, []))
            truth_items = set(int(item_id) for item_id in ground_truth.get(user_id, []))
            if candidate_items & truth_items:
                kept_recommendations[user_id] = batch_recommendations[user_id]
                if len(kept_recommendations) >= target_users:
                    selected_user_ids = list(kept_recommendations)
                    return kept_recommendations, selected_user_ids, scanned_users

    selected_user_ids = list(kept_recommendations)
    return kept_recommendations, selected_user_ids, scanned_users


def main() -> None:
    export_candidates(parse_args())


def export_candidates(args: argparse.Namespace) -> Path:
    dataset_config = load_dataset_settings(args.dataset_config, args.dataset)
    data_dir = processed_dir(dataset_config, args.processed_dir)
    split_path = data_dir / ("train_queries.csv" if args.split == "train" else f"{args.split}.csv")
    eval_interactions = load_interactions(split_path)
    eval_interactions = eval_interactions[
        eval_interactions["rating"] >= float(dataset_config.get("min_rating", 4.0))
    ].copy()
    all_user_ids = sorted(int(user_id) for user_id in eval_interactions["user_id"].unique())
    ground_truth = ground_truth_from_interactions(eval_interactions)

    retriever = load_trained_retriever(
        args.retriever,
        dataset=args.dataset,
        artifact_dir=args.artifact_dir,
        show_progress=not args.no_progress,
    )
    extend_retriever_seen_items(retriever, data_dir, args.split)
    if args.require_ground_truth_in_candidates:
        ordered_user_ids = order_user_ids(
            all_user_ids,
            sample=args.sample_users,
            seed=args.user_sample_seed,
        )
        recommendations, user_ids, scanned_users = _ground_truth_recommendations(
            retriever=retriever,
            ordered_user_ids=ordered_user_ids,
            ground_truth=ground_truth,
            k=args.k,
            max_users=args.max_users,
            batch_size=args.candidate_batch_size,
        )
    else:
        user_ids = select_user_ids(
            all_user_ids,
            max_users=args.max_users,
            sample=args.sample_users,
            seed=args.user_sample_seed,
        )
        scanned_users = len(user_ids)
        recommendations = retriever.recommend(user_ids, k=args.k, exclude_seen=True)

    user_id_set = set(user_ids)

    payload = format_llm_candidates(
        recommendations=recommendations,
        item_metadata=load_item_metadata(data_dir, show_progress=not args.no_progress),
        ground_truth={
            user_id: item_ids
            for user_id, item_ids in ground_truth.items()
            if user_id in user_id_set
        },
        retriever_name=args.retriever,
        split=args.split,
        require_ground_truth_in_candidates=args.require_ground_truth_in_candidates,
        show_progress=not args.no_progress,
    )
    payload["user_selection"] = {
        "num_users_available": len(all_user_ids),
        "num_users_selected": len(user_ids),
        "num_users_scanned": scanned_users,
        "max_users": args.max_users,
        "sample_users": args.sample_users,
        "user_sample_seed": args.user_sample_seed,
        "candidate_batch_size": args.candidate_batch_size,
        "selection_mode": (
            "ground_truth_in_candidates" if args.require_ground_truth_in_candidates else "selected_then_optional_filter"
        ),
    }

    output = Path(args.output) if args.output else candidate_output_path(
        args.dataset,
        args.retriever,
        args.split,
        args.k,
        max_users=args.max_users,
        selected_users=len(user_ids),
        require_ground_truth=args.require_ground_truth_in_candidates,
    )
    ensure_dir(output.parent)
    write_json(payload, output, sort_keys=False)
    print(f"Saved candidates to {output}")
    return output


if __name__ == "__main__":
    main()

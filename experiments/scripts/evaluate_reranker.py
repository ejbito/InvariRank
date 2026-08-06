from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.evaluation.metrics import evaluate_at_k, parser_quality_metrics, permutation_robustness_metrics
from experiments.evaluation.preference import preference_consistency_metrics
from experiments.scripts.common import print_metrics, reranking_metrics_path
from experiments.utils.io import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate reranked candidate lists.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", default=None)
    parser.add_argument("--include-robustness-values", action="store_true")
    parser.add_argument("--preference-buckets", type=int, default=3)
    parser.add_argument("--minimum-bucket-observations", type=int, default=1)
    parser.add_argument("--full-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_reranking_file(
        args.input,
        k=args.k,
        output=args.output,
        include_robustness_values=args.include_robustness_values,
        preference_buckets=args.preference_buckets,
        minimum_bucket_observations=args.minimum_bucket_observations,
        full_output=args.full_output,
    )
    print_metrics(metrics)


def evaluate_reranking_file(
    input_path: str | Path,
    k: int = 10,
    output: str | Path | None = None,
    include_robustness_values: bool = False,
    preference_buckets: int = 3,
    minimum_bucket_observations: int = 1,
    full_output: bool = False,
) -> dict:
    payload = read_json(input_path)
    recommendations, ground_truth = _effectiveness_inputs(payload["users"])

    base_metrics = {
        "stage": "reranking",
        "dataset": payload.get("dataset"),
        "source_retriever": payload.get("source_retriever"),
        "reranker": payload.get("reranker"),
        "reranking_mode": payload.get("reranking_mode"),
        "model": payload.get("model"),
        "architecture": payload.get("architecture"),
        "num_users": len(payload["users"]),
        "num_effectiveness_rows": len(recommendations),
        "k": k,
    }
    effectiveness = evaluate_at_k(recommendations, ground_truth, k)
    parser_metrics = parser_quality_metrics(payload["users"])
    has_permutations = any("permutations" in user_record for user_record in payload["users"].values())
    preference_metrics = {}
    robustness_metrics = {}
    if has_permutations:
        preference_metrics = preference_consistency_metrics(
            payload["users"],
            num_buckets=preference_buckets,
            minimum_bucket_observations=minimum_bucket_observations,
        )
        robustness_metrics = permutation_robustness_metrics(
            payload["users"],
            k=k,
            include_values=include_robustness_values,
        )

    metrics = dict(base_metrics)
    metrics.update(effectiveness)
    if full_output:
        metrics.update(parser_metrics)
        metrics.update(preference_metrics)
        metrics.update(robustness_metrics)
    else:
        metrics.update(_compact_parser_metrics(parser_metrics))
        metrics.update(_compact_preference_metrics(preference_metrics))
        metrics.update(_compact_robustness_metrics(robustness_metrics, k))

    output_path = Path(output) if output else reranking_metrics_path(
        str(payload.get("dataset", "unknown_dataset")),
        str(payload.get("source_retriever", "unknown_retriever")),
        Path(str(payload.get("source_candidates", "candidates.json"))).stem,
        Path(input_path).stem,
        k,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(metrics, output_path)
    print(f"Saved reranking metrics to {output_path}")
    return metrics


def _effectiveness_inputs(
    users: dict[str, Any],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    recommendations = {}
    ground_truth = {}
    for user_key, user_record in users.items():
        relevant = [str(item_id) for item_id in user_record.get("ground_truth_item_ids", [])]
        permutations = user_record.get("permutations")
        if permutations:
            for permutation in permutations:
                row_key = f"{user_key}::perm{permutation.get('permutation_index', len(recommendations))}"
                recommendations[row_key] = [str(item_id) for item_id in permutation.get("reranked_item_ids", [])]
                ground_truth[row_key] = relevant
        else:
            recommendations[str(user_key)] = [str(item_id) for item_id in user_record["reranked_item_ids"]]
            ground_truth[str(user_key)] = relevant
    return recommendations, ground_truth


def _compact_parser_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_parse_errors": metrics["num_parse_errors"],
        "parse_error_rate": metrics["parse_error_rate"],
        "num_parser_repair_attempts": metrics["num_parser_repair_attempts"],
        "parser_repair_attempt_rate": metrics["parser_repair_attempt_rate"],
    }


def _compact_preference_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "num_users_with_preference_consistency",
            "preference_buckets",
            "minimum_bucket_observations",
            "ppi",
            "gpi",
        )
        if key in metrics
    }


def _compact_robustness_metrics(metrics: dict[str, Any], k: int) -> dict[str, Any]:
    keys = (
        "mean_ground_truth_rank",
        "mean_ground_truth_rank_variance",
        "mean_ground_truth_rank_std",
        "mean_user_kendall_tau",
        "mean_user_spearman",
        f"mean_top_{k}_jaccard",
        f"topk_overlap@{k}",
        f"mean_top_{k}_set_agreement",
    )
    return {key: metrics[key] for key in keys if key in metrics}


if __name__ == "__main__":
    main()

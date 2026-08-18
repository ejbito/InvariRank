from __future__ import annotations

import argparse
import gc
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.reranking.history import load_user_histories
from experiments.reranking.methods.common import ranking_sample
from experiments.reranking.scoring import ScoringConfig, load_scorer
from experiments.scripts.common import load_dataset_settings, processed_dir
from experiments.utils.io import ensure_dir, read_json, write_json
from experiments.utils.progress import progress
from invarirank import FINE_TUNED_METHODS, method_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate a STELLA position-transition matrix.")
    parser.add_argument("--input", required=True, help="Ground-truth-filtered candidate artifact.")
    parser.add_argument("--dataset", default="movielens")
    parser.add_argument("--dataset-config", default="experiments/configs/datasets.yaml")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--model-name-or-path", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--scoring", choices=["generation", "marker_logprob"], default="marker_logprob")
    parser.add_argument("--prompt", choices=["rankgpt", "marker"], default="marker")
    parser.add_argument("--architecture", choices=sorted(FINE_TUNED_METHODS), default="lft")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-history-items", type=int, default=20)
    parser.add_argument("--max-users", type=int, default=150)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    calibrate_stella(parse_args())


def calibrate_stella(args: argparse.Namespace) -> Path:
    payload = read_json(args.input)
    _validate_calibration_payload(payload)
    if args.max_users < 1:
        raise ValueError("--max-users must be at least one.")
    if args.repeats < 1 or args.batch_size < 1:
        raise ValueError("--repeats and --batch-size must be positive.")
    if args.smoothing < 0:
        raise ValueError("--smoothing must be non-negative.")

    all_users = list(payload.get("users", {}).values())
    _validate_probe_users(all_users)
    users = _sample_probe_users(all_users, max_users=args.max_users, seed=args.seed)
    if not users:
        raise ValueError("The candidate artifact contains no users.")

    dataset_config = load_dataset_settings(args.dataset_config, args.dataset)
    histories = load_user_histories(
        processed_dir(dataset_config, args.processed_dir),
        max_history_items=args.max_history_items,
        split=str(payload.get("split") or "val"),
    )
    candidate_counts = {len(user["candidates"]) for user in users}
    if len(candidate_counts) != 1:
        raise ValueError("STELLA calibration requires a fixed candidate-list length.")
    count = candidate_counts.pop()
    if count < 2:
        raise ValueError("Calibration requires at least two candidates.")

    scorer = load_scorer(
        ScoringConfig(
            scoring=args.scoring,
            prompt=args.prompt,
            architecture=args.architecture,
            model_name_or_path=args.model_name_or_path,
            adapter_path=args.adapter_path,
            device=args.device,
            torch_dtype=args.torch_dtype,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            parser_repair="error",
            seed=args.seed,
        )
    )
    resolved_architecture = _resolved_architecture(scorer, args.architecture)
    matrix = np.full((count, count), float(args.smoothing), dtype=np.float64)
    requests = []
    rows = []
    for user in users:
        sample = ranking_sample(user, histories.get(int(user["user_id"]), []))
        relevant_ids = {str(item_id) for item_id in user.get("ground_truth_item_ids", [])}
        relevant_indices = [
            index for index, candidate in enumerate(sample.candidates)
            if str(candidate.get("item_id")) in relevant_ids
        ]
        for target_index in relevant_indices:
            for target_position in range(count):
                for repeat in range(args.repeats):
                    others = [index for index in range(count) if index != target_index]
                    seed = args.seed + int(user["user_id"]) * 1_000_003 + target_position * 1009 + repeat
                    random.Random(seed).shuffle(others)
                    permutation = others[:target_position] + [target_index] + others[target_position:]
                    requests.append((sample, permutation))
                    rows.append((target_position, permutation))

    total_batches = (len(requests) + args.batch_size - 1) // args.batch_size
    print(
        f"STELLA calibration: {len(users)} probe users, {len(requests)} scoring requests, "
        f"{total_batches} batches (batch size {args.batch_size})."
    )
    for start in progress(
        range(0, len(requests), args.batch_size),
        desc="Calibrating STELLA",
        total=total_batches,
        enabled=not getattr(args, "no_progress", False),
    ):
        batch_requests = requests[start : start + args.batch_size]
        results = scorer.rank_many(batch_requests, batch_size=args.batch_size)
        for result, (target_position, permutation) in zip(
            results,
            rows[start : start + args.batch_size],
            strict=True,
        ):
            predicted_position = permutation.index(result.items[0].candidate_index)
            matrix[target_position, predicted_position] += 1.0

    del scorer
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    row_sums = matrix.sum(axis=1, keepdims=True)
    transition = matrix / row_sums
    output = Path(args.output)
    ensure_dir(output.parent)
    write_json(
        {
            "transition_matrix": transition.tolist(),
            "provenance": {
                "model_name_or_path": args.model_name_or_path,
                "adapter_path": args.adapter_path,
                "scoring": args.scoring,
                "prompt": args.prompt,
                "architecture": resolved_architecture,
                "candidate_count": count,
                "split": "val",
            },
            "diagnostics": {
                "candidate_count": count,
                "num_probe_users": len(users),
                "num_available_probe_users": len(all_users),
                "num_queries": len(requests),
                "repeats": args.repeats,
                "smoothing": args.smoothing,
                "seed": args.seed,
                "source_candidates": str(args.input),
            },
        },
        output,
        sort_keys=False,
    )
    print(f"Saved STELLA transition matrix to {output}")
    return output


def _validate_probe_users(users: Sequence[Mapping[str, Any]]) -> None:
    invalid = []
    for user in users:
        user_id = str(user.get("user_id", "<unknown>"))
        relevant_ids = {str(item_id) for item_id in user.get("ground_truth_item_ids", [])}
        candidate_ids = {
            str(candidate.get("item_id"))
            for candidate in user.get("candidates", {}).values()
            if candidate.get("item_id") is not None
        }
        if not relevant_ids:
            invalid.append(f"user {user_id} has no ground-truth items")
        elif not relevant_ids.intersection(candidate_ids):
            invalid.append(f"user {user_id} has no ground-truth item in candidates")
    if invalid:
        preview = "; ".join(invalid[:5])
        suffix = f"; and {len(invalid) - 5} more" if len(invalid) > 5 else ""
        raise ValueError(
            "STELLA calibration requires every probe user to have ground truth in candidates: "
            f"{preview}{suffix}. Re-export validation candidates with ground-truth filtering enabled."
        )


def _validate_calibration_payload(payload: Mapping[str, Any]) -> None:
    if str(payload.get("split") or "").lower() != "val":
        raise ValueError(
            "STELLA calibration requires a validation candidate artifact (split='val'); "
            "using test candidates would leak test relevance."
        )
    if not isinstance(payload.get("users"), Mapping):
        raise ValueError("STELLA calibration input must contain a 'users' mapping.")


def _sample_probe_users(
    users: Sequence[Mapping[str, Any]],
    *,
    max_users: int,
    seed: int,
) -> list[Mapping[str, Any]]:
    values = list(users)
    if len(values) <= max_users:
        return values
    return random.Random(seed).sample(values, max_users)


def _resolved_architecture(scorer: Any, requested: str) -> str:
    scorer_config = getattr(scorer, "config", None)
    if scorer_config is None:
        return requested
    if hasattr(scorer_config, "attention_mask") and hasattr(scorer_config, "position_ids"):
        return method_from_config(scorer_config)
    return requested


if __name__ == "__main__":
    main()

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def marginal_position_exposure(
    users: Mapping[str, Mapping[str, Any]],
    *,
    k: int = 5,
    include_values: bool = False,
) -> dict[str, Any]:
    """Estimate top-k exposure by serialized input position across permutation runs."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    position_counts: dict[int, int] = {}
    exposure_counts: dict[int, int] = {}
    baseline_sums: dict[int, float] = {}
    per_observation = []

    for user_key, user_record in users.items():
        for permutation in user_record.get("permutations", []):
            input_ids = [str(item_id) for item_id in permutation.get("input_candidate_item_ids", [])]
            output_ids = [str(item_id) for item_id in permutation.get("reranked_item_ids", [])]
            if not input_ids or set(input_ids) != set(output_ids):
                continue

            top_k = set(output_ids[:k])
            baseline = min(k, len(input_ids)) / len(input_ids)
            for position, item_id in enumerate(input_ids):
                exposed = int(item_id in top_k)
                position_counts[position] = position_counts.get(position, 0) + 1
                exposure_counts[position] = exposure_counts.get(position, 0) + exposed
                baseline_sums[position] = baseline_sums.get(position, 0.0) + baseline
                if include_values:
                    per_observation.append(
                        {
                            "user_id": str(user_key),
                            "permutation_index": permutation.get("permutation_index"),
                            "input_position": position,
                            "item_id": item_id,
                            "top_k_exposed": bool(exposed),
                        }
                    )

    positions = sorted(position_counts)
    exposure = {
        str(position): exposure_counts.get(position, 0) / position_counts[position]
        for position in positions
    }
    expected = {
        str(position): baseline_sums[position] / position_counts[position]
        for position in positions
    }
    deviations = {
        str(position): exposure[str(position)] - expected[str(position)]
        for position in positions
    }
    absolute_deviations = [abs(value) for value in deviations.values()]
    result = {
        "position_bias_metric": "marginal_top_k_position_exposure",
        "k": k,
        "num_positions": len(positions),
        "num_observations": sum(position_counts.values()),
        "exposure_by_position": exposure,
        "expected_exposure_by_position": expected,
        "exposure_deviation_by_position": deviations,
        "mean_absolute_exposure_deviation": _mean(absolute_deviations),
        "max_absolute_exposure_deviation": max(absolute_deviations) if absolute_deviations else None,
    }
    if include_values:
        result["position_observations"] = per_observation
    return result


def position_bias_metrics_path(
    dataset: str,
    retriever: str,
    candidate_file_stem: str,
    reranker_file_stem: str,
    k: int,
) -> Path:
    return (
        Path("artifacts/metrics/position_bias")
        / dataset
        / retriever
        / candidate_file_stem
        / f"{reranker_file_stem}_top{k}.json"
    )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


__all__ = ["marginal_position_exposure", "position_bias_metrics_path"]

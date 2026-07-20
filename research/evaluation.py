from __future__ import annotations

import json
import math
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from statistics import pstdev
from typing import Any

from invarirank.framework import RankingResult


def _to_list(values: Any) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    return [float(value) for value in values]


def hr_at_k(scores: Any, relevance: Any, k: int) -> float:
    score_values = _to_list(scores)
    relevance_values = _to_list(relevance)
    if not score_values or not relevance_values:
        return 0.0
    count = min(len(score_values), len(relevance_values))
    order = sorted(range(count), key=lambda index: score_values[index], reverse=True)
    top = order[: min(k, count)]
    return 1.0 if any(relevance_values[index] > 0 for index in top) else 0.0


def ndcg_at_k(scores: Any, relevance: Any, k: int) -> float:
    score_values = _to_list(scores)
    relevance_values = _to_list(relevance)
    count = min(len(score_values), len(relevance_values), k)
    if count <= 0:
        return 0.0
    order = sorted(
        range(min(len(score_values), len(relevance_values))),
        key=lambda index: score_values[index],
        reverse=True,
    )[:count]
    ideal = sorted(relevance_values, reverse=True)[:count]

    def dcg(labels: list[float]) -> float:
        return sum((2.0**label - 1.0) / math.log2(rank + 2) for rank, label in enumerate(labels))

    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return float(dcg([relevance_values[index] for index in order]) / ideal_dcg)


def spearman_rho_from_rank_maps(first: dict[Any, int], second: dict[Any, int]) -> float | None:
    keys = sorted(set(first) & set(second))
    count = len(keys)
    if count < 2:
        return None
    difference_squared = sum((first[key] - second[key]) ** 2 for key in keys)
    return float(1.0 - (6.0 * difference_squared) / (count * (count * count - 1)))


def kendall_tau_from_rank_maps(first: dict[Any, int], second: dict[Any, int]) -> float | None:
    keys = sorted(set(first) & set(second))
    if len(keys) < 2:
        return None
    concordant = 0
    discordant = 0
    for left_index in range(len(keys) - 1):
        for right_index in range(left_index + 1, len(keys)):
            left = keys[left_index]
            right = keys[right_index]
            product = (first[left] - first[right]) * (second[left] - second[right])
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return None
    return float((concordant - discordant) / total)


def topk_overlap_at_k(first: list[Any], second: list[Any], k: int) -> float:
    if k <= 0:
        return 0.0
    first_top = first[:k]
    second_top = second[:k]
    effective_k = min(len(first_top), len(second_top))
    if effective_k == 0:
        return 0.0
    return float(len(set(first_top[:effective_k]) & set(second_top[:effective_k])) / effective_k)


def load_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        text = handle.read().strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Ranked lists file must contain a JSON list or JSONL records: {path}")


def load_ranked_lists(path: str | Path) -> list[dict[str, Any]]:
    return load_json_or_jsonl(path)


def build_score_vector(
    candidate_indices: list[int],
    output_candidates: list[int],
    output_scores: list[float],
) -> list[float]:
    position_by_candidate = {candidate: index for index, candidate in enumerate(candidate_indices)}
    scores = [float("-inf")] * len(candidate_indices)
    for score, candidate in zip(output_scores, output_candidates):
        if candidate in position_by_candidate:
            scores[position_by_candidate[candidate]] = float(score)
    return scores


def rank_map(order: list[Any]) -> dict[Any, int]:
    return {item: rank for rank, item in enumerate(order)}


def mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def standard_deviation(values: list[float]) -> float | None:
    if not values:
        return None
    return 0.0 if len(values) == 1 else float(pstdev(values))


def evaluate_permutation_effectiveness(permutation: dict[str, Any], top_k: Sequence[int]) -> dict[str, float]:
    input_data = permutation.get("input", {})
    output_data = permutation.get("output_ranking", {})
    candidate_indices = [int(value) for value in input_data.get("candidate_indices", [])]
    relevance = [int(value) for value in input_data.get("relevance", [])]
    output_candidates = [int(value) for value in output_data.get("candidate_indices", [])]
    output_scores = [float(value) for value in output_data.get("scores", [])]
    if not candidate_indices or not relevance or not output_candidates:
        return {}

    scores = build_score_vector(candidate_indices, output_candidates, output_scores)
    metrics: dict[str, float] = {}
    for k in top_k:
        metrics[f"hr@{k}"] = hr_at_k(scores, relevance, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(scores, relevance, k)
    return metrics


def evaluate_record_robustness(record: dict[str, Any], top_k: Sequence[int]) -> dict[str, float]:
    output_maps: list[dict[int, int]] = []
    output_orders: list[list[int]] = []
    for permutation in record.get("permutations", []):
        candidates = [int(value) for value in permutation.get("output_ranking", {}).get("candidate_indices", [])]
        if candidates:
            output_orders.append(candidates)
            output_maps.append(rank_map(candidates))
    if len(output_maps) < 2:
        return {}

    kendalls: list[float] = []
    spearmans: list[float] = []
    topk_values: dict[int, list[float]] = {k: [] for k in top_k}
    for (first_map, first_order), (second_map, second_order) in combinations(
        zip(output_maps, output_orders),
        2,
    ):
        spearman = spearman_rho_from_rank_maps(first_map, second_map)
        if spearman is not None:
            spearmans.append(spearman)
        kendall = kendall_tau_from_rank_maps(first_map, second_map)
        if kendall is not None:
            kendalls.append(kendall)
        for k in top_k:
            topk_values[k].append(topk_overlap_at_k(first_order, second_order, k))

    metrics: dict[str, float] = {}
    if (value := mean(spearmans)) is not None:
        metrics["permutation_spearman"] = value
    if (value := mean(kendalls)) is not None:
        metrics["permutation_kendall"] = value
    for k, values in topk_values.items():
        if (value := mean(values)) is not None:
            metrics[f"permutation_topk_overlap@{k}"] = value
    return metrics


def input_position_map(permutation: dict[str, Any]) -> dict[int, int]:
    candidate_indices = [int(value) for value in permutation.get("input", {}).get("candidate_indices", [])]
    return {candidate: position for position, candidate in enumerate(candidate_indices)}


def position_bucket(position: int, list_length: int) -> int:
    if list_length <= 1:
        return 0
    return min(2, int((position * 3) / list_length))


def pairwise_preference_probabilities(
    items: list[int],
    output_maps: list[dict[int, int]],
) -> dict[tuple[int, int], float]:
    probabilities: dict[tuple[int, int], float] = {}
    for left, right in combinations(items, 2):
        preferences = [
            1.0 if output_map[left] < output_map[right] else 0.0
            for output_map in output_maps
            if left in output_map and right in output_map
        ]
        if preferences:
            value = sum(preferences) / len(preferences)
            probabilities[(left, right)] = value
            probabilities[(right, left)] = 1.0 - value
    return probabilities


def pairwise_preference_instability(
    items: list[int],
    output_maps: list[dict[int, int]],
    input_maps: list[dict[int, int]],
) -> float | None:
    pair_deltas: list[float] = []
    for left, right in combinations(items, 2):
        bucket_preferences: dict[tuple[int, int], list[float]] = {}
        for output_map, input_map in zip(output_maps, input_maps):
            if left not in output_map or right not in output_map or left not in input_map or right not in input_map:
                continue
            list_length = len(input_map)
            bucket_pair = (
                position_bucket(input_map[left], list_length),
                position_bucket(input_map[right], list_length),
            )
            preference = 1.0 if output_map[left] < output_map[right] else 0.0
            bucket_preferences.setdefault(bucket_pair, []).append(preference)
        bucket_probabilities = [sum(values) / len(values) for values in bucket_preferences.values() if values]
        if len(bucket_probabilities) >= 2:
            pair_deltas.append(max(bucket_probabilities) - min(bucket_probabilities))
    return mean(pair_deltas)


def global_preference_inconsistency(
    items: list[int],
    pairwise_probabilities: dict[tuple[int, int], float],
) -> float | None:
    if len(items) < 2:
        return None
    best_fit_order = sorted(
        items,
        key=lambda item: sum(pairwise_probabilities.get((item, other), 0.5) for other in items if other != item),
        reverse=True,
    )
    best_fit_map = rank_map(best_fit_order)
    disagreements: list[float] = []
    for left, right in combinations(items, 2):
        preference = pairwise_probabilities.get((left, right))
        if preference is None:
            continue
        disagreements.append(1.0 - preference if best_fit_map[left] < best_fit_map[right] else preference)
    return mean(disagreements)


def preference_cycle_rate(
    items: list[int],
    pairwise_probabilities: dict[tuple[int, int], float],
) -> float | None:
    if len(items) < 3:
        return None
    cycles = 0
    total = 0
    for first, second, third in combinations(items, 3):
        first_second = pairwise_probabilities.get((first, second), 0.5) > 0.5
        second_third = pairwise_probabilities.get((second, third), 0.5) > 0.5
        first_third = pairwise_probabilities.get((first, third), 0.5) > 0.5
        total += 1
        if (first_second and second_third and not first_third) or (
            not first_second and not second_third and first_third
        ):
            cycles += 1
    return float(cycles / total) if total else None


def listwise_ranking_instability(output_maps: list[dict[int, int]]) -> float | None:
    distances: list[float] = []
    for first, second in combinations(output_maps, 2):
        kendall = kendall_tau_from_rank_maps(first, second)
        if kendall is not None:
            distances.append((1.0 - kendall) / 2.0)
    return mean(distances)


def evaluate_record_validity(record: dict[str, Any]) -> dict[str, float]:
    output_maps: list[dict[int, int]] = []
    input_maps: list[dict[int, int]] = []
    for permutation in record.get("permutations", []):
        candidates = [int(value) for value in permutation.get("output_ranking", {}).get("candidate_indices", [])]
        if candidates:
            output_maps.append(rank_map(candidates))
            input_maps.append(input_position_map(permutation))
    if len(output_maps) < 2:
        return {}

    items = sorted(set().union(*(set(output_map) for output_map in output_maps)))
    if len(items) < 2:
        return {}
    pairwise_probabilities = pairwise_preference_probabilities(items, output_maps)
    metrics = {
        "PPI": pairwise_preference_instability(items, output_maps, input_maps),
        "GPI": global_preference_inconsistency(items, pairwise_probabilities),
        "PCR": preference_cycle_rate(items, pairwise_probabilities),
        "LRI": listwise_ranking_instability(output_maps),
    }
    return {key: value for key, value in metrics.items() if value is not None}


def aggregate_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    aggregated: dict[str, list[float]] = {}
    for metrics in metrics_list:
        for key, value in metrics.items():
            aggregated.setdefault(key, []).append(value)
    return {key: sum(values) / len(values) for key, values in aggregated.items() if values}


def aggregate_validity(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    aggregated: dict[str, list[float]] = {}
    for metrics in metrics_list:
        for key, value in metrics.items():
            aggregated.setdefault(key, []).append(value)
    output: dict[str, float] = {}
    for key, values in aggregated.items():
        if values:
            output[key] = float(sum(values) / len(values))
            output[f"{key}_std"] = standard_deviation(values) or 0.0
    return output


def ranking_results_to_records(results: Sequence[RankingResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str | None, Any], dict[str, Any]] = {}
    for result in results:
        candidate_set = tuple(sorted(item.item_id for item in result.items))
        group_id = result.metadata.get("sample_index", candidate_set)
        key = (result.user_id, result.split, group_id)
        record = grouped.setdefault(
            key,
            {
                "sample_index": group_id,
                "user_id": result.user_id,
                "split": result.split,
                "permutations": [],
            },
        )
        by_index = {item.candidate_index: item for item in result.items}
        input_items = [by_index[index] for index in result.permutation]
        record["permutations"].append(
            {
                "permutation_index": len(record["permutations"]),
                "input": {
                    "candidate_indices": list(result.permutation),
                    "item_ids": [item.item_id for item in input_items],
                    "relevance": [item.relevance or 0 for item in input_items],
                },
                "scores": [item.score for item in input_items],
                "output_ranking": {
                    "candidate_indices": [item.candidate_index for item in result.items],
                    "item_ids": [item.item_id for item in result.items],
                    "scores": [item.score for item in result.items],
                },
            }
        )
    return list(grouped.values())


def evaluate(
    results: Sequence[RankingResult] | Sequence[dict[str, Any]],
    *,
    top_k: Sequence[int] = (5, 10),
    show_progress: bool = False,
) -> dict[str, Any]:
    values = list(results)
    if values and isinstance(values[0], RankingResult):
        records = ranking_results_to_records(values)  # type: ignore[arg-type]
    else:
        records = values  # type: ignore[assignment]

    effectiveness: list[dict[str, float]] = []
    robustness: list[dict[str, float]] = []
    validity: list[dict[str, float]] = []
    record_iterator: Any = records
    if show_progress:
        from tqdm.auto import tqdm

        record_iterator = tqdm(records, desc="[Evaluation] Records", unit="record", dynamic_ncols=True)
    for record in record_iterator:
        effectiveness.extend(
            metrics
            for permutation in record.get("permutations", [])
            if (metrics := evaluate_permutation_effectiveness(permutation, top_k))
        )
        if metrics := evaluate_record_robustness(record, top_k):
            robustness.append(metrics)
        if metrics := evaluate_record_validity(record):
            validity.append(metrics)
    return {
        "num_records": len(records),
        "num_permutations": len(effectiveness),
        "effectiveness": aggregate_metrics(effectiveness),
        "robustness": aggregate_metrics(robustness),
        "validity": aggregate_validity(validity),
    }


__all__ = [
    "aggregate_metrics",
    "aggregate_validity",
    "evaluate",
    "evaluate_permutation_effectiveness",
    "evaluate_record_robustness",
    "evaluate_record_validity",
    "hr_at_k",
    "kendall_tau_from_rank_maps",
    "load_json_or_jsonl",
    "load_ranked_lists",
    "ndcg_at_k",
    "ranking_results_to_records",
    "spearman_rho_from_rank_maps",
    "topk_overlap_at_k",
]

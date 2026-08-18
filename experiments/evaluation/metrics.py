from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def evaluate_at_k(
    recommendations: Mapping[Any, Sequence[Any]],
    ground_truth: Mapping[Any, Iterable[Any]],
    k: int,
) -> dict[str, float]:
    return {
        f"hit_rate@{k}": hit_rate_at_k(recommendations, ground_truth, k),
        f"recall@{k}": recall_at_k(recommendations, ground_truth, k),
        f"ndcg@{k}": ndcg_at_k(recommendations, ground_truth, k),
        f"mrr@{k}": mrr_at_k(recommendations, ground_truth, k),
    }
def hit_rate_at_k(
    recommendations: Mapping[Any, Sequence[Any]],
    ground_truth: Mapping[Any, Iterable[Any]],
    k: int,
) -> float:
    rows = _normalize_inputs(recommendations, ground_truth)
    if not rows:
        return 0.0
    hits = 0
    for top_items, relevant in rows:
        hits += int(any(item_id in relevant for item_id in top_items[:k]))
    return hits / len(rows)


def recall_at_k(
    recommendations: Mapping[Any, Sequence[Any]],
    ground_truth: Mapping[Any, Iterable[Any]],
    k: int,
) -> float:
    recalls = []
    for top_items, relevant in _normalize_inputs(recommendations, ground_truth):
        if relevant:
            recalls.append(len(set(top_items[:k]) & relevant) / len(relevant))
    return sum(recalls) / len(recalls) if recalls else 0.0


def ndcg_at_k(
    recommendations: Mapping[Any, Sequence[Any]],
    ground_truth: Mapping[Any, Iterable[Any]],
    k: int,
) -> float:
    scores = []
    for top_items, relevant in _normalize_inputs(recommendations, ground_truth):
        if not relevant:
            continue
        dcg = 0.0
        for rank, item_id in enumerate(top_items[:k], start=1):
            if item_id in relevant:
                dcg += 1.0 / math.log2(rank + 1)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        scores.append(dcg / idcg if idcg else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def mrr_at_k(
    recommendations: Mapping[Any, Sequence[Any]],
    ground_truth: Mapping[Any, Iterable[Any]],
    k: int,
) -> float:
    reciprocal_ranks = []
    for top_items, relevant in _normalize_inputs(recommendations, ground_truth):
        rank_score = 0.0
        for rank, item_id in enumerate(top_items[:k], start=1):
            if item_id in relevant:
                rank_score = 1.0 / rank
                break
        reciprocal_ranks.append(rank_score)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def permutation_robustness_metrics(
    users: Mapping[str, Mapping[str, Any]],
    k: int = 10,
    include_values: bool = False,
) -> dict[str, Any]:
    per_user = {}
    mean_kendall_values = []
    mean_spearman_values = []
    mean_gt_rank_values = []
    gt_rank_variance_values = []
    gt_rank_std_values = []
    mean_top_k_jaccard_values = []
    topk_overlap_values = []
    first_place_agreement_values = []
    top_k_set_agreement_values = []

    for user_key, user_record in users.items():
        permutations = user_record.get("permutations", [])
        rankings = [permutation["reranked_item_ids"] for permutation in permutations]
        correlations = pairwise_rank_correlations(rankings)
        gt_ranks = _ground_truth_ranks(rankings, user_record.get("ground_truth_item_ids", []))
        gt_rank_variance = _variance(gt_ranks)
        gt_rank_std = math.sqrt(gt_rank_variance) if gt_rank_variance is not None else None
        top_k_stability = _pairwise_top_k_stability(rankings, k)
        user_result = {
            "num_permutations": len(permutations),
            "num_pairs": correlations["num_pairs"],
            "mean_kendall_tau": correlations["mean_kendall_tau"],
            "mean_spearman": correlations["mean_spearman"],
            "mean_ground_truth_rank": _mean(gt_ranks),
            "ground_truth_rank_variance": gt_rank_variance,
            "ground_truth_rank_std": gt_rank_std,
            f"mean_top_{k}_jaccard": top_k_stability["mean_top_k_jaccard"],
            f"topk_overlap@{k}": top_k_stability["topk_overlap"],
            "first_place_agreement": top_k_stability["first_place_agreement"],
            f"top_{k}_set_agreement": top_k_stability["top_k_set_agreement"],
        }
        if include_values:
            user_result["kendall_tau_values"] = correlations["kendall_tau_values"]
            user_result["spearman_values"] = correlations["spearman_values"]
            user_result["ground_truth_ranks"] = gt_ranks
            user_result[f"top_{k}_jaccard_values"] = top_k_stability["top_k_jaccard_values"]
            user_result[f"topk_overlap@{k}_values"] = top_k_stability["topk_overlap_values"]
            user_result["first_place_agreement_values"] = top_k_stability["first_place_agreement_values"]
            user_result[f"top_{k}_set_agreement_values"] = top_k_stability["top_k_set_agreement_values"]
        per_user[str(user_key)] = user_result
        if correlations["mean_kendall_tau"] is not None:
            mean_kendall_values.append(correlations["mean_kendall_tau"])
        if correlations["mean_spearman"] is not None:
            mean_spearman_values.append(correlations["mean_spearman"])
        if user_result["mean_ground_truth_rank"] is not None:
            mean_gt_rank_values.append(user_result["mean_ground_truth_rank"])
        if gt_rank_variance is not None:
            gt_rank_variance_values.append(gt_rank_variance)
        if gt_rank_std is not None:
            gt_rank_std_values.append(gt_rank_std)
        if top_k_stability["mean_top_k_jaccard"] is not None:
            mean_top_k_jaccard_values.append(top_k_stability["mean_top_k_jaccard"])
        if top_k_stability["topk_overlap"] is not None:
            topk_overlap_values.append(top_k_stability["topk_overlap"])
        if top_k_stability["first_place_agreement"] is not None:
            first_place_agreement_values.append(top_k_stability["first_place_agreement"])
        if top_k_stability["top_k_set_agreement"] is not None:
            top_k_set_agreement_values.append(top_k_stability["top_k_set_agreement"])

    return {
        "num_users_with_pairwise_robustness": len(mean_kendall_values),
        "mean_user_kendall_tau": _mean(mean_kendall_values),
        "mean_user_spearman": _mean(mean_spearman_values),
        "num_users_with_ground_truth_rank_robustness": len(mean_gt_rank_values),
        "mean_ground_truth_rank": _mean(mean_gt_rank_values),
        "mean_ground_truth_rank_variance": _mean(gt_rank_variance_values),
        "mean_ground_truth_rank_std": _mean(gt_rank_std_values),
        f"mean_top_{k}_jaccard": _mean(mean_top_k_jaccard_values),
        f"topk_overlap@{k}": _mean(topk_overlap_values),
        "mean_first_place_agreement": _mean(first_place_agreement_values),
        f"mean_top_{k}_set_agreement": _mean(top_k_set_agreement_values),
        "per_user_robustness": per_user,
    }


def parser_quality_metrics(users: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    outputs = list(_iter_reranking_outputs(users))
    if not outputs:
        return {
            "num_outputs": 0,
            "num_parse_errors": 0,
            "parse_error_rate": 0.0,
            "num_outputs_with_invalid_item_ids": 0,
            "num_users_with_invalid_item_ids": 0,
            "invalid_item_id_user_rate": 0.0,
            "num_invalid_item_ids": 0,
            "avg_invalid_item_ids": 0.0,
            "num_outputs_with_duplicate_item_ids": 0,
            "num_duplicate_item_ids": 0,
            "duplicate_item_id_output_rate": 0.0,
            "avg_duplicate_item_ids": 0.0,
            "num_outputs_with_missing_candidate_item_ids": 0,
            "num_missing_candidate_item_ids": 0,
            "missing_candidate_output_rate": 0.0,
            "avg_missing_candidate_item_ids": 0.0,
            "num_parser_repair_attempts": 0,
            "parser_repair_attempt_rate": 0.0,
        }

    num_outputs = len(outputs)
    parse_errors = sum(1 for output in outputs if output.get("parse_error") is not None)
    invalid_counts = [_count(output, "invalid_item_ids", "num_invalid_item_ids") for output in outputs]
    duplicate_counts = [_count(output, "duplicate_item_ids", "num_duplicate_item_ids") for output in outputs]
    missing_counts = [_count(output, "missing_candidate_item_ids", "num_missing_candidate_item_ids") for output in outputs]
    repair_attempts = sum(1 for output in outputs if output.get("parser_repair_applied"))
    outputs_with_invalid = sum(1 for count in invalid_counts if count > 0)
    outputs_with_duplicate = sum(1 for count in duplicate_counts if count > 0)
    outputs_with_missing = sum(1 for count in missing_counts if count > 0)

    return {
        "num_outputs": num_outputs,
        "num_parse_errors": parse_errors,
        "parse_error_rate": parse_errors / num_outputs,
        "num_outputs_with_invalid_item_ids": outputs_with_invalid,
        "num_users_with_invalid_item_ids": outputs_with_invalid,
        "invalid_item_id_user_rate": outputs_with_invalid / num_outputs,
        "num_invalid_item_ids": sum(invalid_counts),
        "avg_invalid_item_ids": sum(invalid_counts) / num_outputs,
        "num_outputs_with_duplicate_item_ids": outputs_with_duplicate,
        "num_duplicate_item_ids": sum(duplicate_counts),
        "duplicate_item_id_output_rate": outputs_with_duplicate / num_outputs,
        "avg_duplicate_item_ids": sum(duplicate_counts) / num_outputs,
        "num_outputs_with_missing_candidate_item_ids": outputs_with_missing,
        "num_missing_candidate_item_ids": sum(missing_counts),
        "missing_candidate_output_rate": outputs_with_missing / num_outputs,
        "avg_missing_candidate_item_ids": sum(missing_counts) / num_outputs,
        "num_parser_repair_attempts": repair_attempts,
        "parser_repair_attempt_rate": repair_attempts / num_outputs,
    }


def _normalize_inputs(
    recommendations: Mapping[Any, Sequence[Any]],
    ground_truth: Mapping[Any, Iterable[Any]],
) -> list[tuple[list[str], set[str]]]:
    normalized_recommendations = {
        str(user_id): [str(item_id) for item_id in item_ids]
        for user_id, item_ids in recommendations.items()
    }
    return [
        (
            normalized_recommendations.get(str(user_id), []),
            {str(item_id) for item_id in relevant_items},
        )
        for user_id, relevant_items in ground_truth.items()
    ]


def _ground_truth_ranks(
    rankings: Sequence[Sequence[str]],
    ground_truth_item_ids: Iterable[str],
) -> list[int]:
    relevant = {str(item_id) for item_id in ground_truth_item_ids}
    if not relevant:
        return []

    ranks = []
    for ranking in rankings:
        rank_by_item = {str(item_id): rank for rank, item_id in enumerate(ranking, start=1)}
        matching_ranks = [rank_by_item[item_id] for item_id in relevant if item_id in rank_by_item]
        if matching_ranks:
            ranks.append(min(matching_ranks))
    return ranks


def _pairwise_top_k_stability(rankings: Sequence[Sequence[str]], k: int) -> dict[str, Any]:
    pairs = list(_ranking_pairs(rankings))
    if not pairs:
        return {
            "mean_top_k_jaccard": None,
            "topk_overlap": None,
            "first_place_agreement": None,
            "top_k_set_agreement": None,
            "top_k_jaccard_values": [],
            "topk_overlap_values": [],
            "first_place_agreement_values": [],
            "top_k_set_agreement_values": [],
        }

    jaccard_values = []
    overlap_values = []
    first_place_values = []
    top_k_set_values = []
    for first, second in pairs:
        effective_k = min(k, len(first), len(second))
        first_top_k = set(str(item_id) for item_id in first[:effective_k])
        second_top_k = set(str(item_id) for item_id in second[:effective_k])
        union = first_top_k | second_top_k
        jaccard_values.append(len(first_top_k & second_top_k) / len(union) if union else 1.0)
        overlap_values.append(len(first_top_k & second_top_k) / effective_k if effective_k else 0.0)
        first_place_values.append(float(bool(first and second and str(first[0]) == str(second[0]))))
        top_k_set_values.append(float(first_top_k == second_top_k))

    return {
        "mean_top_k_jaccard": _mean(jaccard_values),
        "topk_overlap": _mean(overlap_values),
        "first_place_agreement": _mean(first_place_values),
        "top_k_set_agreement": _mean(top_k_set_values),
        "top_k_jaccard_values": jaccard_values,
        "topk_overlap_values": overlap_values,
        "first_place_agreement_values": first_place_values,
        "top_k_set_agreement_values": top_k_set_values,
    }


def _ranking_pairs(rankings: Sequence[Sequence[str]]):
    for left in range(len(rankings)):
        for right in range(left + 1, len(rankings)):
            yield [str(item_id) for item_id in rankings[left]], [str(item_id) for item_id in rankings[right]]


def pairwise_rank_correlations(rankings: Sequence[Sequence[str]]) -> dict[str, Any]:
    pairs = list(_ranking_pairs(rankings))
    if not pairs:
        return {
            "num_pairs": 0,
            "mean_kendall_tau": None,
            "mean_spearman": None,
            "kendall_tau_values": [],
            "spearman_values": [],
        }

    kendall_values = [kendall_tau(first, second) for first, second in pairs]
    spearman_values = [spearman_correlation(first, second) for first, second in pairs]
    return {
        "num_pairs": len(pairs),
        "mean_kendall_tau": sum(kendall_values) / len(kendall_values),
        "mean_spearman": sum(spearman_values) / len(spearman_values),
        "kendall_tau_values": kendall_values,
        "spearman_values": spearman_values,
    }


def kendall_tau(first: Sequence[str], second: Sequence[str]) -> float:
    first_ranks, second_ranks = _aligned_ranks(first, second)
    count = len(first_ranks)
    if count < 2:
        return 1.0

    concordant = 0
    discordant = 0
    for left in range(count):
        for right in range(left + 1, count):
            first_order = first_ranks[left] - first_ranks[right]
            second_order = second_ranks[left] - second_ranks[right]
            if first_order * second_order > 0:
                concordant += 1
            elif first_order * second_order < 0:
                discordant += 1
    total = count * (count - 1) / 2
    return (concordant - discordant) / total


def spearman_correlation(first: Sequence[str], second: Sequence[str]) -> float:
    first_ranks, second_ranks = _aligned_ranks(first, second)
    count = len(first_ranks)
    if count < 2:
        return 1.0
    mean_first = sum(first_ranks) / count
    mean_second = sum(second_ranks) / count
    numerator = sum(
        (left - mean_first) * (right - mean_second)
        for left, right in zip(first_ranks, second_ranks)
    )
    first_var = sum((rank - mean_first) ** 2 for rank in first_ranks)
    second_var = sum((rank - mean_second) ** 2 for rank in second_ranks)
    denominator = (first_var * second_var) ** 0.5
    return numerator / denominator if denominator else 0.0


def _aligned_ranks(first: Sequence[str], second: Sequence[str]) -> tuple[list[int], list[int]]:
    first = [str(item) for item in first]
    second = [str(item) for item in second]
    if set(first) != set(second):
        raise ValueError("Rankings must contain the same item IDs.")
    first_rank_map = {item_id: rank for rank, item_id in enumerate(first)}
    second_rank_map = {item_id: rank for rank, item_id in enumerate(second)}
    item_ids = sorted(first_rank_map)
    return (
        [first_rank_map[item_id] for item_id in item_ids],
        [second_rank_map[item_id] for item_id in item_ids],
    )


def _mean(values: Sequence[float | int]) -> float | None:
    return sum(values) / len(values) if values else None


def _variance(values: Sequence[float | int]) -> float | None:
    if not values:
        return None
    mean_value = sum(values) / len(values)
    return sum((value - mean_value) ** 2 for value in values) / len(values)


def _iter_reranking_outputs(users: Mapping[str, Mapping[str, Any]]):
    for user_record in users.values():
        permutations = user_record.get("permutations")
        if permutations:
            yield from permutations
        else:
            yield user_record


def _count(output: Mapping[str, Any], list_key: str, count_key: str) -> int:
    if count_key in output:
        return int(output[count_key])
    return len(output.get(list_key, []))

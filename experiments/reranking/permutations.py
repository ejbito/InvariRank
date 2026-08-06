from __future__ import annotations

import itertools
import random
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from experiments.reranking.parsers import candidate_item_ids


def permute_user_record(
    user_record: Mapping[str, Any],
    seed: int,
    avoid_original_order: bool = True,
) -> dict[str, Any]:
    output = deepcopy(dict(user_record))
    candidates = dict(user_record["candidates"])
    original_ids = candidate_item_ids(user_record)
    ordered_candidates = [
        dict(record)
        for _, record in sorted(candidates.items(), key=lambda item: int(item[0]))
    ]

    if len(ordered_candidates) <= 1:
        shuffled = ordered_candidates
    else:
        rng = random.Random(seed)
        shuffled = ordered_candidates.copy()
        for _ in range(20):
            rng.shuffle(shuffled)
            shuffled_ids = [str(record["item_id"]) for record in shuffled]
            if not avoid_original_order or shuffled_ids != original_ids:
                break

    output["candidates"] = {
        str(index): record
        for index, record in enumerate(shuffled, start=1)
    }
    return output


def pairwise_rank_correlations(rankings: Sequence[Sequence[str]]) -> dict[str, Any]:
    pairs = list(itertools.combinations([[str(item) for item in ranking] for ranking in rankings], 2))
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
    n = len(first_ranks)
    if n < 2:
        return 1.0

    concordant = 0
    discordant = 0
    for left in range(n):
        for right in range(left + 1, n):
            first_order = first_ranks[left] - first_ranks[right]
            second_order = second_ranks[left] - second_ranks[right]
            if first_order * second_order > 0:
                concordant += 1
            elif first_order * second_order < 0:
                discordant += 1
    total = n * (n - 1) / 2
    return (concordant - discordant) / total


def spearman_correlation(first: Sequence[str], second: Sequence[str]) -> float:
    first_ranks, second_ranks = _aligned_ranks(first, second)
    n = len(first_ranks)
    if n < 2:
        return 1.0
    mean_first = sum(first_ranks) / n
    mean_second = sum(second_ranks) / n
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

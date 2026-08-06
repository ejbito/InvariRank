from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def preference_consistency_metrics(
    users: Mapping[str, Mapping[str, Any]],
    *,
    num_buckets: int = 3,
    minimum_bucket_observations: int = 1,
) -> dict[str, Any]:
    """Compute RecSys preference-consistency metrics for permutation reranking runs."""
    if num_buckets < 1:
        raise ValueError("num_buckets must be at least 1.")
    if minimum_bucket_observations < 1:
        raise ValueError("minimum_bucket_observations must be at least 1.")

    ppi_values = []
    gpi_values = []
    valid_users = 0
    for user_record in users.values():
        observations = _observations(user_record)
        if len(observations) < 2:
            continue
        valid_users += 1
        ppi = pairwise_preference_instability(
            observations,
            num_buckets=num_buckets,
            minimum_bucket_observations=minimum_bucket_observations,
        )
        gpi = global_preference_inconsistency(observations)
        if ppi is not None:
            ppi_values.append(ppi)
        if gpi is not None:
            gpi_values.append(gpi)

    return {
        "num_users_with_preference_consistency": valid_users,
        "preference_buckets": num_buckets,
        "minimum_bucket_observations": minimum_bucket_observations,
        "ppi": _mean(ppi_values),
        "gpi": _mean(gpi_values),
    }


def pairwise_preference_instability(
    observations: Sequence[Mapping[str, Sequence[str]]],
    *,
    num_buckets: int = 3,
    minimum_bucket_observations: int = 1,
) -> float | None:
    candidate_ids = _candidate_universe(observations)
    if len(candidate_ids) < 2:
        return None
    if minimum_bucket_observations < 1:
        raise ValueError("minimum_bucket_observations must be at least 1.")

    pair_values = []
    for left_index, left_id in enumerate(candidate_ids):
        for right_id in candidate_ids[left_index + 1 :]:
            bucket_counts: dict[tuple[int, int], list[int]] = {}
            for observation in observations:
                input_positions = _rank_map(observation["input"])
                output_positions = _rank_map(observation["output"])
                left_bucket = _bucket(input_positions[left_id], len(candidate_ids), num_buckets)
                right_bucket = _bucket(input_positions[right_id], len(candidate_ids), num_buckets)
                wins, total = bucket_counts.setdefault((left_bucket, right_bucket), [0, 0])
                bucket_counts[(left_bucket, right_bucket)] = [
                    wins + int(output_positions[left_id] < output_positions[right_id]),
                    total + 1,
                ]
            probabilities = [
                wins / total
                for wins, total in bucket_counts.values()
                if total >= minimum_bucket_observations
            ]
            if len(probabilities) >= 2:
                pair_values.append(max(probabilities) - min(probabilities))
    return _mean(pair_values)


def global_preference_inconsistency(observations: Sequence[Mapping[str, Sequence[str]]]) -> float | None:
    candidate_ids = _candidate_universe(observations)
    pair_count = len(candidate_ids) * (len(candidate_ids) - 1) // 2
    if pair_count == 0:
        return None

    preferences = _pairwise_preferences(candidate_ids, observations)
    ranking = sorted(
        candidate_ids,
        key=lambda item_id: (-_win_score(item_id, candidate_ids, preferences), item_id),
    )
    best_cost = _ranking_disagreement(ranking, preferences)

    improved = True
    while improved:
        improved = False
        best_swap = None
        for left in range(len(ranking)):
            for right in range(left + 1, len(ranking)):
                candidate = ranking.copy()
                candidate[left], candidate[right] = candidate[right], candidate[left]
                cost = _ranking_disagreement(candidate, preferences)
                if cost < best_cost:
                    best_cost = cost
                    best_swap = candidate
        if best_swap is not None:
            ranking = best_swap
            improved = True

    return best_cost / pair_count


def _observations(user_record: Mapping[str, Any]) -> list[dict[str, list[str]]]:
    observations = []
    for permutation in user_record.get("permutations", []):
        input_ids = [str(item_id) for item_id in permutation.get("input_candidate_item_ids", [])]
        output_ids = [str(item_id) for item_id in permutation.get("reranked_item_ids", [])]
        if input_ids and set(input_ids) == set(output_ids) and len(input_ids) == len(set(input_ids)):
            observations.append({"input": input_ids, "output": output_ids})
    return observations


def _candidate_universe(observations: Sequence[Mapping[str, Sequence[str]]]) -> list[str]:
    if not observations:
        return []
    first = [str(item_id) for item_id in observations[0]["input"]]
    candidate_set = set(first)
    if any(
        set(observation["input"]) != candidate_set or set(observation["output"]) != candidate_set
        for observation in observations
    ):
        return []
    return sorted(candidate_set)


def _pairwise_preferences(
    candidate_ids: Sequence[str],
    observations: Sequence[Mapping[str, Sequence[str]]],
) -> dict[tuple[str, str], float]:
    counts: dict[tuple[str, str], list[int]] = {}
    for observation in observations:
        output_positions = _rank_map(observation["output"])
        for left_index, left_id in enumerate(candidate_ids):
            for right_id in candidate_ids[left_index + 1 :]:
                wins, total = counts.setdefault((left_id, right_id), [0, 0])
                counts[(left_id, right_id)] = [
                    wins + int(output_positions[left_id] < output_positions[right_id]),
                    total + 1,
                ]
    return {pair: wins / total for pair, (wins, total) in counts.items() if total > 0}


def _win_score(
    item_id: str,
    candidate_ids: Sequence[str],
    preferences: Mapping[tuple[str, str], float],
) -> float:
    score = 0.0
    for other_id in candidate_ids:
        if item_id == other_id:
            continue
        pair = tuple(sorted((item_id, other_id)))
        probability = preferences[pair]
        score += probability if pair[0] == item_id else 1.0 - probability
    return score


def _ranking_disagreement(ranking: Sequence[str], preferences: Mapping[tuple[str, str], float]) -> float:
    positions = _rank_map(ranking)
    disagreement = 0.0
    for (left_id, right_id), probability in preferences.items():
        if positions[left_id] < positions[right_id]:
            disagreement += 1.0 - probability
        else:
            disagreement += probability
    return disagreement


def _bucket(position: int, count: int, num_buckets: int) -> int:
    return min(position * num_buckets // count, num_buckets - 1)


def _rank_map(ranking: Sequence[str]) -> dict[str, int]:
    return {str(item_id): rank for rank, item_id in enumerate(ranking)}


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


__all__ = [
    "global_preference_inconsistency",
    "pairwise_preference_instability",
    "preference_consistency_metrics",
]

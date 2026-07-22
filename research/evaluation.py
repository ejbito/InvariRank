from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import pstdev
from typing import Any

from invarirank.framework import RankingResult

EVALUATION_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class PermutationObservation:
    input_order: tuple[int, ...]
    output_order: tuple[int, ...]
    relevance: Mapping[int, int]

    @property
    def input_positions(self) -> dict[int, int]:
        return {candidate: position for position, candidate in enumerate(self.input_order)}

    @property
    def output_ranks(self) -> dict[int, int]:
        return {candidate: rank for rank, candidate in enumerate(self.output_order)}


@dataclass(frozen=True)
class QueryObservations:
    sample_index: Any
    user_id: str
    split: str | None
    candidates: tuple[int, ...]
    observations: tuple[PermutationObservation, ...]

    @property
    def list_length(self) -> int:
        return len(self.candidates)


def _to_list(values: Any) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    return [float(value) for value in values]


def mean(values: Sequence[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def standard_deviation(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return 0.0 if len(values) == 1 else float(pstdev(values))


def hr_at_k(scores: Any, relevance: Any, k: int) -> float:
    score_values = _to_list(scores)
    relevance_values = _to_list(relevance)
    if not score_values or not relevance_values or k <= 0:
        return 0.0
    count = min(len(score_values), len(relevance_values))
    order = sorted(range(count), key=lambda index: score_values[index], reverse=True)
    return 1.0 if any(relevance_values[index] > 0 for index in order[: min(k, count)]) else 0.0


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

    def dcg(labels: Sequence[float]) -> float:
        return sum((2.0**label - 1.0) / math.log2(rank + 2) for rank, label in enumerate(labels))

    ideal_dcg = dcg(ideal)
    return 0.0 if ideal_dcg == 0 else float(dcg([relevance_values[index] for index in order]) / ideal_dcg)


def spearman_rho_from_rank_maps(first: Mapping[Any, int], second: Mapping[Any, int]) -> float | None:
    keys = sorted(set(first) & set(second))
    count = len(keys)
    if count < 2:
        return None
    difference_squared = sum((first[key] - second[key]) ** 2 for key in keys)
    return float(1.0 - (6.0 * difference_squared) / (count * (count * count - 1)))


def kendall_tau_from_rank_maps(first: Mapping[Any, int], second: Mapping[Any, int]) -> float | None:
    keys = sorted(set(first) & set(second))
    if len(keys) < 2:
        return None
    concordant = 0
    discordant = 0
    for left, right in combinations(keys, 2):
        product = (first[left] - first[right]) * (second[left] - second[right])
        if product > 0:
            concordant += 1
        elif product < 0:
            discordant += 1
    total = concordant + discordant
    return None if total == 0 else float((concordant - discordant) / total)


def topk_overlap_at_k(first: Sequence[Any], second: Sequence[Any], k: int) -> float:
    if k <= 0:
        return 0.0
    effective_k = min(k, len(first), len(second))
    if effective_k == 0:
        return 0.0
    return float(len(set(first[:effective_k]) & set(second[:effective_k])) / effective_k)


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


def rank_map(order: Sequence[Any]) -> dict[Any, int]:
    return {item: rank for rank, item in enumerate(order)}


def normalize_record(record: Mapping[str, Any]) -> tuple[QueryObservations | None, list[str]]:
    errors: list[str] = []
    raw_permutations = record.get("permutations", [])
    if not isinstance(raw_permutations, list) or len(raw_permutations) < 2:
        return None, ["fewer_than_two_permutations"]

    candidate_set: set[int] | None = None
    canonical_relevance: dict[int, int] | None = None
    observations: list[PermutationObservation] = []
    for permutation_index, permutation in enumerate(raw_permutations):
        input_data = permutation.get("input", {})
        output_data = permutation.get("output_ranking", {})
        try:
            input_order = tuple(int(value) for value in input_data.get("candidate_indices", []))
            output_order = tuple(int(value) for value in output_data.get("candidate_indices", []))
            relevance_values = tuple(int(value) for value in input_data.get("relevance", []))
        except (TypeError, ValueError):
            errors.append(f"permutation_{permutation_index}_contains_non_integer_values")
            continue
        if not input_order or not output_order:
            errors.append(f"permutation_{permutation_index}_is_empty")
            continue
        if len(input_order) != len(set(input_order)):
            errors.append(f"permutation_{permutation_index}_input_has_duplicates")
        if len(output_order) != len(set(output_order)):
            errors.append(f"permutation_{permutation_index}_output_has_duplicates")
        if len(relevance_values) != len(input_order):
            errors.append(f"permutation_{permutation_index}_relevance_length_mismatch")
            continue
        local_set = set(input_order)
        if set(output_order) != local_set or len(output_order) != len(input_order):
            errors.append(f"permutation_{permutation_index}_output_candidate_set_mismatch")
        if candidate_set is None:
            candidate_set = local_set
        elif local_set != candidate_set:
            errors.append(f"permutation_{permutation_index}_input_candidate_set_mismatch")
        relevance = dict(zip(input_order, relevance_values))
        if canonical_relevance is None:
            canonical_relevance = relevance
        elif relevance != canonical_relevance:
            errors.append(f"permutation_{permutation_index}_relevance_mismatch")
        observations.append(
            PermutationObservation(
                input_order=input_order,
                output_order=output_order,
                relevance=relevance,
            )
        )

    if errors:
        return None, sorted(set(errors))
    assert candidate_set is not None
    declared_length = record.get("list_length", record.get("num_items"))
    if declared_length is not None and int(declared_length) != len(candidate_set):
        return None, ["declared_list_length_mismatch"]
    return (
        QueryObservations(
            sample_index=record.get("sample_index"),
            user_id=str(record.get("user_id", "")),
            split=record.get("split"),
            candidates=tuple(sorted(candidate_set)),
            observations=tuple(observations),
        ),
        [],
    )


def _observation_effectiveness(
    observation: PermutationObservation,
    top_k: Sequence[int],
) -> dict[str, float]:
    relevance = [observation.relevance[candidate] for candidate in observation.input_order]
    output_ranks = observation.output_ranks
    scores = [float(len(observation.output_order) - output_ranks[candidate]) for candidate in observation.input_order]
    metrics: dict[str, float] = {}
    for k in top_k:
        metrics[f"hr@{k}"] = hr_at_k(scores, relevance, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(scores, relevance, k)
    return metrics


def evaluate_permutation_effectiveness(
    permutation: dict[str, Any],
    top_k: Sequence[int],
) -> dict[str, float]:
    normalized, errors = normalize_record({"permutations": [permutation, permutation]})
    if errors or normalized is None:
        return {}
    return _observation_effectiveness(normalized.observations[0], top_k)


def pairwise_preference_probabilities(
    items: Sequence[int],
    output_maps: Sequence[Mapping[int, int]],
) -> dict[tuple[int, int], float]:
    probabilities: dict[tuple[int, int], float] = {}
    for left, right in combinations(items, 2):
        preferences = [1.0 if output_map[left] < output_map[right] else 0.0 for output_map in output_maps]
        value = float(sum(preferences) / len(preferences))
        probabilities[(left, right)] = value
        probabilities[(right, left)] = 1.0 - value
    return probabilities


def position_bucket(position: int, list_length: int, bucket_count: int = 3) -> int:
    if bucket_count < 1:
        raise ValueError("position bucket count must be positive.")
    if not 0 <= position < list_length:
        raise ValueError("position is outside the candidate list.")
    return min(bucket_count - 1, int((position * bucket_count) / list_length))


def position_bucket_boundaries(list_length: int, bucket_count: int = 3) -> list[dict[str, int]]:
    if list_length < 1 or bucket_count < 1:
        raise ValueError("list length and position bucket count must be positive.")
    boundaries = []
    for bucket in range(bucket_count):
        positions = [
            position
            for position in range(list_length)
            if position_bucket(position, list_length, bucket_count) == bucket
        ]
        if positions:
            boundaries.append(
                {"bucket": bucket, "start": min(positions), "end": max(positions), "size": len(positions)}
            )
    return boundaries


def pairwise_preference_instability(
    items: Sequence[int],
    output_maps: Sequence[Mapping[int, int]],
    input_maps: Sequence[Mapping[int, int]],
    *,
    bucket_count: int = 3,
    minimum_bucket_observations: int = 1,
) -> tuple[float | None, dict[str, float | int]]:
    if minimum_bucket_observations < 1:
        raise ValueError("minimum_bucket_observations must be positive.")
    pair_deltas: list[float] = []
    supported_bucket_pairs = 0
    total_bucket_pairs = 0
    for left, right in combinations(items, 2):
        bucket_preferences: dict[tuple[int, int], list[float]] = {}
        for output_map, input_map in zip(output_maps, input_maps):
            list_length = len(input_map)
            bucket_pair = (
                position_bucket(input_map[left], list_length, bucket_count),
                position_bucket(input_map[right], list_length, bucket_count),
            )
            preference = 1.0 if output_map[left] < output_map[right] else 0.0
            bucket_preferences.setdefault(bucket_pair, []).append(preference)
        total_bucket_pairs += len(bucket_preferences)
        bucket_probabilities = [
            float(sum(values) / len(values))
            for values in bucket_preferences.values()
            if len(values) >= minimum_bucket_observations
        ]
        supported_bucket_pairs += len(bucket_probabilities)
        if len(bucket_probabilities) >= 2:
            pair_deltas.append(max(bucket_probabilities) - min(bucket_probabilities))
    total_pairs = math.comb(len(items), 2)
    return (
        mean(pair_deltas),
        {
            "eligible_candidate_pairs": len(pair_deltas),
            "total_candidate_pairs": total_pairs,
            "candidate_pair_coverage": float(len(pair_deltas) / total_pairs) if total_pairs else 0.0,
            "supported_bucket_pairs": supported_bucket_pairs,
            "observed_bucket_pairs": total_bucket_pairs,
        },
    )


def preference_disagreement(
    order: Sequence[int],
    pairwise_probabilities: Mapping[tuple[int, int], float],
) -> float:
    return float(
        sum(
            1.0 - pairwise_probabilities[(left, right)]
            for left_index, left in enumerate(order)
            for right in order[left_index + 1 :]
        )
    )


def _swap_disagreement_delta(
    order: Sequence[int],
    left_position: int,
    right_position: int,
    pairwise_probabilities: Mapping[tuple[int, int], float],
) -> float:
    left = order[left_position]
    right = order[right_position]
    current = 1.0 - pairwise_probabilities[(left, right)]
    proposed = 1.0 - pairwise_probabilities[(right, left)]
    for middle in order[left_position + 1 : right_position]:
        current += 1.0 - pairwise_probabilities[(left, middle)]
        current += 1.0 - pairwise_probabilities[(middle, right)]
        proposed += 1.0 - pairwise_probabilities[(right, middle)]
        proposed += 1.0 - pairwise_probabilities[(middle, left)]
    return float(proposed - current)


def approximate_global_order(
    items: Sequence[int],
    pairwise_probabilities: Mapping[tuple[int, int], float],
    *,
    tolerance: float = 1e-12,
) -> tuple[list[int], float, float, int]:
    order = sorted(
        items,
        key=lambda item: (
            -sum(pairwise_probabilities[(item, other)] for other in items if other != item),
            item,
        ),
    )
    initial_disagreement = preference_disagreement(order, pairwise_probabilities)
    swaps = 0
    while True:
        best_delta = 0.0
        best_swap: tuple[int, int] | None = None
        for left_position in range(len(order) - 1):
            for right_position in range(left_position + 1, len(order)):
                delta = _swap_disagreement_delta(
                    order,
                    left_position,
                    right_position,
                    pairwise_probabilities,
                )
                if delta < best_delta - tolerance:
                    best_delta = delta
                    best_swap = (left_position, right_position)
        if best_swap is None:
            break
        left_position, right_position = best_swap
        order[left_position], order[right_position] = order[right_position], order[left_position]
        swaps += 1
    return order, initial_disagreement, preference_disagreement(order, pairwise_probabilities), swaps


def global_preference_inconsistency(
    items: Sequence[int],
    pairwise_probabilities: Mapping[tuple[int, int], float],
) -> tuple[float | None, dict[str, float | int]]:
    pair_count = math.comb(len(items), 2)
    if pair_count == 0:
        return None, {}
    _, initial_disagreement, final_disagreement, swaps = approximate_global_order(items, pairwise_probabilities)
    return (
        float(final_disagreement / pair_count),
        {
            "initial_disagreement": initial_disagreement,
            "optimized_disagreement": final_disagreement,
            "local_swaps": swaps,
        },
    )


def evaluate_query_preference_validity(
    query: QueryObservations,
    *,
    bucket_count: int,
    minimum_bucket_observations: int,
) -> tuple[dict[str, float], dict[str, float | int]]:
    output_maps = [observation.output_ranks for observation in query.observations]
    input_maps = [observation.input_positions for observation in query.observations]
    pairwise_probabilities = pairwise_preference_probabilities(query.candidates, output_maps)
    ppi, ppi_diagnostics = pairwise_preference_instability(
        query.candidates,
        output_maps,
        input_maps,
        bucket_count=bucket_count,
        minimum_bucket_observations=minimum_bucket_observations,
    )
    gpi, gpi_diagnostics = global_preference_inconsistency(query.candidates, pairwise_probabilities)
    metrics = {}
    if ppi is not None:
        metrics["PPI"] = ppi
    if gpi is not None:
        metrics["GPI"] = gpi
    return metrics, {**ppi_diagnostics, **{f"GPI_{key}": value for key, value in gpi_diagnostics.items()}}


def evaluate_query_listwise_stability(
    query: QueryObservations,
    top_k: Sequence[int],
) -> dict[str, float]:
    output_orders = [observation.output_order for observation in query.observations]
    output_maps = [observation.output_ranks for observation in query.observations]
    kendalls: list[float] = []
    spearmans: list[float] = []
    topk_values: dict[int, list[float]] = {k: [] for k in top_k}
    for (first_order, first_map), (second_order, second_map) in combinations(zip(output_orders, output_maps), 2):
        if (value := kendall_tau_from_rank_maps(first_map, second_map)) is not None:
            kendalls.append(value)
        if (value := spearman_rho_from_rank_maps(first_map, second_map)) is not None:
            spearmans.append(value)
        for k in top_k:
            topk_values[k].append(topk_overlap_at_k(first_order, second_order, k))
    metrics: dict[str, float] = {}
    if (value := mean(kendalls)) is not None:
        metrics["kendall_tau"] = value
    if (value := mean(spearmans)) is not None:
        metrics["spearman_rho"] = value
    for k, values in topk_values.items():
        if (value := mean(values)) is not None:
            metrics[f"topk_overlap@{k}"] = value
    return metrics


def evaluate_record_robustness(record: dict[str, Any], top_k: Sequence[int]) -> dict[str, float]:
    query, errors = normalize_record(record)
    return {} if errors or query is None else evaluate_query_listwise_stability(query, top_k)


def evaluate_record_validity(
    record: dict[str, Any],
    *,
    position_buckets: int = 3,
    minimum_bucket_observations: int = 1,
) -> dict[str, float]:
    query, errors = normalize_record(record)
    if errors or query is None:
        return {}
    metrics, _ = evaluate_query_preference_validity(
        query,
        bucket_count=position_buckets,
        minimum_bucket_observations=minimum_bucket_observations,
    )
    return metrics


def _aggregate_query_metrics(metrics_list: Sequence[Mapping[str, float]]) -> dict[str, float]:
    aggregated: dict[str, list[float]] = {}
    for metrics in metrics_list:
        for key, value in metrics.items():
            aggregated.setdefault(key, []).append(float(value))
    return {key: float(sum(values) / len(values)) for key, values in aggregated.items() if values}


def aggregate_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    return _aggregate_query_metrics(metrics_list)


def aggregate_validity(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    return _aggregate_query_metrics(metrics_list)


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile of an empty sequence.")
    if len(ordered) == 1:
        return float(ordered[0])
    location = probability * (len(ordered) - 1)
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return float(ordered[lower])
    fraction = location - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def bootstrap_uncertainty(
    metrics_list: Sequence[Mapping[str, float]],
    *,
    confidence_level: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one.")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative.")
    keys = sorted(set().union(*(set(metrics) for metrics in metrics_list))) if metrics_list else []
    output: dict[str, dict[str, float | int]] = {}
    alpha = 1.0 - confidence_level
    for key_index, key in enumerate(keys):
        values = [float(metrics[key]) for metrics in metrics_list if key in metrics]
        if not values:
            continue
        details: dict[str, float | int] = {
            "num_records": len(values),
            "std": standard_deviation(values) or 0.0,
        }
        if bootstrap_samples:
            generator = random.Random(seed + key_index * 1009)
            bootstrap_means = [
                float(sum(values[generator.randrange(len(values))] for _ in values) / len(values))
                for _ in range(bootstrap_samples)
            ]
            details["ci_low"] = _percentile(bootstrap_means, alpha / 2.0)
            details["ci_high"] = _percentile(bootstrap_means, 1.0 - alpha / 2.0)
            details["confidence_level"] = confidence_level
        output[key] = details
    return output


def _position_exposure(
    queries: Sequence[QueryObservations],
    top_k: Sequence[int],
) -> dict[str, Any]:
    totals: dict[tuple[int, int], list[int]] = {}
    exposed: dict[tuple[int, int, int], list[int]] = {}
    for query in queries:
        length = query.list_length
        totals.setdefault((length, 0), [0] * length)
        for k in top_k:
            exposed.setdefault((length, k, 0), [0] * length)
        for observation in query.observations:
            output_ranks = observation.output_ranks
            for candidate, position in observation.input_positions.items():
                totals[(length, 0)][position] += 1
                for k in top_k:
                    if output_ranks[candidate] < min(k, length):
                        exposed[(length, k, 0)][position] += 1
    output: dict[str, Any] = {}
    for length in sorted({query.list_length for query in queries}):
        observations = totals[(length, 0)]
        length_output: dict[str, Any] = {}
        for k in top_k:
            counts = exposed[(length, k, 0)]
            probabilities = [float(count / total) if total else 0.0 for count, total in zip(counts, observations)]
            length_output[f"top@{k}"] = {
                "exposure": probabilities,
                "observations": list(observations),
                "exposed": list(counts),
            }
        output[str(length)] = length_output
    return output


def write_position_exposure_csv(exposure: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("list_length", "cutoff", "input_position", "exposure", "observations", "exposed"),
        )
        writer.writeheader()
        for length, cutoffs in exposure.items():
            for cutoff, values in cutoffs.items():
                for position, probability in enumerate(values["exposure"]):
                    writer.writerow(
                        {
                            "list_length": int(length),
                            "cutoff": int(str(cutoff).split("@", 1)[1]),
                            "input_position": position,
                            "exposure": probability,
                            "observations": values["observations"][position],
                            "exposed": values["exposed"][position],
                        }
                    )


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


def evaluate_with_per_record(
    results: Sequence[RankingResult] | Sequence[dict[str, Any]],
    *,
    top_k: Sequence[int] = (5, 10),
    position_buckets: int = 3,
    minimum_bucket_observations: int = 1,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 1000,
    seed: int = 42,
    show_progress: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not top_k or any(k <= 0 for k in top_k):
        raise ValueError("top_k must contain positive cutoffs.")
    values = list(results)
    records = (
        ranking_results_to_records(values)  # type: ignore[arg-type]
        if values and isinstance(values[0], RankingResult)
        else values
    )

    queries: list[QueryObservations] = []
    skipped_reasons: Counter[str] = Counter()
    record_iterator: Any = records
    if show_progress:
        from tqdm.auto import tqdm

        record_iterator = tqdm(records, desc="[Evaluation] Records", unit="record", dynamic_ncols=True)
    for record in record_iterator:
        query, errors = normalize_record(record)
        if query is None:
            skipped_reasons.update(errors or ["unknown_validation_error"])
        else:
            queries.append(query)

    effectiveness_rows: list[dict[str, float]] = []
    preference_rows: list[dict[str, float]] = []
    preference_diagnostic_rows: list[dict[str, float | int]] = []
    stability_rows: list[dict[str, float]] = []
    per_record: list[dict[str, Any]] = []
    for query in queries:
        permutation_effectiveness = [
            _observation_effectiveness(observation, top_k) for observation in query.observations
        ]
        effectiveness = _aggregate_query_metrics(permutation_effectiveness)
        preference_validity, preference_diagnostics = evaluate_query_preference_validity(
            query,
            bucket_count=position_buckets,
            minimum_bucket_observations=minimum_bucket_observations,
        )
        listwise_stability = evaluate_query_listwise_stability(query, top_k)
        effectiveness_rows.append(effectiveness)
        preference_rows.append(preference_validity)
        preference_diagnostic_rows.append(preference_diagnostics)
        stability_rows.append(listwise_stability)
        per_record.append(
            {
                "sample_index": query.sample_index,
                "user_id": query.user_id,
                "split": query.split,
                "list_length": query.list_length,
                "num_permutations": len(query.observations),
                "effectiveness": effectiveness,
                "preference_validity": preference_validity,
                "preference_diagnostics": preference_diagnostics,
                "listwise_stability": listwise_stability,
            }
        )

    list_lengths = sorted({query.list_length for query in queries})
    permutation_counts = [len(query.observations) for query in queries]
    report = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "num_records": len(queries),
        "num_permutations": sum(permutation_counts),
        "protocol": {
            "input_records": len(records),
            "evaluated_records": len(queries),
            "skipped_records": len(records) - len(queries),
            "skipped_reasons": dict(sorted(skipped_reasons.items())),
            "list_lengths": list_lengths,
            "permutations_per_record": {
                "minimum": min(permutation_counts, default=0),
                "maximum": max(permutation_counts, default=0),
                "mean": mean(permutation_counts) or 0.0,
            },
            "position_buckets": position_buckets,
            "position_bucket_boundaries": {
                str(length): position_bucket_boundaries(length, position_buckets) for length in list_lengths
            },
            "minimum_bucket_observations": minimum_bucket_observations,
            "statistical_unit": "query",
            "confidence_level": confidence_level,
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
        },
        "effectiveness": _aggregate_query_metrics(effectiveness_rows),
        "preference_validity": _aggregate_query_metrics(preference_rows),
        "preference_diagnostics": {
            "mean_candidate_pair_coverage": mean(
                [float(row["candidate_pair_coverage"]) for row in preference_diagnostic_rows]
            )
            or 0.0,
            "eligible_candidate_pairs": sum(int(row["eligible_candidate_pairs"]) for row in preference_diagnostic_rows),
            "total_candidate_pairs": sum(int(row["total_candidate_pairs"]) for row in preference_diagnostic_rows),
            "supported_bucket_pairs": sum(int(row["supported_bucket_pairs"]) for row in preference_diagnostic_rows),
            "observed_bucket_pairs": sum(int(row["observed_bucket_pairs"]) for row in preference_diagnostic_rows),
        },
        "listwise_stability": _aggregate_query_metrics(stability_rows),
        "position_exposure": _position_exposure(queries, top_k),
        "uncertainty": {
            "effectiveness": bootstrap_uncertainty(
                effectiveness_rows,
                confidence_level=confidence_level,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            ),
            "preference_validity": bootstrap_uncertainty(
                preference_rows,
                confidence_level=confidence_level,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 100_003,
            ),
            "listwise_stability": bootstrap_uncertainty(
                stability_rows,
                confidence_level=confidence_level,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 200_003,
            ),
        },
    }
    return report, per_record


def evaluate(
    results: Sequence[RankingResult] | Sequence[dict[str, Any]],
    *,
    top_k: Sequence[int] = (5, 10),
    position_buckets: int = 3,
    minimum_bucket_observations: int = 1,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 1000,
    seed: int = 42,
    show_progress: bool = False,
) -> dict[str, Any]:
    report, _ = evaluate_with_per_record(
        results,
        top_k=top_k,
        position_buckets=position_buckets,
        minimum_bucket_observations=minimum_bucket_observations,
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        show_progress=show_progress,
    )
    return report


__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "PermutationObservation",
    "QueryObservations",
    "aggregate_metrics",
    "aggregate_validity",
    "approximate_global_order",
    "bootstrap_uncertainty",
    "evaluate",
    "evaluate_permutation_effectiveness",
    "evaluate_query_listwise_stability",
    "evaluate_query_preference_validity",
    "evaluate_record_robustness",
    "evaluate_record_validity",
    "evaluate_with_per_record",
    "global_preference_inconsistency",
    "hr_at_k",
    "kendall_tau_from_rank_maps",
    "load_json_or_jsonl",
    "load_ranked_lists",
    "ndcg_at_k",
    "normalize_record",
    "pairwise_preference_instability",
    "pairwise_preference_probabilities",
    "position_bucket",
    "position_bucket_boundaries",
    "preference_disagreement",
    "ranking_results_to_records",
    "spearman_rho_from_rank_maps",
    "topk_overlap_at_k",
    "write_position_exposure_csv",
]

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from invarirank.contracts import RankedItem, RankingResult, RankingSample, Reranker

from experiments.reranking.parsers import validate_ranking


def ranking_sample(
    user_record: Mapping[str, Any],
    user_history: list[Mapping[str, Any]] | None = None,
) -> RankingSample:
    truth = {str(item_id) for item_id in user_record.get("ground_truth_item_ids", [])}
    candidates = []
    for candidate_index, (_rank_key, candidate) in enumerate(
        sorted(user_record["candidates"].items(), key=lambda item: int(item[0]))
    ):
        value = dict(candidate)
        value.setdefault("item_id", str(candidate.get("item_id", candidate_index)))
        value.setdefault("title", candidate_title(value))
        value.setdefault("relevance", 1 if str(value["item_id"]) in truth else 0)
        candidates.append(value)
    history = [dict(item) for item in (user_history or [])]
    for item in history:
        item.setdefault("title", candidate_title(item))
    return RankingSample(
        user_id=str(user_record.get("user_id", "")),
        history=history,
        candidates=candidates,
    )


def candidate_export_result(
    result: RankingResult,
    *,
    reranker: str,
    model: str,
    scoring: str,
    prompt: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    reranked = [item.item_id for item in result.items]
    valid_ids = [item_id_for_input(result.items, index) for index in result.permutation]
    parse_error = result.metadata.get("parse_error")
    if parse_error is None and result.metadata.get("parse_errors"):
        parse_error = "; ".join(str(value) for value in result.metadata["parse_errors"])
    output = validate_ranking(reranked, valid_ids, parse_error=parse_error)
    output.update(
        {
            "reranker": reranker,
            "model": model,
            "scoring": scoring,
            "prompt": prompt,
            "output_backend": result.metadata.get("output_backend"),
            "prompt_family": result.metadata.get("prompt_family"),
            "prompt_version": result.metadata.get("prompt_version"),
            "forward_passes": result.metadata.get("forward_passes", 1),
            "scores": {item.item_id: item.score for item in result.items},
            "input_candidate_item_ids": valid_ids,
            "metadata": dict(result.metadata),
            "elapsed_seconds": elapsed_seconds,
            "parser_repair_applied": bool(
                result.metadata.get("repaired") or result.metadata.get("repaired_outputs", 0)
            ),
        }
    )
    return output


def rank_many(
    scorer: Reranker,
    requests: Sequence[tuple[RankingSample, Sequence[int] | None]],
    *,
    batch_size: int,
) -> list[RankingResult]:
    batched = getattr(scorer, "rank_many", None)
    if callable(batched) and batch_size > 1:
        return list(batched(requests, batch_size=batch_size))
    return [scorer.rank(sample, permutation=permutation) for sample, permutation in requests]


def normalize_method_requests(
    samples: Sequence[
        RankingSample | Mapping[str, Any] | tuple[RankingSample | Mapping[str, Any], Sequence[int] | None]
    ],
    permutations: Sequence[Sequence[int] | None] | None,
) -> list[tuple[RankingSample | Mapping[str, Any], Sequence[int] | None]]:
    values = list(samples)
    if permutations is not None:
        if len(permutations) != len(values):
            raise ValueError("permutations must contain one entry per sample.")
        if any(isinstance(value, tuple) for value in values):
            raise ValueError("Do not combine request tuples with the permutations argument.")
        return list(zip(values, permutations, strict=True))  # type: ignore[arg-type]
    return [(value[0], value[1]) if isinstance(value, tuple) else (value, None) for value in values]


def sample(value: RankingSample | Mapping[str, Any]) -> RankingSample:
    return value if isinstance(value, RankingSample) else RankingSample.from_dict(value)


def permutation(value: Sequence[int] | None, count: int) -> list[int]:
    result = list(range(count)) if value is None else [int(index) for index in value]
    if len(result) != count or set(result) != set(range(count)):
        raise ValueError(f"permutation must contain every candidate index from 0 to {count - 1} exactly once.")
    return result


def request_seed(seed: int, ranking_sample: RankingSample, input_permutation: Sequence[int]) -> int:
    payload = {
        "seed": int(seed),
        "user_id": ranking_sample.user_id,
        "split": ranking_sample.split,
        "candidate_ids": [
            candidate_id(candidate, index) for index, candidate in enumerate(ranking_sample.candidates)
        ],
        "permutation": [int(value) for value in input_permutation],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def with_metadata(result: RankingResult, method: str, forward_passes: int) -> RankingResult:
    return replace_metadata(result, {**result.metadata, "method": method, "forward_passes": forward_passes})


def replace_metadata(result: RankingResult, metadata: Mapping[str, Any]) -> RankingResult:
    return RankingResult(
        user_id=result.user_id,
        items=result.items,
        permutation=result.permutation,
        split=result.split,
        metadata=dict(metadata),
    )


def combined_metadata(rankings: Sequence[RankingResult]) -> dict[str, Any]:
    if not rankings:
        return {}
    metadata = [dict(result.metadata) for result in rankings]
    combined: dict[str, Any] = {}
    for key in ("output_backend", "prompt_family", "prompt_version", "scoring", "prompt"):
        values = {str(value[key]) for value in metadata if value.get(key) is not None}
        if len(values) == 1:
            combined[key] = values.pop()
    combined["raw_outputs"] = [value["raw_output"] for value in metadata if "raw_output" in value]
    combined["parse_statuses"] = [value["parse_status"] for value in metadata if "parse_status" in value]
    combined["parse_errors"] = [value["parse_error"] for value in metadata if value.get("parse_error")]
    combined["repaired_outputs"] = sum(bool(value.get("repaired")) for value in metadata)
    combined["generated_tokens"] = sum(int(value.get("generated_tokens", 0)) for value in metadata)
    combined["latency_seconds"] = sum(float(value.get("latency_seconds", 0.0)) for value in metadata)
    return combined


def borda_aggregate(
    ranking_sample: RankingSample,
    rankings: Sequence[RankingResult],
    input_permutation: Sequence[int],
    *,
    method: str,
    forward_passes: int,
) -> RankingResult:
    if not rankings:
        raise ValueError("Borda aggregation requires at least one ranking.")
    count = len(ranking_sample.candidates)
    points = {index: 0.0 for index in range(count)}
    raw_scores = {index: 0.0 for index in range(count)}
    for result in rankings:
        for rank, item in enumerate(result.items):
            points[item.candidate_index] += count - rank
            raw_scores[item.candidate_index] += item.score
    input_positions = {candidate: position for position, candidate in enumerate(input_permutation)}
    order = sorted(
        range(count),
        key=lambda candidate: (-points[candidate], -raw_scores[candidate], input_positions[candidate]),
    )
    return ranking_from_order(
        ranking_sample,
        input_permutation,
        order,
        scores=points,
        metadata={
            "method": method,
            "forward_passes": forward_passes,
            "aggregation": "borda",
            "num_rankings": len(rankings),
            **combined_metadata(rankings),
        },
    )


def ranking_from_order(
    ranking_sample: RankingSample,
    input_permutation: Sequence[int],
    order: Sequence[int],
    *,
    scores: Mapping[int, float],
    metadata: Mapping[str, Any],
) -> RankingResult:
    input_positions = {candidate: position for position, candidate in enumerate(input_permutation)}
    return RankingResult(
        user_id=ranking_sample.user_id,
        items=tuple(
            RankedItem(
                candidate_index=index,
                item_id=candidate_id(ranking_sample.candidates[index], index),
                score=float(scores[index]),
                input_position=input_positions[index],
                relevance=relevance(ranking_sample.candidates[index]),
                candidate=dict(ranking_sample.candidates[index]),
            )
            for index in order
        ),
        permutation=tuple(input_permutation),
        split=ranking_sample.split,
        metadata=dict(metadata),
    )


def rebase_result(
    ranking_sample: RankingSample,
    result: RankingResult,
    input_permutation: Sequence[int],
) -> RankingResult:
    return ranking_from_order(
        ranking_sample,
        input_permutation,
        [item.candidate_index for item in result.items],
        scores={item.candidate_index: item.score for item in result.items},
        metadata=result.metadata,
    )


def candidate_title(candidate: Mapping[str, Any]) -> str:
    for key in ("title", "name", "main_category", "item_id"):
        value = candidate.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return "unknown item"


def candidate_id(candidate: Mapping[str, Any], fallback: int) -> str:
    for key in ("item_id", "id", "asin", "movie_id"):
        if key in candidate:
            return str(candidate[key])
    return str(fallback)


def item_id_for_input(items: Sequence[RankedItem], candidate_index: int) -> str:
    for item in items:
        if item.candidate_index == candidate_index:
            return item.item_id
    return str(candidate_index)


def relevance(candidate: Mapping[str, Any]) -> int | None:
    value = candidate.get("relevance")
    return None if value is None else int(value)

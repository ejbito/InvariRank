"""Domain-neutral adapters and controlled input-order experiments."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from numbers import Real
from typing import Any

from .framework import RankedItem, RankingResult, RankingSample, Reranker

SampleLike = RankingSample | Mapping[str, Any]
Permutation = Sequence[int] | None
RankRequest = SampleLike | tuple[SampleLike, Permutation]
ScoreFunction = Callable[[RankingSample, list[Mapping[str, Any]]], Sequence[float]]
OrderFunction = Callable[[RankingSample, list[Mapping[str, Any]]], Sequence[str]]
BatchScoreFunction = Callable[
    [list[RankingSample], list[list[Mapping[str, Any]]]],
    Sequence[Sequence[float]],
]
BatchOrderFunction = Callable[
    [list[RankingSample], list[list[Mapping[str, Any]]]],
    Sequence[Sequence[str]],
]


class CallableReranker(Reranker):
    """Adapt a score or item-ID order callback to the shared ranking contract."""

    def __init__(
        self,
        *,
        score_fn: ScoreFunction | None = None,
        order_fn: OrderFunction | None = None,
        batch_score_fn: BatchScoreFunction | None = None,
        batch_order_fn: BatchOrderFunction | None = None,
        higher_is_better: bool = True,
        method_name: str,
    ) -> None:
        if (score_fn is None) == (order_fn is None):
            raise ValueError("Provide exactly one of score_fn or order_fn.")
        if score_fn is None and batch_score_fn is not None:
            raise ValueError("batch_score_fn requires score_fn.")
        if order_fn is None and batch_order_fn is not None:
            raise ValueError("batch_order_fn requires order_fn.")
        if not method_name:
            raise ValueError("method_name must be non-empty.")
        self.score_fn = score_fn
        self.order_fn = order_fn
        self.batch_score_fn = batch_score_fn
        self.batch_order_fn = batch_order_fn
        self.higher_is_better = bool(higher_is_better)
        self.method_name = method_name

    @classmethod
    def from_scores(
        cls,
        score_fn: ScoreFunction,
        *,
        batch_score_fn: BatchScoreFunction | None = None,
        higher_is_better: bool = True,
        method_name: str = "custom_score_reranker",
    ) -> CallableReranker:
        if not callable(score_fn):
            raise TypeError("score_fn must be callable.")
        if batch_score_fn is not None and not callable(batch_score_fn):
            raise TypeError("batch_score_fn must be callable.")
        return cls(
            score_fn=score_fn,
            batch_score_fn=batch_score_fn,
            higher_is_better=higher_is_better,
            method_name=method_name,
        )

    @classmethod
    def from_order(
        cls,
        order_fn: OrderFunction,
        *,
        batch_order_fn: BatchOrderFunction | None = None,
        method_name: str = "custom_order_reranker",
    ) -> CallableReranker:
        if not callable(order_fn):
            raise TypeError("order_fn must be callable.")
        if batch_order_fn is not None and not callable(batch_order_fn):
            raise TypeError("batch_order_fn must be callable.")
        return cls(order_fn=order_fn, batch_order_fn=batch_order_fn, method_name=method_name)

    def rank(self, sample: SampleLike, *, permutation: Permutation = None) -> RankingResult:
        ranking_sample, resolved, ordered_items = _prepare_request(sample, permutation)
        if self.score_fn is not None:
            return self._result_from_scores(ranking_sample, resolved, self.score_fn(ranking_sample, ordered_items))
        if self.order_fn is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("No callback configured.")
        return self._result_from_order(ranking_sample, resolved, self.order_fn(ranking_sample, ordered_items))

    def rank_many(
        self,
        samples: Sequence[RankRequest],
        *,
        permutations: Sequence[Permutation] | None = None,
        batch_size: int = 8,
    ) -> list[RankingResult]:
        size = _positive_integer(batch_size, "batch_size")
        requests = _normalize_requests(samples, permutations)
        if self.batch_score_fn is None and self.batch_order_fn is None:
            return [self.rank(sample, permutation=permutation) for sample, permutation in requests]

        results = []
        for start in range(0, len(requests), size):
            prepared = [_prepare_request(sample, permutation) for sample, permutation in requests[start : start + size]]
            ranking_samples = [sample for sample, _, _ in prepared]
            ordered_batches = [items for _, _, items in prepared]
            if self.batch_score_fn is not None:
                rows = list(self.batch_score_fn(ranking_samples, ordered_batches))
                _validate_batch_row_count(rows, prepared)
                results.extend(
                    self._result_from_scores(sample, permutation, row)
                    for (sample, permutation, _), row in zip(prepared, rows, strict=True)
                )
            else:
                if self.batch_order_fn is None:  # pragma: no cover - constructor invariant
                    raise RuntimeError("No batch callback configured.")
                rows = list(self.batch_order_fn(ranking_samples, ordered_batches))
                _validate_batch_row_count(rows, prepared)
                results.extend(
                    self._result_from_order(sample, permutation, row)
                    for (sample, permutation, _), row in zip(prepared, rows, strict=True)
                )
        return results

    def _result_from_scores(
        self,
        sample: RankingSample,
        permutation: list[int],
        values: Sequence[float],
    ) -> RankingResult:
        scores = _validate_scores(values, len(permutation))
        canonical_scores = scores if self.higher_is_better else [-score for score in scores]
        items = [
            _ranked_item(sample, candidate_index, input_position, canonical_scores[input_position])
            for input_position, candidate_index in enumerate(permutation)
        ]
        items.sort(key=lambda item: (-item.score, item.input_position))
        return _callable_result(
            sample,
            permutation,
            items,
            self.method_name,
            "scores",
            {"higher_is_better": self.higher_is_better},
        )

    def _result_from_order(
        self,
        sample: RankingSample,
        permutation: list[int],
        values: Sequence[str],
    ) -> RankingResult:
        candidate_ids = [_required_item_id(candidate, index) for index, candidate in enumerate(sample.candidates)]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("Order callbacks require unique candidate item IDs.")
        order = [str(value) for value in values]
        _validate_order(order, candidate_ids)
        index_by_id = {item_id: index for index, item_id in enumerate(candidate_ids)}
        input_positions = {candidate: position for position, candidate in enumerate(permutation)}
        count = len(order)
        items = [
            _ranked_item(sample, index_by_id[item_id], input_positions[index_by_id[item_id]], float(count - rank))
            for rank, item_id in enumerate(order)
        ]
        return _callable_result(sample, permutation, items, self.method_name, "order", {})


class PermutationSuite:
    """Run reproducible controlled-input-order experiments on any Reranker."""

    def __init__(self, reranker: Reranker) -> None:
        if not isinstance(reranker, Reranker):
            raise TypeError("reranker must implement Reranker.")
        self.reranker = reranker

    def random(
        self,
        sample: SampleLike,
        count: int,
        *,
        seed: int = 42,
        include_identity: bool = True,
        batch_size: int = 8,
    ) -> list[RankingResult]:
        ranking_sample = _sample(sample)
        permutations = _random_permutations(
            len(ranking_sample.candidates),
            _positive_integer(count, "count"),
            seed=seed,
            include_identity=bool(include_identity),
        )
        return self._execute(ranking_sample, permutations, batch_size)

    def fixed(
        self,
        sample: SampleLike,
        *,
        item: int,
        position: int,
        count: int,
        seed: int = 42,
        batch_size: int = 8,
    ) -> list[RankingResult]:
        ranking_sample = _sample(sample)
        candidate_count = len(ranking_sample.candidates)
        item = _index(item, candidate_count, "item")
        position = _index(position, candidate_count, "position")
        permutations = _template_permutations(
            candidate_count,
            [item if current == position else None for current in range(candidate_count)],
            _positive_integer(count, "count"),
            seed,
        )
        return self._execute(ranking_sample, permutations, batch_size)

    def sweep(
        self,
        sample: SampleLike,
        *,
        item: int,
        repeats: int = 1,
        seed: int = 42,
        batch_size: int = 8,
    ) -> list[RankingResult]:
        ranking_sample = _sample(sample)
        candidate_count = len(ranking_sample.candidates)
        item = _index(item, candidate_count, "item")
        repeat_count = _positive_integer(repeats, "repeats")
        permutations = []
        for position in range(candidate_count):
            template = [item if current == position else None for current in range(candidate_count)]
            permutations.extend(_template_permutations(candidate_count, template, repeat_count, seed + position * 1009))
        return self._execute(ranking_sample, permutations, batch_size)

    def templates(
        self,
        sample: SampleLike,
        templates: Sequence[Sequence[int | None]],
        *,
        seed: int = 42,
        batch_size: int = 8,
    ) -> list[RankingResult]:
        ranking_sample = _sample(sample)
        if not templates:
            raise ValueError("templates must contain at least one template.")
        permutations = [
            _complete_template(len(ranking_sample.candidates), template, random.Random(seed + index * 1009))
            for index, template in enumerate(templates)
        ]
        return self._execute(ranking_sample, permutations, batch_size)

    def _execute(
        self,
        sample: RankingSample,
        permutations: Sequence[Sequence[int]],
        batch_size: int,
    ) -> list[RankingResult]:
        size = _positive_integer(batch_size, "batch_size")
        requests = [(sample, permutation) for permutation in permutations]
        return list(self.reranker.rank_many(requests, batch_size=size))


def _sample(sample: SampleLike) -> RankingSample:
    return sample if isinstance(sample, RankingSample) else RankingSample.from_dict(sample)


def _validate_permutation(permutation: Permutation, count: int) -> list[int]:
    resolved = (
        list(range(count)) if permutation is None else [_integer(value, "permutation index") for value in permutation]
    )
    if len(resolved) != count or set(resolved) != set(range(count)):
        raise ValueError(f"permutation must contain every candidate index from 0 to {count - 1} exactly once.")
    return resolved


def _prepare_request(
    sample: SampleLike, permutation: Permutation
) -> tuple[RankingSample, list[int], list[Mapping[str, Any]]]:
    ranking_sample = _sample(sample)
    resolved = _validate_permutation(permutation, len(ranking_sample.candidates))
    return ranking_sample, resolved, [ranking_sample.candidates[index] for index in resolved]


def _normalize_requests(
    samples: Sequence[RankRequest],
    permutations: Sequence[Permutation] | None,
) -> list[tuple[SampleLike, Permutation]]:
    values = list(samples)
    if permutations is not None:
        if len(permutations) != len(values):
            raise ValueError("permutations must contain one entry per sample.")
        if any(isinstance(value, tuple) for value in values):
            raise ValueError("Do not combine request tuples with the permutations argument.")
        return list(zip(values, permutations, strict=True))  # type: ignore[arg-type]
    requests = []
    for value in values:
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError("Rank request tuples must contain (sample, permutation).")
            requests.append((value[0], value[1]))
        else:
            requests.append((value, None))
    return requests


def _validate_scores(values: Sequence[float], expected: int) -> list[float]:
    try:
        scores = list(values)
    except TypeError as exc:
        raise TypeError("Score callback must return a sequence of numeric values.") from exc
    if len(scores) != expected:
        raise ValueError(f"Score callback returned {len(scores)} scores for {expected} candidates.")
    validated = []
    for index, score in enumerate(scores):
        if isinstance(score, bool) or not isinstance(score, Real):
            raise TypeError(f"Score at position {index} must be numeric.")
        value = float(score)
        if not math.isfinite(value):
            raise ValueError(f"Score at position {index} must be finite.")
        validated.append(value)
    return validated


def _validate_order(order: list[str], expected_ids: list[str]) -> None:
    if len(order) != len(expected_ids):
        raise ValueError(f"Order callback returned {len(order)} item IDs for {len(expected_ids)} candidates.")
    duplicates = sorted({item_id for item_id in order if order.count(item_id) > 1})
    unknown = sorted(set(order) - set(expected_ids))
    missing = sorted(set(expected_ids) - set(order))
    if duplicates or unknown or missing:
        details = []
        if unknown:
            details.append(f"unknown IDs: {unknown}")
        if duplicates:
            details.append(f"duplicate IDs: {duplicates}")
        if missing:
            details.append(f"missing IDs: {missing}")
        raise ValueError("Invalid order callback result (" + "; ".join(details) + ").")


def _validate_batch_row_count(rows: Sequence[Any], prepared: Sequence[Any]) -> None:
    if len(rows) != len(prepared):
        raise ValueError(f"Batch callback returned {len(rows)} rows for {len(prepared)} requests.")


def _ranked_item(
    sample: RankingSample,
    candidate_index: int,
    input_position: int,
    score: float,
) -> RankedItem:
    candidate = sample.candidates[candidate_index]
    relevance = candidate.get("relevance")
    return RankedItem(
        candidate_index=candidate_index,
        item_id=_candidate_id(candidate, candidate_index),
        score=float(score),
        input_position=input_position,
        relevance=None if relevance is None else int(relevance),
        candidate=dict(candidate),
    )


def _callable_result(
    sample: RankingSample,
    permutation: list[int],
    items: Sequence[RankedItem],
    method_name: str,
    callback_type: str,
    metadata: Mapping[str, Any],
) -> RankingResult:
    return RankingResult(
        user_id=sample.user_id,
        items=tuple(items),
        permutation=tuple(permutation),
        split=sample.split,
        metadata={"method": method_name, "output_backend": "callable", "callback_type": callback_type, **metadata},
    )


def _candidate_id(candidate: Mapping[str, Any], fallback: int) -> str:
    for key in ("item_id", "id", "asin", "movie_id"):
        if key in candidate:
            return str(candidate[key])
    return str(fallback)


def _required_item_id(candidate: Mapping[str, Any], index: int) -> str:
    for key in ("item_id", "id", "asin", "movie_id"):
        if key in candidate:
            value = str(candidate[key])
            if value:
                return value
    raise ValueError(f"Order callbacks require an item ID for candidate {index}.")


def _random_permutations(count: int, requested: int, *, seed: int, include_identity: bool) -> list[list[int]]:
    identity = tuple(range(count))
    maximum = math.factorial(count) - (0 if include_identity else 1)
    if requested > maximum:
        raise ValueError(f"Requested {requested} unique permutations, but only {maximum} exist for {count} candidates.")
    generator = random.Random(seed)
    selected: list[tuple[int, ...]] = [identity] if include_identity else []
    seen = set(selected)
    while len(selected) < requested:
        value = list(identity)
        generator.shuffle(value)
        permutation = tuple(value)
        if permutation != identity and permutation not in seen:
            selected.append(permutation)
            seen.add(permutation)
    return [list(value) for value in selected[:requested]]


def _template_permutations(
    count: int,
    template: Sequence[int | None],
    requested: int,
    seed: int,
) -> list[list[int]]:
    fixed_count = sum(value is not None for value in template)
    maximum = math.factorial(count - fixed_count)
    if requested > maximum:
        raise ValueError(f"Requested {requested} unique template completions, but only {maximum} exist.")
    generator = random.Random(seed)
    selected = []
    seen = set()
    while len(selected) < requested:
        permutation = _complete_template(count, template, generator)
        key = tuple(permutation)
        if key not in seen:
            selected.append(permutation)
            seen.add(key)
    return selected


def _complete_template(count: int, template: Sequence[int | None], generator: random.Random) -> list[int]:
    if len(template) != count:
        raise ValueError(f"Template must contain exactly {count} positions.")
    fixed = [_index(value, count, "template candidate") for value in template if value is not None]
    if len(fixed) != len(set(fixed)):
        raise ValueError("Fixed template candidate indices must be unique.")
    remaining = [index for index in range(count) if index not in fixed]
    generator.shuffle(remaining)
    iterator = iter(remaining)
    result = [next(iterator) if value is None else _index(value, count, "template candidate") for value in template]
    return _validate_permutation(result, count)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    return value


def _positive_integer(value: Any, name: str) -> int:
    result = _integer(value, name)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _index(value: Any, count: int, name: str) -> int:
    result = _integer(value, name)
    if result < 0 or result >= count:
        raise ValueError(f"{name} must be between 0 and {count - 1}.")
    return result


__all__ = ["CallableReranker", "PermutationSuite"]

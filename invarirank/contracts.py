from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RankingSample:
    """A user context and the retrieved candidates to rerank."""

    user_id: str
    candidates: list[dict[str, Any]]
    history: list[dict[str, Any]] = field(default_factory=list)
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.user_id = str(self.user_id)
        self.history = [dict(item) for item in self.history]
        self.candidates = [dict(item) for item in self.candidates]
        self.metadata = dict(self.metadata)
        if not self.candidates:
            raise ValueError("RankingSample requires at least one candidate.")

    @classmethod
    def from_dict(cls, sample: Mapping[str, Any]) -> RankingSample:
        if "candidates" not in sample:
            raise ValueError("Ranking sample is missing required field: candidates")
        known = {"user_id", "history", "candidates", "split"}
        metadata = {key: value for key, value in sample.items() if key not in known}
        return cls(
            user_id=str(sample.get("user_id", "")),
            history=list(sample.get("history") or []),
            candidates=list(sample["candidates"]),
            split=sample.get("split"),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        sample = dict(self.metadata)
        sample.update(
            {
                "user_id": self.user_id,
                "history": [dict(item) for item in self.history],
                "candidates": [dict(item) for item in self.candidates],
            }
        )
        if self.split is not None:
            sample["split"] = self.split
        return sample


@dataclass(frozen=True)
class RankedItem:
    """One candidate in the final output order."""

    candidate_index: int
    item_id: str
    score: float
    input_position: int
    relevance: int | None = None
    candidate: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "item_id": self.item_id,
            "score": self.score,
            "input_position": self.input_position,
            "relevance": self.relevance,
            "candidate": dict(self.candidate),
        }


@dataclass(frozen=True)
class RankingResult:
    """A complete ranking plus the input permutation used to score it."""

    user_id: str
    items: tuple[RankedItem, ...]
    permutation: tuple[int, ...]
    split: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        candidate_indices = [item.candidate_index for item in self.items]
        if len(self.permutation) != len(set(self.permutation)):
            raise ValueError("RankingResult input permutation contains duplicate candidates.")
        if len(candidate_indices) != len(set(candidate_indices)):
            raise ValueError("RankingResult contains duplicate candidates.")
        if len(candidate_indices) != len(self.permutation) or set(candidate_indices) != set(self.permutation):
            raise ValueError("RankingResult must contain every candidate in the input permutation exactly once.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "split": self.split,
            "permutation": list(self.permutation),
            "items": [item.to_dict() for item in self.items],
            "metadata": dict(self.metadata),
        }


class Reranker(ABC):
    """Common contract implemented by framework and experiment rerankers."""

    @abstractmethod
    def rank(
        self,
        sample: RankingSample | Mapping[str, Any],
        *,
        permutation: Sequence[int] | None = None,
    ) -> RankingResult:
        """Score and order every candidate in a sample."""

    def rank_many(
        self,
        samples: Sequence[
            RankingSample | Mapping[str, Any] | tuple[RankingSample | Mapping[str, Any], Sequence[int] | None]
        ],
        *,
        permutations: Sequence[Sequence[int] | None] | None = None,
        batch_size: int = 1,
    ) -> list[RankingResult]:
        """Rank requests in order, safely falling back to repeated single calls."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        requests = _normalize_rank_requests(samples, permutations)
        return [self.rank(sample, permutation=permutation) for sample, permutation in requests]


def _normalize_rank_requests(
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

    requests = []
    for value in values:
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError("Rank request tuples must contain (sample, permutation).")
            requests.append((value[0], value[1]))
        else:
            requests.append((value, None))
    return requests


__all__ = [
    "RankedItem",
    "RankingResult",
    "RankingSample",
    "Reranker",
]

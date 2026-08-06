from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from invarirank.contracts import RankingResult, RankingSample, Reranker

from experiments.reranking.methods.common import (
    normalize_method_requests,
    rank_many,
    sample,
    with_metadata,
)


class ZeroShot(Reranker):
    """One scoring pass over the retrieved candidate order."""

    def __init__(self, scorer: Reranker):
        self.scorer = scorer

    def rank(
        self,
        sample_value: RankingSample | Mapping[str, Any],
        *,
        permutation: Sequence[int] | None = None,
    ) -> RankingResult:
        result = self.scorer.rank(sample_value, permutation=permutation)
        return with_metadata(result, "zero_shot", 1)

    def rank_many(
        self,
        samples: Sequence[
            RankingSample | Mapping[str, Any] | tuple[RankingSample | Mapping[str, Any], Sequence[int] | None]
        ],
        *,
        permutations: Sequence[Sequence[int] | None] | None = None,
        batch_size: int = 8,
    ) -> list[RankingResult]:
        requests = normalize_method_requests(samples, permutations)
        results = rank_many(
            self.scorer,
            [(sample(sample_value), input_permutation) for sample_value, input_permutation in requests],
            batch_size=batch_size,
        )
        return [with_metadata(result, "zero_shot", 1) for result in results]

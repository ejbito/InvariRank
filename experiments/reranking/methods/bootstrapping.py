from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from invarirank.contracts import RankingResult, RankingSample, Reranker

from experiments.reranking.methods.common import (
    borda_aggregate,
    normalize_method_requests,
    permutation,
    rank_many,
    replace_metadata,
    request_seed,
    sample,
)


class Bootstrapping(Reranker):
    """Permutation ensembling with Borda aggregation."""

    def __init__(self, scorer: Reranker, *, num_samples: int = 3, seed: int = 42):
        if num_samples < 1:
            raise ValueError("num_samples must be at least one.")
        self.scorer = scorer
        self.num_samples = int(num_samples)
        self.seed = int(seed)

    def rank(
        self,
        sample_value: RankingSample | Mapping[str, Any],
        *,
        permutation: Sequence[int] | None = None,
    ) -> RankingResult:
        return self.rank_many([(sample_value, permutation)], batch_size=1)[0]

    def rank_many(
        self,
        samples: Sequence[
            RankingSample | Mapping[str, Any] | tuple[RankingSample | Mapping[str, Any], Sequence[int] | None]
        ],
        *,
        permutations: Sequence[Sequence[int] | None] | None = None,
        batch_size: int = 8,
    ) -> list[RankingResult]:
        prepared = []
        internal_requests: list[tuple[RankingSample, Sequence[int]]] = []
        for sample_value, input_permutation in normalize_method_requests(samples, permutations):
            ranking_sample = sample(sample_value)
            outer = permutation(input_permutation, len(ranking_sample.candidates))
            seed = request_seed(self.seed, ranking_sample, outer)
            inner = [list(outer)]
            for sample_index in range(1, self.num_samples):
                shuffled = list(outer)
                random.Random(seed + sample_index * 1009).shuffle(shuffled)
                inner.append(shuffled)
            start = len(internal_requests)
            internal_requests.extend((ranking_sample, value) for value in inner)
            prepared.append((ranking_sample, outer, seed, inner, start))

        rankings = rank_many(self.scorer, internal_requests, batch_size=batch_size)
        outputs = []
        for ranking_sample, outer, seed, inner, start in prepared:
            group = rankings[start : start + self.num_samples]
            result = borda_aggregate(
                ranking_sample,
                group,
                outer,
                method="bootstrapping",
                forward_passes=len(group),
            )
            outputs.append(
                replace_metadata(
                    result,
                    {
                        **result.metadata,
                        "seed": self.seed,
                        "request_seed": seed,
                        "bootstrap_input_permutations": [list(value) for value in inner],
                    },
                )
            )
        return outputs

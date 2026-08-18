from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from invarirank.contracts import RankingResult, RankingSample, Reranker

from experiments.reranking.methods.common import (
    combined_metadata,
    normalize_method_requests,
    permutation as resolve_permutation,
    rank_many as score_many,
    ranking_from_order,
    request_seed,
    sample,
)


class StochasticGreedySelection(Reranker):
    """Repeatedly score the remaining set and append the top selected candidates."""

    def __init__(self, scorer: Reranker, *, selection_size: int = 1, seed: int = 42):
        if selection_size < 1:
            raise ValueError("selection_size must be at least one.")
        self.scorer = scorer
        self.selection_size = int(selection_size)
        self.seed = int(seed)

    def rank(
        self,
        sample_value: RankingSample | Mapping[str, Any],
        *,
        permutation: Sequence[int] | None = None,
    ) -> RankingResult:
        ranking_sample = sample(sample_value)
        outer = resolve_permutation(permutation, len(ranking_sample.candidates))
        remaining = list(outer)
        selected: list[int] = []
        local_rankings: list[RankingResult] = []
        seed = request_seed(self.seed, ranking_sample, outer)
        generator = random.Random(seed)

        while remaining:
            local_sample = RankingSample(
                user_id=ranking_sample.user_id,
                history=ranking_sample.history,
                candidates=[ranking_sample.candidates[index] for index in remaining],
                split=ranking_sample.split,
                metadata=ranking_sample.metadata,
            )
            local_result = self.scorer.rank(local_sample)
            local_rankings.append(local_result)
            chosen_local = [item.candidate_index for item in local_result.items[: self.selection_size]]
            chosen_global = [remaining[index] for index in chosen_local]
            selected.extend(chosen_global)
            chosen_set = set(chosen_global)
            remaining = [index for index in remaining if index not in chosen_set]
            generator.shuffle(remaining)

        return ranking_from_order(
            ranking_sample,
            outer,
            selected,
            scores={index: float(len(selected) - rank) for rank, index in enumerate(selected)},
            metadata={
                "method": "sgs",
                "forward_passes": len(local_rankings),
                "selection_size": self.selection_size,
                "seed": self.seed,
                "request_seed": seed,
                **combined_metadata(local_rankings),
            },
        )

    def rank_many(
        self,
        samples: Sequence[
            RankingSample | Mapping[str, Any] | tuple[RankingSample | Mapping[str, Any], Sequence[int] | None]
        ],
        *,
        permutations: Sequence[Sequence[int] | None] | None = None,
        batch_size: int = 8,
    ) -> list[RankingResult]:
        states = []
        for sample_value, input_permutation in normalize_method_requests(samples, permutations):
            ranking_sample = sample(sample_value)
            outer = resolve_permutation(input_permutation, len(ranking_sample.candidates))
            seed = request_seed(self.seed, ranking_sample, outer)
            states.append(
                {
                    "sample": ranking_sample,
                    "outer": outer,
                    "remaining": list(outer),
                    "selected": [],
                    "local_rankings": [],
                    "generator": random.Random(seed),
                    "request_seed": seed,
                }
            )

        while any(state["remaining"] for state in states):
            active = [state for state in states if state["remaining"]]
            local_requests = []
            for state in active:
                remaining = state["remaining"]
                local_requests.append(
                    (
                        RankingSample(
                            user_id=state["sample"].user_id,
                            history=state["sample"].history,
                            candidates=[state["sample"].candidates[index] for index in remaining],
                            split=state["sample"].split,
                            metadata=state["sample"].metadata,
                        ),
                        None,
                    )
                )
            local_results = score_many(self.scorer, local_requests, batch_size=batch_size)
            for state, local_result in zip(active, local_results, strict=True):
                state["local_rankings"].append(local_result)
                remaining = state["remaining"]
                chosen_local = [item.candidate_index for item in local_result.items[: self.selection_size]]
                chosen_global = [remaining[index] for index in chosen_local]
                state["selected"].extend(chosen_global)
                chosen_set = set(chosen_global)
                state["remaining"] = [index for index in remaining if index not in chosen_set]
                state["generator"].shuffle(state["remaining"])

        outputs = []
        for state in states:
            selected = state["selected"]
            outputs.append(
                ranking_from_order(
                    state["sample"],
                    state["outer"],
                    selected,
                    scores={index: float(len(selected) - rank) for rank, index in enumerate(selected)},
                    metadata={
                        "method": "sgs",
                        "forward_passes": len(state["local_rankings"]),
                        "selection_size": self.selection_size,
                        "seed": self.seed,
                        "request_seed": state["request_seed"],
                        **combined_metadata(state["local_rankings"]),
                    },
                )
            )
        return outputs

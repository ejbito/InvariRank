from __future__ import annotations

from functools import partial

from experiments.reranking.methods import LLMReranker

RERANKERS = {
    "zero_shot": partial(LLMReranker, method="zero_shot"),
    "bootstrapping": partial(LLMReranker, method="bootstrapping"),
    "sgs": partial(LLMReranker, method="sgs"),
    "stella": partial(LLMReranker, method="stella"),
}


def get_reranker_class(name: str):
    try:
        return RERANKERS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(RERANKERS))
        raise ValueError(f"Unknown reranker '{name}'. Valid rerankers: {valid}") from exc

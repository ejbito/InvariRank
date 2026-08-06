from __future__ import annotations

from experiments.retrieval.implicit_als import ImplicitALSRetriever
from experiments.retrieval.lightgcn import LightGCNRetriever

RETRIEVERS = {
    "implicit_als": ImplicitALSRetriever,
    "lightgcn": LightGCNRetriever,
}


def get_retriever_class(name: str):
    try:
        return RETRIEVERS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(RETRIEVERS))
        raise ValueError(f"Unknown retriever '{name}'. Valid retrievers: {valid}") from exc

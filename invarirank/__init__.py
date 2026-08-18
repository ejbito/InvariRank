"""Public API for the InvariRank reranking framework."""

from .config import (
    FINE_TUNED_METHODS,
    INTERACTION_WEIGHTS_NAME,
    RerankerConfig,
    TrainingConfig,
    method_from_config,
)
from .contracts import RankedItem, RankingResult, RankingSample, Reranker
from .permutations import CallableReranker, PermutationSuite
from .reranker import InvariRankReranker
from .training import Trainer

__all__ = [
    "FINE_TUNED_METHODS",
    "CallableReranker",
    "InvariRankReranker",
    "INTERACTION_WEIGHTS_NAME",
    "PermutationSuite",
    "RankedItem",
    "RankingResult",
    "RankingSample",
    "Reranker",
    "RerankerConfig",
    "Trainer",
    "TrainingConfig",
    "method_from_config",
]

__version__ = "0.1.0"

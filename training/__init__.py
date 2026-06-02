from .dataset import ListwiseRankingDataset, RankingSample, filter_and_subsample, listwise_collator, sample_permutation
from .losses import lambda_rank_loss, permutation_invariance_loss
from .metrics import (
    hr_at_k,
    kendall_tau_from_rank_maps,
    ndcg_at_k,
    spearman_rho_from_rank_maps,
    topk_overlap_at_k,
)

__all__ = [
    "ListwiseRankingDataset",
    "RankingSample",
    "evaluate",
    "filter_and_subsample",
    "hr_at_k",
    "kendall_tau_from_rank_maps",
    "lambda_rank_loss",
    "listwise_collator",
    "ndcg_at_k",
    "permutation_invariance_loss",
    "run_training_pipeline",
    "sample_permutation",
    "save_checkpoint",
    "spearman_rho_from_rank_maps",
    "topk_overlap_at_k",
    "train_step",
]


def __getattr__(name: str):
    if name in {"evaluate", "run_training_pipeline", "save_checkpoint", "train_step"}:
        from . import loop

        return getattr(loop, name)
    raise AttributeError(name)

from __future__ import annotations

import math
from typing import Any


def _to_list(values: Any) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    return [float(v) for v in values]


def hr_at_k(scores: Any, relevance: Any, k: int) -> float:
    scores_list = _to_list(scores)
    rel_list = _to_list(relevance)
    if not scores_list or not rel_list:
        return 0.0
    n = min(len(scores_list), len(rel_list))
    order = sorted(range(n), key=lambda i: scores_list[i], reverse=True)
    top = order[: min(k, n)]
    return 1.0 if any(rel_list[i] > 0 for i in top) else 0.0


def ndcg_at_k(scores: Any, relevance: Any, k: int) -> float:
    scores_list = _to_list(scores)
    rel_list = _to_list(relevance)
    n = min(len(scores_list), len(rel_list), k)
    if n <= 0:
        return 0.0

    order = sorted(range(min(len(scores_list), len(rel_list))), key=lambda i: scores_list[i], reverse=True)[:n]
    ideal = sorted(rel_list, reverse=True)[:n]

    def dcg(rels: list[float]) -> float:
        return sum((2.0**rel - 1.0) / math.log2(rank + 2) for rank, rel in enumerate(rels))

    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return float(dcg([rel_list[i] for i in order]) / ideal_dcg)


def spearman_rho_from_rank_maps(a: dict[Any, int], b: dict[Any, int]) -> float | None:
    keys = sorted(set(a) & set(b))
    n = len(keys)
    if n < 2:
        return None
    diff_sq = sum((a[k] - b[k]) ** 2 for k in keys)
    return float(1.0 - (6.0 * diff_sq) / (n * (n * n - 1)))


def kendall_tau_from_rank_maps(a: dict[Any, int], b: dict[Any, int]) -> float | None:
    keys = sorted(set(a) & set(b))
    n = len(keys)
    if n < 2:
        return None

    concordant = 0
    discordant = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            x = keys[i]
            y = keys[j]
            diff_a = a[x] - a[y]
            diff_b = b[x] - b[y]
            product = diff_a * diff_b
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1

    total_pairs = concordant + discordant
    if total_pairs == 0:
        return None
    return float((concordant - discordant) / total_pairs)


def topk_overlap_at_k(a: list[Any], b: list[Any], k: int) -> float:
    if k <= 0:
        return 0.0
    topk_a = a[:k]
    topk_b = b[:k]
    k_eff = min(len(topk_a), len(topk_b))
    if k_eff == 0:
        return 0.0
    return float(len(set(topk_a[:k_eff]) & set(topk_b[:k_eff])) / k_eff)

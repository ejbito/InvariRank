from __future__ import annotations

from typing import Any


def lambda_rank_loss(scores: Any, relevance: Any, sigma: float = 1.0, eps: float = 1e-8):
    import torch
    import torch.nn.functional as F

    device = scores.device
    scores = scores.float()
    relevance = relevance.float()

    if torch.all(relevance == relevance[0]):
        return torch.tensor(0.0, device=device)

    n = scores.numel()
    sorted_idx = torch.argsort(scores, descending=True)
    rank_pos = torch.empty(n, dtype=torch.long, device=device)
    rank_pos[sorted_idx] = torch.arange(n, device=device)

    ideal_rel = torch.sort(relevance, descending=True).values
    discounts = 1.0 / torch.log2(torch.arange(n, device=device).float() + 2.0)
    ideal_dcg = torch.sum((torch.pow(2.0, ideal_rel) - 1.0) * discounts).clamp(min=eps)

    score_diffs = scores.unsqueeze(1) - scores.unsqueeze(0)
    rel_diffs = relevance.unsqueeze(1) - relevance.unsqueeze(0)
    preference_mask = rel_diffs > 0
    if preference_mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    gains = torch.pow(2.0, relevance) - 1.0
    di = discounts[rank_pos].unsqueeze(1)
    dj = discounts[rank_pos].unsqueeze(0)
    delta_ndcg = torch.abs((gains.unsqueeze(1) - gains.unsqueeze(0)) * (di - dj)) / ideal_dcg
    pair_loss = F.softplus(-sigma * score_diffs)

    return (delta_ndcg * pair_loss * preference_mask.float()).sum() / (preference_mask.sum().float() + eps)


def permutation_invariance_loss(scores_list: list[Any], perms: list[list[int]], mode: str = "kl", temperature: float = 1.0):
    import torch
    import torch.nn.functional as F

    from ..ranking.scoring import align_scores_to_shared_candidates

    aligned, _ = align_scores_to_shared_candidates(scores_list, perms)
    if aligned is None or len(aligned) < 2:
        return torch.tensor(0.0, device=scores_list[0].device)

    stacked = torch.stack([s.float() for s in aligned], dim=0)
    log_probs = F.log_softmax(stacked / max(temperature, 1e-6), dim=-1)
    probs = log_probs.exp()

    base_log = log_probs[0]
    base_prob = probs[0]
    losses = []
    for i in range(1, log_probs.shape[0]):
        cur_log = log_probs[i]
        cur_prob = probs[i]
        kl_ab = torch.sum(base_prob * (base_log - cur_log), dim=-1)
        if mode == "kl":
            losses.append(kl_ab)
        elif mode in {"symkl", "jeffreys"}:
            kl_ba = torch.sum(cur_prob * (cur_log - base_log), dim=-1)
            losses.append(kl_ab + kl_ba)
        else:
            raise ValueError(f"Unsupported permutation loss mode: {mode}")
    return torch.stack(losses).mean()

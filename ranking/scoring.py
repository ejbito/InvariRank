from __future__ import annotations

import copy
from typing import Any

from model.invarirank import SpanExtractor, build_attention_mask, build_position_ids


class MeanLogProbListwiseScorer:
    def __init__(self, backbone: Any, tokenizer: Any, cfg: Any):
        import torch.nn as nn

        class _Scorer(nn.Module):
            def __init__(self, outer: MeanLogProbListwiseScorer):
                super().__init__()
                self.outer = outer
                self.backbone = outer.backbone

            def forward(self, input_ids: Any, attention_mask: Any):
                return self.outer(input_ids, attention_mask)

        self.backbone = backbone
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.span_extractor = SpanExtractor(tokenizer, cfg)
        self.module = _Scorer(self)

    def to(self, *args: Any, **kwargs: Any):
        self.module.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        self.module.train(mode)
        return self

    def eval(self):
        self.module.eval()
        return self

    def parameters(self):
        return self.module.parameters()

    def __call__(self, input_ids: Any, attention_mask: Any):
        import torch
        import torch.nn.functional as F

        span_info = self.span_extractor(input_ids)
        dtype = next(self.backbone.parameters()).dtype
        attn = build_attention_mask(attention_mask, span_info, self.cfg, dtype)
        position_ids = build_position_ids(input_ids, span_info, self.cfg)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attn,
            position_ids=position_ids,
            use_cache=False,
        )
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        token_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        scores = []
        for start, end in span_info.candidate_spans:
            shifted_start = max(start - 1, 0)
            shifted_end = max(end - 1, shifted_start + 1)
            span_lp = token_log_probs[0, shifted_start:shifted_end]
            scores.append(span_lp.mean())
        return torch.stack(scores)


def align_scores_to_shared_candidates(scores_list: list[Any], perms: list[list[int]]):
    visible_sets = [set(perm[: scores.numel()]) for scores, perm in zip(scores_list, perms)]
    shared = set.intersection(*visible_sets) if visible_sets else set()
    if len(shared) < 2:
        return None, None

    shared_list = sorted(shared)
    aligned = []
    for scores, perm in zip(scores_list, perms):
        pos = {cand: i for i, cand in enumerate(perm[: scores.numel()])}
        aligned.append(scores[[pos[c] for c in shared_list]])
    return aligned, shared_list


def build_permutation_rank_record(
    *,
    sample_index: int,
    user_id: str,
    candidate_ids: list[str],
    permutation: list[int],
    scores: Any,
    relevance: list[int],
) -> dict[str, Any]:
    score_values = [float(x) for x in scores.detach().cpu().tolist()]
    rows = []
    for local_pos, candidate_index in enumerate(permutation[: len(score_values)]):
        rows.append(
            {
                "candidate_index": int(candidate_index),
                "candidate_id": candidate_ids[candidate_index],
                "input_position": int(local_pos),
                "score": score_values[local_pos],
                "relevance": int(relevance[local_pos]),
            }
        )
    ranking = sorted(rows, key=lambda row: row["score"], reverse=True)
    return {
        "sample_index": sample_index,
        "user_id": user_id,
        "permutation": permutation,
        "ranking": ranking,
    }


def build_rank_record(batch: dict[str, Any], scores_list: list[Any], perms: list[list[int]]) -> dict[str, Any]:
    candidates = [copy.deepcopy(c) for c in batch.get("candidates", [])]
    candidate_ids = list(batch.get("candidate_ids", []))
    if not candidate_ids:
        candidate_ids = [str(i) for i in range(len(candidates))]

    record = {
        "sample_index": int(batch.get("sample_index", -1)),
        "user_id": batch.get("user_id"),
        "split": batch.get("split"),
        "list_length": int(batch.get("list_length", len(candidates))),
        "num_items": int(batch.get("num_items", len(candidates))),
        "history": copy.deepcopy(batch.get("history", [])),
        "candidates": candidates,
        "permutations": [],
    }

    relevance_seqs = batch.get("relevance", [])
    for perm_idx, (scores, perm) in enumerate(zip(scores_list, perms)):
        tensor = scores.detach().float().cpu()
        limit = min(int(tensor.numel()), len(perm))
        visible_perm = [int(x) for x in perm[:limit]]
        visible_scores = [float(x) for x in tensor[:limit].tolist()]
        visible_relevance = []
        if perm_idx < len(relevance_seqs):
            visible_relevance = [int(x) for x in relevance_seqs[perm_idx][:limit]]

        input_item_ids = [candidate_ids[idx] if idx < len(candidate_ids) else None for idx in visible_perm]
        ranking_pairs = sorted(
            zip(visible_perm, input_item_ids, visible_scores),
            key=lambda x: x[2],
            reverse=True,
        )

        record["permutations"].append(
            {
                "permutation_index": int(perm_idx),
                "input": {
                    "candidate_indices": visible_perm,
                    "item_ids": input_item_ids,
                    "relevance": visible_relevance,
                },
                "scores": visible_scores,
                "output_ranking": {
                    "candidate_indices": [idx for idx, _, _ in ranking_pairs],
                    "item_ids": [iid for _, iid, _ in ranking_pairs],
                    "scores": [score for _, _, score in ranking_pairs],
                },
            }
        )

    return record

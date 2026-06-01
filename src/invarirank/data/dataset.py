from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .prompts import build_prompt, candidate_id, extract_relevance_labels
from ..utils import load_jsonl, save_jsonl


@dataclass(frozen=True)
class RankingSample:
    user_id: str
    history: list[dict[str, Any]]
    candidates: list[dict[str, Any]]


def sample_permutation(n: int, *, deterministic: bool = False, seed: int | None = None) -> list[int]:
    perm = list(range(n))
    rng = random.Random(seed) if deterministic else random
    rng.shuffle(perm)
    return perm


class ListwiseRankingDataset:
    def __init__(self, samples: list[dict[str, Any]], cfg: Any, tokenizer: Any, *, mode: str = "train"):
        self.samples = samples
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.mode = mode

    def __len__(self) -> int:
        return len(self.samples)

    def _num_permutations(self) -> int:
        if self.mode == "train":
            return int(getattr(self.cfg, "train_num_permutations", 1))
        return int(getattr(self.cfg, "eval_num_permutations", 1))

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        n = len(sample["candidates"])
        tokenized = []
        relevance = []
        permutations = []

        deterministic = self.mode != "train" and bool(getattr(self.cfg, "val_perms_deterministic", True))
        for pidx in range(self._num_permutations()):
            perm = sample_permutation(n, deterministic=deterministic, seed=index * 1009 + pidx)
            prompt = build_prompt(sample, perm, self.cfg)
            enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=int(self.cfg.max_seq_length),
            )
            tokenized.append(enc)
            relevance.append(extract_relevance_labels(sample, perm))
            permutations.append(perm)

        return {
            "sample_index": index,
            "user_id": sample.get("user_id", str(index)),
            "split": sample.get("split", self.mode),
            "history": sample.get("history", []),
            "candidates": sample["candidates"],
            "num_items": n,
            "list_length": n,
            "candidate_ids": [candidate_id(item, i) for i, item in enumerate(sample["candidates"])],
            "tokenized": tokenized,
            "relevance": relevance,
            "permutations": permutations,
            "sample": sample,
        }


def listwise_collator(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("ListwiseRankingDataset currently expects batch_size=1.")
    return batch[0]


def filter_and_subsample(
    samples: list[dict[str, Any]],
    num_samples: int | None = None,
) -> list[dict[str, Any]]:
    valid = [s for s in samples if s.get("candidates")]
    if num_samples is not None:
        valid = valid[: int(num_samples)]
    return valid


__all__ = [
    "RankingSample",
    "ListwiseRankingDataset",
    "filter_and_subsample",
    "listwise_collator",
    "load_jsonl",
    "sample_permutation",
    "save_jsonl",
]

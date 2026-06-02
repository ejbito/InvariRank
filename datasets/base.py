from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from datasets.utils import set_seed
from retriever import LightGCNRetriever

from .utils import (
    build_target_ranking,
    cfg_get,
    graded_relevance,
    make_candidate,
    save_jsonl,
    stable_hash_int,
    summarize_lengths,
)


class BaseDataset(ABC):
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.seed = int(cfg_get(cfg, "training.seed", cfg_get(cfg, "seed", 42)))
        self.rng = random.Random(self.seed)
        self.item_metadata: dict[Any, dict] = {}
        self.user_histories: dict[Any, list[dict]] = {}

    @classmethod
    @abstractmethod
    def code(cls) -> str:
        pass

    @abstractmethod
    def load_raw(self) -> None:
        pass

    @abstractmethod
    def build_item_metadata(self) -> None:
        pass

    @abstractmethod
    def build_user_histories(self) -> None:
        pass

    def generate_samples(self) -> tuple[list[dict], list[dict], list[dict]]:
        if self.code() == "amazon_books":
            return sample_amazon_books(self)
        return sample_movielens(self)


def build_dataset_splits(cfg: Any) -> tuple[list[dict], list[dict], list[dict]]:
    set_seed(int(cfg_get(cfg, "training.seed", cfg_get(cfg, "seed", 42))))
    dataset_name = str(cfg_get(cfg, "dataset.name", "")).lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Expected one of {sorted(DATASET_REGISTRY)}")

    dataset = DATASET_REGISTRY[dataset_name](cfg)
    print("[Dataset] Build start")
    dataset.load_raw()
    dataset.build_item_metadata()
    dataset.build_user_histories()

    train, val, test = dataset.generate_samples()
    print("[Dataset] Validating samples")
    validate_split(train)
    validate_split(val)
    validate_split(test)
    return train, val, test


def write_dataset_splits(train: list[dict], val: list[dict], test: list[dict], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_jsonl(train, output / "train.jsonl")
    save_jsonl(val, output / "val.jsonl")
    save_jsonl(test, output / "test.jsonl")
    print(f"[Dataset] Wrote train={len(train)}, val={len(val)}, test={len(test)} to {output}")


def validate_sample(sample: dict[str, Any], num_candidates: int) -> None:
    required = {"user_id", "history", "candidates", "target_ranking", "list_length", "split"}
    missing = sorted(required - set(sample))
    if missing:
        raise ValueError(f"Dataset sample is missing field(s): {missing}")

    candidates = sample["candidates"]
    if len(candidates) != num_candidates:
        raise ValueError(f"Expected {num_candidates} candidates, found {len(candidates)}")

    item_ids = [candidate["item_id"] for candidate in candidates]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Duplicate items in candidate list")

    rels = [candidate["relevance"] for candidate in candidates]
    if not all(isinstance(rel, int) for rel in rels):
        raise ValueError("All candidate relevance labels must be integers")


def validate_split(samples: list[dict[str, Any]]) -> None:
    for sample in samples:
        validate_sample(sample, int(sample["list_length"]))


def split_user_histories(
    user_histories: dict, history_length: int, train_pct: float, val_pct: float, train_future_pct: float
):
    splits = {}
    for uid in sorted(user_histories.keys()):
        hist = sorted(user_histories[uid], key=lambda x: x["timestamp"])
        n = len(hist)
        n_train_total = int(n * train_pct)
        n_val = int(n * val_pct)
        n_test = n - n_train_total - n_val
        if n_train_total <= 0 or n_val <= 0 or n_test <= 0:
            continue

        train_segment = hist[:n_train_total]
        val_segment = hist[n_train_total : n_train_total + n_val]
        test_segment = hist[n_train_total + n_val :]

        n_train_future = max(1, int(len(train_segment) * train_future_pct))
        if len(train_segment) - n_train_future < history_length:
            continue

        past_train = train_segment[: len(train_segment) - n_train_future]
        future_train = train_segment[len(train_segment) - n_train_future :]
        if future_train and val_segment and test_segment:
            splits[uid] = (past_train, future_train, val_segment, test_segment)
    return splits


def build_train_interactions(splits: dict, implicit_min_rating: float | None) -> list[tuple]:
    interactions = []
    for uid, (past_train, _future_train, _future_val, _future_test) in splits.items():
        for h in past_train:
            if implicit_min_rating is None or float(h.get("rating", 0.0)) >= implicit_min_rating:
                interactions.append((uid, h["item_id"]))
    return interactions


def build_retriever(cfg: Any):
    method = str(cfg_get(cfg, "retrieval.method", "lightgcn")).lower()
    if method in {"lightgcn", "lgcn"}:
        return LightGCNRetriever(cfg)
    raise ValueError(f"Unsupported retrieval method: {method}")


def log_recall_metrics(recall_at: dict[int, list[float]]) -> None:
    if not recall_at:
        print("[Sampling] No recall metrics computed (empty splits or retrieval).")
        return
    print("[Sampling] Retriever Recall Metrics")
    for k in sorted(recall_at):
        values = recall_at[k]
        if values:
            print(f"[Sampling]   Recall@{k}: {sum(values) / len(values):.4f}")


def split_stats(splits: dict) -> dict[str, float]:
    lengths = [
        len(past) + len(future_train) + len(future_val) + len(future_test)
        for (past, future_train, future_val, future_test) in splits.values()
    ]
    return summarize_lengths(lengths)


def sample_movielens(dataset: BaseDataset):
    cfg = dataset.cfg
    seed = int(cfg_get(cfg, "training.seed", 42))
    list_sizes = list(cfg_get(cfg, "sampling.list_sizes", [15, 25, 50]))
    history_length = min(
        int(cfg_get(cfg, "split.history_length", 20)), int(cfg_get(cfg, "reranking.max_history_items", 20))
    )
    train_pct = float(cfg_get(cfg, "split.train_pct", 0.7))
    val_pct = float(cfg_get(cfg, "split.val_pct", 0.1))
    train_future_pct = float(cfg_get(cfg, "split.train_future_pct", 0.2))
    implicit_min_rating = float(cfg_get(cfg, "dataset.implicit_min_rating", 4.0))
    deterministic = bool(cfg_get(cfg, "sampling.deterministic", True))
    k_max = int(cfg_get(cfg, "retrieval.k_max", 1500))
    retrieval_pool = min(300, k_max)
    pos_dist = [(1, 0.25), (2, 0.45), (3, 0.30)]

    print("[Sampling] Starting MovieLens sampling")
    splits = split_user_histories(dataset.user_histories, history_length, train_pct, val_pct, train_future_pct)
    stats = split_stats(splits)
    print(
        "[Sampling] Split users: "
        f"{len(splits)} (min={stats['min']:.0f}, mean={stats['mean']:.1f}, "
        f"max={stats['max']:.0f} interactions)"
    )

    retriever = build_retriever(cfg)
    train_interactions = build_train_interactions(splits, implicit_min_rating)
    print(f"[Sampling] Retriever edges: {len(train_interactions)}")
    retriever.fit(train_interactions)

    all_items = sorted(dataset.item_metadata.keys())
    train, val, test = [], [], []
    recall_at = defaultdict(list)

    for uid in tqdm(sorted(splits.keys()), desc="[Sampling] Users"):
        past_train, future_train, future_val, future_test = splits[uid]
        future_all = future_train + future_val + future_test
        future_all_ids = {x["item_id"] for x in future_all}
        retrieved_ranked = retriever.retrieve(uid, retrieval_pool)
        if not retrieved_ranked:
            continue

        for k in (10, 50, 100):
            recall_at[k].append(len(set(retrieved_ranked[:k]) & future_all_ids) / max(1, len(future_all_ids)))

        split_data = {
            "train": (past_train, future_train),
            "val": (past_train + future_train, future_val),
            "test": (past_train + future_train + future_val, future_test),
        }

        for split_name, (past, future) in split_data.items():
            history = past[-history_length:]
            hist_ids = {h["item_id"] for h in history}
            positives_all = [
                (h["item_id"], h["relevance"])
                for h in sorted(future, key=lambda x: (-x["rating"], -x["timestamp"]))
                if h["relevance"] > 0
            ]
            if not positives_all:
                continue

            retrieved_band = [
                mid for mid in retrieved_ranked[3:100] if mid not in hist_ids and mid in dataset.item_metadata
            ]
            max_k = max(k for k, _ in pos_dist)
            if deterministic:
                positives = positives_all[:max_k]
            else:
                rng = random.Random(stable_hash_int(f"{seed}-{uid}-{split_name}-pos"))
                r = rng.random()
                cum, k_pos = 0.0, 1
                for k, p in pos_dist:
                    cum += p
                    if r <= cum:
                        k_pos = k
                        break
                k_pos = min(k_pos, len(positives_all), max_k)
                weights = [max(rel, 1) for _, rel in positives_all]
                idxs = list(dict.fromkeys(rng.choices(range(len(positives_all)), weights=weights, k=k_pos)))
                if len(idxs) < k_pos:
                    rem = [i for i in range(len(positives_all)) if i not in idxs]
                    rng.shuffle(rem)
                    idxs += rem[: k_pos - len(idxs)]
                positives = [positives_all[i] for i in idxs]

            for list_size in list_sizes:
                candidates = []
                banned = set(hist_ids)
                for mid, rel in positives:
                    candidates.append(make_candidate(mid, rel, dataset.item_metadata[mid]))
                    banned.add(mid)

                min_hard = max(6, list_size // 3)
                if not deterministic:
                    random.Random(stable_hash_int(f"{seed}-{uid}-{split_name}-neg")).shuffle(retrieved_band)
                for mid in retrieved_band:
                    if len(candidates) >= len(positives) + min_hard:
                        break
                    if mid not in banned:
                        candidates.append(make_candidate(mid, 0, dataset.item_metadata[mid]))
                        banned.add(mid)

                fill_candidates(
                    candidates,
                    banned,
                    all_items,
                    dataset.item_metadata,
                    list_size,
                    deterministic,
                    f"{seed}-{uid}-{split_name}-fill",
                )
                append_sample(train, val, test, split_name, uid, history, candidates, list_size)

    print(f"[Sampling] Samples: train={len(train)}, val={len(val)}, test={len(test)}")
    log_recall_metrics(recall_at)
    return train, val, test


def sample_amazon_books(dataset: BaseDataset):
    cfg = dataset.cfg
    seed = int(cfg_get(cfg, "training.seed", 42))
    list_sizes = list(cfg_get(cfg, "sampling.list_sizes", [15, 25, 50]))
    history_length = min(
        int(cfg_get(cfg, "split.history_length", 20)), int(cfg_get(cfg, "reranking.max_history_items", 20))
    )
    train_pct = float(cfg_get(cfg, "split.train_pct", 0.7))
    val_pct = float(cfg_get(cfg, "split.val_pct", 0.1))
    train_future_pct = float(cfg_get(cfg, "split.train_future_pct", 0.2))
    min_pos = int(cfg_get(cfg, "sampling.min_future_positives", 1))
    require_retrieved_pos = bool(cfg_get(cfg, "sampling.amazon.require_retrieved_positive", True))
    deterministic = bool(cfg_get(cfg, "sampling.deterministic", True))

    print("[Sampling] Starting Amazon Books sampling")
    splits = split_user_histories(dataset.user_histories, history_length, train_pct, val_pct, train_future_pct)
    stats = split_stats(splits)
    print(
        "[Sampling] Split users: "
        f"{len(splits)} (min={stats['min']:.0f}, mean={stats['mean']:.1f}, "
        f"max={stats['max']:.0f} interactions)"
    )

    retriever = build_retriever(cfg)
    train_interactions = build_train_interactions(splits, implicit_min_rating=None)
    print(f"[Sampling] Retriever edges: {len(train_interactions)}")
    retriever.fit(train_interactions)

    train, val, test = [], [], []
    recall_at = defaultdict(list)
    all_items = sorted(dataset.item_metadata.keys())

    for uid in tqdm(sorted(splits.keys()), desc="[Sampling] Users", total=len(splits)):
        past_train, future_train, future_val, future_test = splits[uid]
        future_all = future_train + future_val + future_test
        future_all_ids = {x["item_id"] for x in future_all}
        retrieved = retriever.retrieve(uid, max(list_sizes) * 10) or []
        for k in (10, 50, 100):
            if retrieved:
                recall_at[k].append(len(set(retrieved[:k]) & future_all_ids) / max(1, len(future_all_ids)))

        neg_pool = [i for i in retrieved if i not in future_all_ids and i in dataset.item_metadata]
        split_data = {
            "train": (past_train, future_train),
            "val": (past_train + future_train, future_val),
            "test": (past_train + future_train + future_val, future_test),
        }

        for split_name, (past, future) in split_data.items():
            history = past[-history_length:]
            future_ids = {x["item_id"] for x in future}
            if require_retrieved_pos:
                positives = [i for i in retrieved if i in future_ids]
                if len(positives) < min_pos:
                    continue
                pos_ids = positives[:min_pos]
            else:
                positives_all = [
                    x
                    for x in sorted(future, key=lambda x: (-float(x.get("rating", 0.0)), -int(x["timestamp"])))
                    if x["item_id"] in dataset.item_metadata and int(x.get("relevance", 0)) >= 2
                ]
                if len(positives_all) < min_pos:
                    continue
                pos_ids = [x["item_id"] for x in positives_all[:min_pos]]

            for list_size in list_sizes:
                candidates = []
                banned = set()
                for mid in pos_ids:
                    rating = next((x["rating"] for x in future if x["item_id"] == mid), None)
                    candidates.append(make_candidate(mid, graded_relevance(rating), dataset.item_metadata[mid]))
                    banned.add(mid)

                if not deterministic:
                    random.Random(stable_hash_int(f"{seed}-{uid}-{split_name}-neg")).shuffle(neg_pool)
                for mid in neg_pool:
                    if len(candidates) >= list_size:
                        break
                    if mid not in banned:
                        candidates.append(make_candidate(mid, 0, dataset.item_metadata[mid]))
                        banned.add(mid)

                fill_candidates(
                    candidates,
                    banned,
                    all_items,
                    dataset.item_metadata,
                    list_size,
                    deterministic,
                    f"{seed}-{uid}-{split_name}-fill",
                )
                if sum(candidate["relevance"] >= 2 for candidate in candidates) >= min_pos:
                    append_sample(train, val, test, split_name, uid, history, candidates, list_size)

    print(f"[Sampling] Samples: train={len(train)}, val={len(val)}, test={len(test)}")
    log_recall_metrics(recall_at)
    return train, val, test


def fill_candidates(candidates, banned, all_items, item_metadata, list_size, deterministic, seed_key):
    if len(candidates) >= list_size:
        return
    if deterministic:
        for item_id in all_items:
            if len(candidates) >= list_size:
                break
            if item_id in banned:
                continue
            candidates.append(make_candidate(item_id, 0, item_metadata[item_id]))
            banned.add(item_id)
        return

    rng = random.Random(stable_hash_int(seed_key))
    while len(candidates) < list_size:
        item_id = all_items[rng.randrange(len(all_items))]
        if item_id in banned:
            continue
        candidates.append(make_candidate(item_id, 0, item_metadata[item_id]))
        banned.add(item_id)


def append_sample(train, val, test, split_name, uid, history, candidates, list_size):
    sample = {
        "user_id": uid,
        "history": history,
        "candidates": candidates,
        "target_ranking": build_target_ranking(candidates, None),
        "list_length": list_size,
        "split": split_name,
    }
    if split_name == "train":
        train.append(sample)
    elif split_name == "val":
        val.append(sample)
    else:
        test.append(sample)


DATASET_REGISTRY: dict[str, type[BaseDataset]] = {}

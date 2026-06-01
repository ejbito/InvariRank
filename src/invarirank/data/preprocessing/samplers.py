import hashlib
import random
from collections import defaultdict

from tqdm.auto import tqdm

from .config_utils import cfg_get
from .retrievers import build_retriever
from .utils import (
    build_target_ranking,
    graded_relevance,
    make_candidate,
    summarize_lengths,
)


def _log(msg: str):
    print(f"[Sampling] {msg}")


def _stable_hash_int(text: str) -> int:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _split_user_histories(
    user_histories: dict,
    history_length: int,
    train_pct: float,
    val_pct: float,
    test_pct: float,
    train_future_pct: float,
):
    splits = {}
    for uid in sorted(user_histories.keys()):
        hist = user_histories[uid]
        hist = sorted(hist, key=lambda x: x["timestamp"])

        n = len(hist)
        n_train_total = int(n * train_pct)
        n_val = int(n * val_pct)
        n_test = n - n_train_total - n_val
        if n_train_total <= 0 or n_val <= 0 or n_test <= 0:
            continue

        t_train_end = n_train_total
        t_val_end = t_train_end + n_val

        train_segment = hist[:t_train_end]
        val_segment = hist[t_train_end:t_val_end]
        test_segment = hist[t_val_end:]

        n_train_future = max(1, int(len(train_segment) * train_future_pct))
        if len(train_segment) - n_train_future < history_length:
            continue

        past_train = train_segment[: len(train_segment) - n_train_future]
        future_train = train_segment[len(train_segment) - n_train_future :]
        future_val = val_segment
        future_test = test_segment

        if not future_train or not future_val or not future_test:
            continue

        splits[uid] = (past_train, future_train, future_val, future_test)

    return splits


def _summarize_splits(splits: dict) -> dict[str, float]:
    lengths = [
        len(past) + len(future_train) + len(future_val) + len(future_test)
        for (past, future_train, future_val, future_test) in splits.values()
    ]
    return summarize_lengths(lengths)


def _build_train_interactions(
    splits: dict,
    implicit_min_rating: float | None,
):
    interactions: list[tuple] = []
    for uid, (past_train, _future_train, _future_val, _future_test) in splits.items():
        for h in past_train:
            if implicit_min_rating is None or float(h.get("rating", 0.0)) >= implicit_min_rating:
                interactions.append((uid, h["item_id"]))
    return interactions


def _log_recall_metrics(recall_at: dict[int, list[float]]):
    if not recall_at:
        _log("No recall metrics computed (empty splits or retrieval).")
        return
    _log("Retriever Recall Metrics")
    for K in sorted(recall_at):
        vals = recall_at[K]
        if not vals:
            continue
        mean_recall = sum(vals) / len(vals)
        print(f"[Sampling]   Recall@{K}: {mean_recall:.4f}")


def sample_movielens(dataset):
    cfg = dataset.cfg
    seed = int(cfg_get(cfg, "training.seed", 42))
    list_sizes = list(cfg_get(cfg, "sampling.list_sizes", [15, 25, 50]))
    H = min(
        int(cfg_get(cfg, "split.history_length", 20)),
        int(cfg_get(cfg, "reranking.max_history_items", 20)),
    )

    train_pct = float(cfg_get(cfg, "split.train_pct", 0.7))
    val_pct = float(cfg_get(cfg, "split.val_pct", 0.1))
    test_pct = float(cfg_get(cfg, "split.test_pct", 0.2))
    train_future_pct = float(cfg_get(cfg, "split.train_future_pct", 0.2))
    implicit_min_rating = float(cfg_get(cfg, "dataset.implicit_min_rating", 4.0))

    pos_dist = [(1, 0.25), (2, 0.45), (3, 0.30)]

    k_max = int(cfg_get(cfg, "retrieval.k_max", 1500))
    retrieval_pool = min(300, k_max)
    deterministic = bool(cfg_get(cfg, "sampling.deterministic", True))
    root_det = cfg_get(cfg, "deterministic", None)
    if root_det is not None:
        deterministic = bool(root_det)
    _log("Starting MovieLens sampling")
    _log(f"Users total: {len(dataset.user_histories)}")
    _log(f"History length: {H}, list sizes: {list_sizes}")
    _log(f"Deterministic: {deterministic}")

    splits = _split_user_histories(
        dataset.user_histories,
        H,
        train_pct,
        val_pct,
        test_pct,
        train_future_pct,
    )
    split_stats = _summarize_splits(splits)
    _log(
        f"Split users: {len(splits)} (min={split_stats['min']:.0f}, "
        f"mean={split_stats['mean']:.1f}, max={split_stats['max']:.0f} interactions)"
    )

    train_interactions = _build_train_interactions(splits, implicit_min_rating)
    _log(f"Retriever edges: {len(train_interactions)}")

    retriever = build_retriever(cfg)
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

        for K in (10, 50, 100):
            recall_at[K].append(len(set(retrieved_ranked[:K]) & future_all_ids) / max(1, len(future_all_ids)))

        split_data = {
            "train": (past_train, future_train),
            "val": (past_train + future_train, future_val),
            "test": (past_train + future_train + future_val, future_test),
        }

        for split_name, (past, future) in split_data.items():
            history = past[-H:]
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
                rng = random.Random(_stable_hash_int(f"{seed}-{uid}-{split_name}-pos"))
                r = rng.random()
                cum, K = 0.0, 1
                for k, p in pos_dist:
                    cum += p
                    if r <= cum:
                        K = k
                        break
                K = min(K, len(positives_all), max_k)
                weights = [max(rel, 1) for _, rel in positives_all]
                idxs = rng.choices(range(len(positives_all)), weights=weights, k=K)
                idxs = list(dict.fromkeys(idxs))
                if len(idxs) < K:
                    rem = [i for i in range(len(positives_all)) if i not in idxs]
                    rng.shuffle(rem)
                    idxs += rem[: K - len(idxs)]
                positives = [positives_all[i] for i in idxs]

            for L in list_sizes:
                cands = []
                banned = set(hist_ids)

                for mid, rel in positives:
                    meta = dataset.item_metadata[mid]
                    cands.append(make_candidate(mid, rel, meta))
                    banned.add(mid)

                min_hard = max(6, L // 3)
                if not deterministic:
                    rng = random.Random(_stable_hash_int(f"{seed}-{uid}-{split_name}-neg"))
                    rng.shuffle(retrieved_band)
                for mid in retrieved_band:
                    if len(cands) >= len(positives) + min_hard:
                        break
                    if mid not in banned:
                        meta = dataset.item_metadata[mid]
                        cands.append(make_candidate(mid, 0, meta))
                        banned.add(mid)

                if len(cands) < L:
                    if deterministic:
                        for mid in all_items:
                            if len(cands) >= L:
                                break
                            if mid in banned:
                                continue
                            meta = dataset.item_metadata[mid]
                            cands.append(make_candidate(mid, 0, meta))
                            banned.add(mid)
                    else:
                        rng = random.Random(_stable_hash_int(f"{seed}-{uid}-{split_name}-fill-{L}"))
                        while len(cands) < L:
                            mid = all_items[rng.randrange(len(all_items))]
                            if mid in banned:
                                continue
                            meta = dataset.item_metadata[mid]
                            cands.append(make_candidate(mid, 0, meta))
                            banned.add(mid)

                ranking = build_target_ranking(cands, None)

                record = {
                    "user_id": uid,
                    "history": history,
                    "candidates": cands,
                    "target_ranking": ranking,
                    "list_length": L,
                    "split": split_name,
                }

                if split_name == "train":
                    train.append(record)
                elif split_name == "val":
                    val.append(record)
                else:
                    test.append(record)

    _log(f"Samples: train={len(train)}, val={len(val)}, test={len(test)}")
    _log_recall_metrics(recall_at)

    return train, val, test


def sample_amazon(dataset):
    cfg = dataset.cfg
    seed = int(cfg_get(cfg, "training.seed", 42))
    train, val, test = [], [], []

    H = min(
        int(cfg_get(cfg, "split.history_length", 20)),
        int(cfg_get(cfg, "reranking.max_history_items", 20)),
    )
    train_pct = float(cfg_get(cfg, "split.train_pct", 0.7))
    val_pct = float(cfg_get(cfg, "split.val_pct", 0.1))
    test_pct = float(cfg_get(cfg, "split.test_pct", 0.2))
    train_future_pct = float(cfg_get(cfg, "split.train_future_pct", 0.2))
    list_sizes = list(cfg_get(cfg, "sampling.list_sizes", [15, 25, 50]))
    min_pos = int(cfg_get(cfg, "sampling.min_future_positives", 1))
    require_retrieved_pos = bool(cfg_get(cfg, "sampling.amazon.require_retrieved_positive", True))
    deterministic = bool(cfg_get(cfg, "sampling.deterministic", True))
    root_det = cfg_get(cfg, "deterministic", None)
    if root_det is not None:
        deterministic = bool(root_det)
    _log("Starting Amazon sampling")
    _log(f"Users total: {len(dataset.user_histories)}")
    _log(f"History length: {H}, list sizes: {list_sizes}")
    _log(f"Deterministic: {deterministic}")

    splits = _split_user_histories(
        dataset.user_histories,
        H,
        train_pct,
        val_pct,
        test_pct,
        train_future_pct,
    )
    split_stats = _summarize_splits(splits)
    _log(
        f"Split users: {len(splits)} (min={split_stats['min']:.0f}, "
        f"mean={split_stats['mean']:.1f}, max={split_stats['max']:.0f} interactions)"
    )

    train_interactions = _build_train_interactions(splits, implicit_min_rating=None)
    _log(f"Retriever edges: {len(train_interactions)}")

    retriever = build_retriever(cfg)
    retriever.fit(train_interactions)

    recall_at = defaultdict(list)

    for uid in tqdm(
        sorted(splits.keys()),
        desc="[Sampling] Users",
        total=len(splits),
    ):
        past_train, future_train, future_val, future_test = splits[uid]
        future_all = future_train + future_val + future_test
        future_all_ids = {x["item_id"] for x in future_all}

        K_pool = max(list_sizes) * 10
        retrieved = retriever.retrieve(uid, K_pool)
        if not retrieved:
            retrieved = []

        for K in (10, 50, 100):
            if retrieved:
                recall_at[K].append(len(set(retrieved[:K]) & future_all_ids) / max(1, len(future_all_ids)))

        negatives = [i for i in retrieved if i not in future_all_ids]
        neg_pool = [i for i in negatives if i in dataset.item_metadata]
        all_items = sorted(dataset.item_metadata.keys())

        split_data = {
            "train": (past_train, future_train),
            "val": (past_train + future_train, future_val),
            "test": (past_train + future_train + future_val, future_test),
        }

        for split_name, (past, future) in split_data.items():
            history = past[-H:]
            future_ids = {x["item_id"] for x in future}
            if require_retrieved_pos:
                positives = [i for i in retrieved if i in future_ids]
                if len(positives) < min_pos:
                    continue

                if deterministic:
                    pos_ids = positives[:min_pos]
                else:
                    rng = random.Random(_stable_hash_int(f"{seed}-{uid}-{split_name}-pos"))
                    rng.shuffle(positives)
                    pos_ids = positives[:min_pos]
            else:
                positives_all = [
                    x
                    for x in sorted(future, key=lambda x: (-float(x.get("rating", 0.0)), -int(x["timestamp"])))
                    if x["item_id"] in dataset.item_metadata and int(x.get("relevance", 0)) >= 2
                ]
                if len(positives_all) < min_pos:
                    continue
                if deterministic:
                    pos_ids = [x["item_id"] for x in positives_all[:min_pos]]
                else:
                    rng = random.Random(_stable_hash_int(f"{seed}-{uid}-{split_name}-pos"))
                    rng.shuffle(positives_all)
                    pos_ids = [x["item_id"] for x in positives_all[:min_pos]]

            for L in list_sizes:
                cands = []
                banned = set()

                for mid in pos_ids:
                    meta = dataset.item_metadata[mid]
                    rating = next(
                        (x["rating"] for x in future if x["item_id"] == mid),
                        None,
                    )
                    rel = graded_relevance(rating) if rating is not None else 0
                    cands.append(make_candidate(mid, rel, meta))
                    banned.add(mid)

                if not deterministic:
                    rng = random.Random(_stable_hash_int(f"{seed}-{uid}-{split_name}-neg"))
                    rng.shuffle(neg_pool)
                for mid in neg_pool:
                    if len(cands) >= L:
                        break
                    if mid in banned:
                        continue
                    meta = dataset.item_metadata[mid]
                    cands.append(make_candidate(mid, 0, meta))
                    banned.add(mid)

                if len(cands) < L:
                    if deterministic:
                        for mid in all_items:
                            if len(cands) >= L:
                                break
                            if mid in banned:
                                continue
                            meta = dataset.item_metadata[mid]
                            cands.append(make_candidate(mid, 0, meta))
                            banned.add(mid)
                    else:
                        rng = random.Random(_stable_hash_int(f"{seed}-{uid}-{split_name}-fill-{L}"))
                        while len(cands) < L:
                            mid = all_items[rng.randrange(len(all_items))]
                            if mid in banned:
                                continue
                            meta = dataset.item_metadata[mid]
                            cands.append(make_candidate(mid, 0, meta))
                            banned.add(mid)

                if sum(c["relevance"] >= 2 for c in cands) < min_pos:
                    continue

                ranking = build_target_ranking(cands, None)

                sample = {
                    "user_id": uid,
                    "history": history,
                    "candidates": cands,
                    "target_ranking": ranking,
                    "list_length": L,
                    "split": split_name,
                }

                if split_name == "train":
                    train.append(sample)
                elif split_name == "val":
                    val.append(sample)
                else:
                    test.append(sample)

    _log(f"Samples: train={len(train)}, val={len(val)}, test={len(test)}")
    _log_recall_metrics(recall_at)

    return train, val, test

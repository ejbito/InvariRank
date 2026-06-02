from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from tqdm.auto import tqdm

from .base import BaseDataset
from .utils import cfg_get, graded_relevance, json_loads


@dataclass(frozen=True)
class ReviewsCachePaths:
    parquet_path: str
    meta_path: str


class AmazonBooksDataset(BaseDataset):
    @classmethod
    def code(cls) -> str:
        return "amazon_books"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.reviews_path: str | None = None
        self.meta_path: str | None = None
        self.max_per_user: int | None = None
        self.selected_users: list[str] = []
        self.raw_interactions_by_user: dict[str, list[dict]] = {}
        self.metadata_raw: dict[str, dict] = {}
        self.reviews_cache: ReviewsCachePaths | None = None
        self.review_popularity: dict[str, int] | None = None
        self.review_user_counts: dict[str, int] | None = None
        self.needed_parent_asins: set[str] = set()

    def load_raw(self) -> None:
        self.reviews_path = cfg_get(self.cfg, "paths.reviews")
        self.meta_path = cfg_get(self.cfg, "paths.meta")
        use_cache = bool(cfg_get(self.cfg, "dataset.amazon.use_cache", True))
        self.max_per_user = cfg_get(self.cfg, "dataset.max_interactions_per_user", None)
        self.max_per_user = int(self.max_per_user) if self.max_per_user is not None else None

        if use_cache:
            self.reviews_cache = self.ensure_reviews_cache(self.reviews_path)
            self.ensure_review_stats()
        else:
            print("[Dataset] Reviews cache disabled; scanning JSONL")
            self.review_popularity, self.review_user_counts = self.scan_reviews_jsonl(
                self.reviews_path, build_cache=False, cache_parquet_path=None
            )

        min_total = int(cfg_get(self.cfg, "dataset.min_user_interactions", 40))
        max_users = cfg_get(self.cfg, "training.max_users", None)
        self.selected_users = select_users(self.review_user_counts or {}, min_total=min_total, max_users=max_users)
        print(f"[Dataset] Candidate users: {len(self.selected_users)}")

        selected = set(self.selected_users)
        if self.reviews_cache and self.reviews_cache.parquet_path:
            interactions = self.collect_selected_interactions_parquet(selected)
        else:
            interactions = self.collect_selected_interactions_jsonl(selected)

        self.raw_interactions_by_user = interactions
        self.trim_interactions_per_user()
        self.needed_parent_asins = {
            it["parent_asin"] for hist in self.raw_interactions_by_user.values() for it in hist if it.get("parent_asin")
        }
        print(f"[Dataset] Needed items (for metadata): {len(self.needed_parent_asins)}")

    def build_item_metadata(self) -> None:
        self.ensure_review_stats()
        popularity = defaultdict(int)
        if self.review_popularity:
            popularity.update(self.review_popularity)

        if not self.metadata_raw:
            load_all = bool(cfg_get(self.cfg, "dataset.amazon.load_all_metadata", False))
            self.metadata_raw = self.load_metadata(
                self.meta_path, keep_parent_asins=None if load_all else self.needed_parent_asins
            )

        for parent_asin, meta in tqdm(self.metadata_raw.items(), desc="[Dataset] Building item metadata"):
            self.item_metadata[parent_asin] = {
                "title": meta.get("title", ""),
                "genres": extract_categories(meta),
                "year": None,
                "popularity": int(popularity.get(parent_asin, 0)),
            }
        print(f"[Dataset] Item metadata: {len(self.item_metadata)} items")

    def build_user_histories(self) -> None:
        print("[Dataset] Building user histories")
        min_total = int(cfg_get(self.cfg, "dataset.min_user_interactions", 40))
        histories = defaultdict(list)

        for uid in self.selected_users:
            for interaction in self.raw_interactions_by_user.get(uid, []):
                parent_asin = interaction.get("parent_asin")
                meta = self.item_metadata.get(parent_asin) if parent_asin else None
                if meta is None:
                    continue
                rating = float(interaction["rating"])
                histories[uid].append(
                    {
                        "item_id": parent_asin,
                        "rating": rating,
                        "relevance": int(graded_relevance(rating)),
                        "timestamp": int(interaction["timestamp"]),
                        **meta,
                    }
                )

        self.user_histories = {}
        for uid in self.selected_users:
            history = histories.get(uid, [])
            history.sort(key=lambda x: x["timestamp"])
            if self.max_per_user is not None and len(history) > self.max_per_user:
                history = history[-self.max_per_user :]
            if len(history) >= min_total:
                self.user_histories[uid] = history

        print(f"[Dataset] User histories: {len(self.user_histories)} users")

    def cache_dir(self, reviews_path: str) -> str:
        cache_root = cfg_get(self.cfg, "paths.cache_dir", None) or cfg_get(self.cfg, "paths.output_dir", None)
        if isinstance(cache_root, str) and cache_root.strip():
            return os.path.join(cache_root, "cache")
        return os.path.join(str(Path(reviews_path).resolve().parent), ".cache")

    def ensure_reviews_cache(self, reviews_path: str) -> ReviewsCachePaths:
        try:
            import pyarrow as pa  # type: ignore
        except Exception:
            self.reviews_cache = None
            self.review_popularity, self.review_user_counts = self.scan_reviews_jsonl(
                reviews_path, build_cache=False, cache_parquet_path=None
            )
            return ReviewsCachePaths(parquet_path="", meta_path="")

        cache_dir = self.cache_dir(reviews_path)
        os.makedirs(cache_dir, exist_ok=True)
        parquet_path = os.path.join(cache_dir, f"{os.path.basename(reviews_path)}.minimal.parquet")
        tmp_path = parquet_path + ".tmp"

        if os.path.exists(parquet_path):
            print(f"[Dataset] Reviews cache: {parquet_path}")
            return ReviewsCachePaths(parquet_path=parquet_path, meta_path=parquet_path + ".meta.json")

        print("[Dataset] Building reviews cache")
        schema = pa.schema(
            [
                ("user_id", pa.string()),
                ("parent_asin", pa.string()),
                ("rating", pa.float32()),
                ("timestamp", pa.int64()),
            ]
        )
        pop, user_counts = self.scan_reviews_jsonl(
            reviews_path, build_cache=True, cache_parquet_path=tmp_path, parquet_schema=schema
        )
        if os.path.exists(parquet_path):
            os.remove(parquet_path)
        os.replace(tmp_path, parquet_path)
        self.review_popularity = pop
        self.review_user_counts = user_counts
        print(f"[Dataset] Reviews cache: {parquet_path}")
        return ReviewsCachePaths(parquet_path=parquet_path, meta_path=parquet_path + ".meta.json")

    def iter_review_batches(self, columns: list[str], batch_size: int = 200_000):
        if not self.reviews_cache or not self.reviews_cache.parquet_path:
            return iter(())
        try:
            import pyarrow.parquet as pq  # type: ignore
        except Exception:
            return iter(())
        pf = pq.ParquetFile(self.reviews_cache.parquet_path)
        return pf.iter_batches(batch_size=batch_size, columns=columns)

    def ensure_review_stats(self) -> None:
        if self.review_popularity is not None and self.review_user_counts is not None:
            return
        pop = Counter()
        user_counts = Counter()
        for batch in tqdm(self.iter_review_batches(columns=["user_id", "parent_asin"]), desc="[Dataset] Review stats"):
            cols = batch.to_pydict()
            user_counts.update([u for u in cols.get("user_id", []) if u])
            pop.update([p for p in cols.get("parent_asin", []) if p])
        self.review_popularity = dict(pop)
        self.review_user_counts = dict(user_counts)

    def scan_reviews_jsonl(
        self, reviews_path: str, build_cache: bool, cache_parquet_path: str | None, parquet_schema=None
    ):
        loads = json_loads()
        pop = Counter()
        user_counts = Counter()
        chunk_size = int(cfg_get(self.cfg, "dataset.amazon.cache_chunk_size", 200_000))
        user_ids, parent_asins, ratings, timestamps = [], [], [], []
        writer = None

        if build_cache and cache_parquet_path:
            import pyarrow.parquet as pq  # type: ignore

            writer = pq.ParquetWriter(cache_parquet_path, parquet_schema)

        def flush():
            nonlocal user_ids, parent_asins, ratings, timestamps
            if not user_ids:
                return
            pop.update(parent_asins)
            user_counts.update(user_ids)
            if writer is not None:
                import pyarrow as pa  # type: ignore

                writer.write_table(
                    pa.table(
                        {
                            "user_id": pa.array(user_ids, type=pa.string()),
                            "parent_asin": pa.array(parent_asins, type=pa.string()),
                            "rating": pa.array(ratings, type=pa.float32()),
                            "timestamp": pa.array(timestamps, type=pa.int64()),
                        }
                    )
                )
            user_ids, parent_asins, ratings, timestamps = [], [], [], []

        with open(reviews_path, "rb") as f:
            for line in tqdm(f, desc="[Dataset] Reviews"):
                if not line.strip():
                    continue
                try:
                    row = loads(line)
                except Exception:
                    continue
                uid = row.get("user_id")
                parent_asin = row.get("parent_asin")
                rating = row.get("rating")
                timestamp = row.get("sort_timestamp", row.get("timestamp"))
                if not uid or not parent_asin or rating is None or timestamp is None:
                    continue
                user_ids.append(str(uid))
                parent_asins.append(str(parent_asin))
                ratings.append(float(rating))
                timestamps.append(int(timestamp))
                if len(user_ids) >= chunk_size:
                    flush()

        flush()
        if writer is not None:
            writer.close()
        return dict(pop), dict(user_counts)

    def collect_selected_interactions_parquet(self, selected: set[str]) -> dict[str, list[dict]]:
        interactions = defaultdict(list)
        for batch in tqdm(
            self.iter_review_batches(columns=["user_id", "parent_asin", "rating", "timestamp"]),
            desc="[Dataset] Reviews (selected users)",
        ):
            cols = batch.to_pydict()
            for uid, parent_asin, rating, timestamp in zip(
                cols.get("user_id", []), cols.get("parent_asin", []), cols.get("rating", []), cols.get("timestamp", [])
            ):
                if uid in selected and parent_asin and rating is not None and timestamp is not None:
                    interactions[str(uid)].append(
                        {"parent_asin": str(parent_asin), "rating": float(rating), "timestamp": int(timestamp)}
                    )
        return interactions

    def collect_selected_interactions_jsonl(self, selected: set[str]) -> dict[str, list[dict]]:
        loads = json_loads()
        interactions = defaultdict(list)
        with open(self.reviews_path, "rb") as f:
            for line in tqdm(f, desc="[Dataset] Reviews (selected users)"):
                try:
                    row = loads(line)
                except Exception:
                    continue
                uid = row.get("user_id")
                if uid not in selected:
                    continue
                parent_asin = row.get("parent_asin")
                rating = row.get("rating")
                timestamp = row.get("sort_timestamp", row.get("timestamp"))
                if parent_asin and rating is not None and timestamp is not None:
                    interactions[str(uid)].append(
                        {"parent_asin": str(parent_asin), "rating": float(rating), "timestamp": int(timestamp)}
                    )
        return interactions

    def trim_interactions_per_user(self) -> None:
        if self.max_per_user is None:
            return
        for uid, history in list(self.raw_interactions_by_user.items()):
            if len(history) > self.max_per_user:
                history.sort(key=lambda x: x["timestamp"])
                self.raw_interactions_by_user[uid] = history[-self.max_per_user :]

    def load_metadata(self, meta_path: str, keep_parent_asins: set[str] | None) -> dict[str, dict]:
        loads = json_loads()
        keep_all = keep_parent_asins is None
        remaining = set(keep_parent_asins) if keep_parent_asins is not None else set()
        out = {}
        with open(meta_path, "rb") as f:
            for line in tqdm(f, desc="[Dataset] Metadata"):
                if not keep_all:
                    parent_asin = fast_extract_parent_asin(line)
                    if not parent_asin or parent_asin not in remaining:
                        continue
                try:
                    meta = loads(line)
                except Exception:
                    continue
                parent_asin = meta.get("parent_asin")
                if not parent_asin:
                    continue
                parent_asin = str(parent_asin)
                if not keep_all:
                    if parent_asin not in remaining:
                        continue
                    remaining.remove(parent_asin)
                out[parent_asin] = meta
                if not keep_all and not remaining:
                    break
        print(f"[Dataset] Metadata rows kept: {len(out)}")
        return out


def select_users(user_counts: dict[str, int], min_total: int, max_users) -> list[str]:
    eligible = [uid for uid, count in user_counts.items() if count >= min_total]
    if max_users is not None:
        import heapq

        return heapq.nsmallest(int(max_users), eligible)
    return sorted(eligible)


def extract_categories(meta: dict) -> list[str]:
    categories = []
    main = meta.get("main_category")
    if isinstance(main, str):
        categories.append(main)
    extra = meta.get("categories", [])
    if isinstance(extra, list):
        categories.extend([category for category in extra if isinstance(category, str)])
    return sorted(set(categories))


def fast_extract_parent_asin(line: bytes) -> str | None:
    key = b'"parent_asin"'
    idx = line.find(key)
    if idx < 0:
        return None
    idx = line.find(b":", idx + len(key))
    if idx < 0:
        return None
    idx = line.find(b'"', idx)
    if idx < 0:
        return None
    end = line.find(b'"', idx + 1)
    if end < 0:
        return None
    try:
        return line[idx + 1 : end].decode("utf-8")
    except Exception:
        return None

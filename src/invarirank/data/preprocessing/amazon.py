import json as std_json
import os
import random
from collections import defaultdict
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tqdm.auto import tqdm

from .config_utils import cfg_get
from .samplers import sample_amazon
from .utils import graded_relevance


def _json_loads():
    try:
        import ujson  # type: ignore

        return ujson.loads
    except Exception:
        return std_json.loads


def _fast_extract_parent_asin(line: bytes) -> Optional[str]:
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


@dataclass(frozen=True)
class _ReviewsCachePaths:
    parquet_path: str
    meta_path: str


class AmazonDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.seed = int(cfg_get(self.cfg, "training.seed", 42))
        self.rng = random.Random(self.seed)

        self._reviews_path: Optional[str] = None
        self._meta_path: Optional[str] = None
        self._max_per_user: Optional[int] = None
        self._selected_users: list[str] = []
        self._raw_interactions_by_user: dict[str, list[dict]] = {}
        self.metadata_raw = {}
        self._reviews_cache: Optional[_ReviewsCachePaths] = None
        self._review_popularity: Optional[dict[str, int]] = None
        self._review_user_counts: Optional[dict[str, int]] = None
        self._needed_parent_asins: set[str] = set()

        self.item_metadata: dict[str, dict] = {}
        self.user_histories: dict[str, list[dict]] = {}

    def load_raw(self):
        reviews_path = cfg_get(self.cfg, "paths.reviews")
        meta_path = cfg_get(self.cfg, "paths.meta")
        self._reviews_path = reviews_path
        self._meta_path = meta_path

        use_cache = bool(cfg_get(self.cfg, "dataset.amazon.use_cache", True))
        self._max_per_user = cfg_get(self.cfg, "dataset.max_interactions_per_user", None)
        self._max_per_user = int(self._max_per_user) if self._max_per_user is not None else None

        if use_cache:
            self._reviews_cache = self._ensure_reviews_cache(reviews_path=reviews_path)
            self._ensure_review_stats()
        else:
            print("[Dataset] Reviews cache disabled; scanning JSONL (two-pass)")
            pop, user_counts = self._scan_reviews_jsonl(
                reviews_path=reviews_path,
                build_cache=False,
                cache_parquet_path=None,
            )
            self._review_popularity = pop
            self._review_user_counts = user_counts
            print(f"[Dataset] Reviews rows: {sum(user_counts.values())}")

        min_total = int(cfg_get(self.cfg, "dataset.min_user_interactions", 40))
        max_users = cfg_get(self.cfg, "training.max_users", None)
        self._selected_users = self._select_users(
            user_counts=self._review_user_counts or {},
            min_total=min_total,
            max_users=max_users,
        )
        selected = set(self._selected_users)
        print(f"[Dataset] Candidate users: {len(self._selected_users)}")

        if self._reviews_cache and self._reviews_cache.parquet_path:
            interactions = self._collect_selected_interactions_parquet(selected)
        else:
            interactions = self._collect_selected_interactions_jsonl(selected)

        self._raw_interactions_by_user = interactions
        self._trim_interactions_per_user()
        self._needed_parent_asins = {
            it["parent_asin"] for hist in self._raw_interactions_by_user.values() for it in hist if it.get("parent_asin")
        }
        print(f"[Dataset] Needed items (for metadata): {len(self._needed_parent_asins)}")

    def build_item_metadata(self):
        self._ensure_review_stats()
        popularity = defaultdict(int)

        if self._review_popularity:
            popularity.update(self._review_popularity)

        if not self.metadata_raw:
            load_all = bool(cfg_get(self.cfg, "dataset.amazon.load_all_metadata", False))
            if not self._meta_path:
                raise RuntimeError("Amazon meta path not set; did you call load_raw()?")
            self.metadata_raw = self._load_metadata(meta_path=self._meta_path, keep_parent_asins=None if load_all else self._needed_parent_asins)

        for pid, meta in tqdm(self.metadata_raw.items(), desc="[Dataset] Building item metadata"):
            self.item_metadata[pid] = {
                "title": meta.get("title", ""),
                "genres": self._extract_categories(meta),
                "year": None,
                "popularity": int(popularity.get(pid, 0)),
            }
        print(f"[Dataset] Item metadata: {len(self.item_metadata)} items")

    @staticmethod
    def _extract_categories(meta) -> list[str]:
        cats = []
        main = meta.get("main_category")
        if isinstance(main, str):
            cats.append(main)

        extra = meta.get("categories", [])
        if isinstance(extra, list):
            cats.extend([c for c in extra if isinstance(c, str)])

        return sorted(set(cats))

    def build_user_histories(self):
        print("[Dataset] Building user histories")

        min_total = int(cfg_get(self.cfg, "dataset.min_user_interactions", 40))
        histories = defaultdict(list)
        for uid in self._selected_users:
            for it in self._raw_interactions_by_user.get(uid, []):
                pid = it.get("parent_asin")
                meta = self.item_metadata.get(pid) if pid else None
                if meta is None:
                    continue
                rating_f = float(it["rating"])
                histories[uid].append(
                    {
                        "item_id": pid,
                        "rating": rating_f,
                        "relevance": int(graded_relevance(rating_f)),
                        "timestamp": int(it["timestamp"]),
                        **meta,
                    }
                )

        self.user_histories = {}
        max_per_user = self._max_per_user
        for uid in self._selected_users:
            hist = histories.get(uid, [])
            hist.sort(key=lambda x: x["timestamp"])
            if max_per_user is not None and len(hist) > max_per_user:
                hist = hist[-max_per_user:]
            if len(hist) >= min_total:
                self.user_histories[uid] = hist

        print(f"[Dataset] User histories: {len(self.user_histories)} users")

    def generate_samples(self):
        return sample_amazon(self)

    def _cache_dir(self, reviews_path: str) -> str:
        cache_root = cfg_get(self.cfg, "paths.cache_dir", None) or cfg_get(self.cfg, "paths.output_dir", None)
        if isinstance(cache_root, str) and cache_root.strip():
            return os.path.join(cache_root, "cache")
        return os.path.join(str(Path(reviews_path).resolve().parent), ".cache")

    def _ensure_reviews_cache(self, reviews_path: str) -> _ReviewsCachePaths:
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
        except Exception:
            # Fallback: no cache support; stats computed by streaming the JSONL (histories will do a second pass).
            self._reviews_cache = None
            self._review_popularity, self._review_user_counts = self._scan_reviews_jsonl(
                reviews_path=reviews_path,
                build_cache=False,
                cache_parquet_path=None,
            )
            print(f"[Dataset] Reviews rows: {sum(self._review_user_counts.values()) if self._review_user_counts else 0}")
            return _ReviewsCachePaths(parquet_path="", meta_path="")

        cache_dir = self._cache_dir(reviews_path)
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception:
            cache_dir = os.path.join(str(Path(reviews_path).resolve().parent), ".cache")
            os.makedirs(cache_dir, exist_ok=True)

        parquet_path = os.path.join(cache_dir, f"{os.path.basename(reviews_path)}.minimal.parquet")
        meta_cache_path = parquet_path + ".meta.json"

        def _read_meta() -> Optional[dict]:
            if not os.path.exists(meta_cache_path):
                return None
            try:
                with open(meta_cache_path, encoding="utf-8") as f:
                    return std_json.load(f)
            except Exception:
                return None

        def _is_cache_valid(meta: Optional[dict]) -> bool:
            if not (meta and os.path.exists(parquet_path)):
                return False
            try:
                src_reviews_mtime = os.stat(reviews_path).st_mtime_ns
            except Exception:
                return False
            return (
                meta.get("version") == 2
                and meta.get("reviews_path") == str(Path(reviews_path).resolve())
                and meta.get("reviews_mtime_ns") == int(src_reviews_mtime)
            )

        meta = _read_meta()
        if _is_cache_valid(meta):
            self._review_popularity = None
            self._review_user_counts = None
            print(f"[Dataset] Reviews cache: {parquet_path}")
            return _ReviewsCachePaths(parquet_path=parquet_path, meta_path=meta_cache_path)

        # (Re)build cache and compute stats in one pass.
        print("[Dataset] Building reviews cache (first run can take a while)")
        schema = pa.schema(
            [
                ("user_id", pa.string()),
                ("parent_asin", pa.string()),
                ("rating", pa.float32()),
                ("timestamp", pa.int64()),
            ]
        )
        # Write to a temp file then atomically replace, to avoid leaving a partial parquet behind.
        tmp_path = parquet_path + ".tmp"
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        pop, user_counts = self._scan_reviews_jsonl(
            reviews_path=reviews_path,
            build_cache=True,
            cache_parquet_path=tmp_path,
            parquet_schema=schema,
        )

        # Finalize parquet + meta
        try:
            if os.path.exists(parquet_path):
                os.remove(parquet_path)
            os.replace(tmp_path, parquet_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        cache_meta = {
            "version": 2,
            "reviews_path": str(Path(reviews_path).resolve()),
            "reviews_mtime_ns": int(os.stat(reviews_path).st_mtime_ns),
        }
        with open(meta_cache_path, "w", encoding="utf-8") as f:
            std_json.dump(cache_meta, f)

        self._review_popularity = pop
        self._review_user_counts = user_counts
        print(f"[Dataset] Reviews cache: {parquet_path}")
        print(f"[Dataset] Reviews rows: {sum(user_counts.values())}")
        return _ReviewsCachePaths(parquet_path=parquet_path, meta_path=meta_cache_path)

    def _iter_review_batches(self, columns: list[str], batch_size: int = 200_000):
        if not self._reviews_cache or not self._reviews_cache.parquet_path:
            return iter(())
        try:
            import pyarrow.parquet as pq  # type: ignore
        except Exception:
            return iter(())

        pf = pq.ParquetFile(self._reviews_cache.parquet_path)
        return pf.iter_batches(batch_size=batch_size, columns=columns)

    def _ensure_review_stats(self):
        if self._review_popularity is not None and self._review_user_counts is not None:
            return

        pop = Counter()
        user_counts = Counter()
        for batch in tqdm(self._iter_review_batches(columns=["user_id", "parent_asin"]), desc="[Dataset] Review stats"):
            cols = batch.to_pydict()
            uids = [u for u in cols.get("user_id", []) if u]
            pids = [p for p in cols.get("parent_asin", []) if p]
            user_counts.update(uids)
            pop.update(pids)

        self._review_popularity = dict(pop)
        self._review_user_counts = dict(user_counts)

    def _scan_reviews_jsonl(
        self,
        reviews_path: str,
        build_cache: bool,
        cache_parquet_path: Optional[str],
        parquet_schema=None,
    ) -> tuple[dict[str, int], dict[str, int]]:
        loads = _json_loads()
        pop = Counter()
        user_counts = Counter()

        chunk_size = int(cfg_get(self.cfg, "dataset.amazon.cache_chunk_size", 200_000))
        flush_size = max(10_000, chunk_size)

        user_ids: list[str] = []
        parent_asins: list[str] = []
        ratings: list[float] = []
        timestamps: list[int] = []

        writer = None
        if build_cache and cache_parquet_path:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore

            writer = pq.ParquetWriter(cache_parquet_path, parquet_schema)

        def _flush():
            nonlocal user_ids, parent_asins, ratings, timestamps, writer
            if not user_ids:
                return

            pop.update(parent_asins)
            user_counts.update(user_ids)

            if writer is not None:
                import pyarrow as pa  # type: ignore

                table = pa.table(
                    {
                        "user_id": pa.array(user_ids, type=pa.string()),
                        "parent_asin": pa.array(parent_asins, type=pa.string()),
                        "rating": pa.array(ratings, type=pa.float32()),
                        "timestamp": pa.array(timestamps, type=pa.int64()),
                    }
                )
                writer.write_table(table)

            user_ids = []
            parent_asins = []
            ratings = []
            timestamps = []

        total_bytes = None
        try:
            total_bytes = os.path.getsize(reviews_path)
        except Exception:
            total_bytes = None

        with open(reviews_path, "rb") as f, tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            desc="[Dataset] Reviews",
            mininterval=1.0,
        ) as pbar:
            for line in f:
                pbar.update(len(line))
                if not line.strip():
                    continue

                try:
                    r = loads(line)
                except Exception:
                    continue

                uid = r.get("user_id")
                pid = r.get("parent_asin")
                if not uid or not pid:
                    continue

                rating = r.get("rating")
                ts = r.get("sort_timestamp")
                if ts is None:
                    ts = r.get("timestamp")
                if rating is None or ts is None:
                    continue

                try:
                    rating_f = float(rating)
                    ts_i = int(ts)
                except Exception:
                    continue

                user_ids.append(str(uid))
                parent_asins.append(str(pid))
                ratings.append(rating_f)
                timestamps.append(ts_i)

                if len(user_ids) >= flush_size:
                    _flush()

        _flush()
        if writer is not None:
            writer.close()

        return dict(pop), dict(user_counts)

    @staticmethod
    def _select_users(user_counts: dict[str, int], min_total: int, max_users):
        eligible = [uid for uid, c in user_counts.items() if c >= min_total]
        if max_users is not None:
            import heapq

            return heapq.nsmallest(int(max_users), eligible)
        return sorted(eligible)

    def _collect_selected_interactions_parquet(self, selected: set[str]) -> dict[str, list[dict]]:
        interactions: dict[str, list[dict]] = defaultdict(list)
        for batch in tqdm(
            self._iter_review_batches(columns=["user_id", "parent_asin", "rating", "timestamp"]),
            desc="[Dataset] Reviews (selected users)",
        ):
            cols = batch.to_pydict()
            for uid, pid, rating, ts in zip(
                cols.get("user_id", []),
                cols.get("parent_asin", []),
                cols.get("rating", []),
                cols.get("timestamp", []),
            ):
                if uid not in selected:
                    continue
                if not uid or not pid or rating is None or ts is None:
                    continue
                interactions[str(uid)].append(
                    {
                        "parent_asin": str(pid),
                        "rating": float(rating),
                        "timestamp": int(ts),
                    }
                )
        return interactions

    def _collect_selected_interactions_jsonl(self, selected: set[str]) -> dict[str, list[dict]]:
        if not self._reviews_path:
            raise RuntimeError("Amazon reviews path not set; did you call load_raw()?")

        loads = _json_loads()
        interactions: dict[str, list[dict]] = defaultdict(list)

        total_bytes = None
        try:
            total_bytes = os.path.getsize(self._reviews_path)
        except Exception:
            total_bytes = None

        with open(self._reviews_path, "rb") as f, tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            desc="[Dataset] Reviews (selected users)",
            mininterval=1.0,
        ) as pbar:
            for line in f:
                pbar.update(len(line))
                if not line.strip():
                    continue
                try:
                    r = loads(line)
                except Exception:
                    continue

                uid = r.get("user_id")
                if uid not in selected:
                    continue
                pid = r.get("parent_asin")
                if not pid:
                    continue
                rating = r.get("rating")
                ts = r.get("sort_timestamp")
                if ts is None:
                    ts = r.get("timestamp")
                if rating is None or ts is None:
                    continue
                try:
                    rating_f = float(rating)
                    ts_i = int(ts)
                except Exception:
                    continue
                interactions[str(uid)].append(
                    {
                        "parent_asin": str(pid),
                        "rating": rating_f,
                        "timestamp": ts_i,
                    }
                )
        return interactions

    def _trim_interactions_per_user(self):
        max_per_user = self._max_per_user
        if max_per_user is None:
            return
        for uid, hist in list(self._raw_interactions_by_user.items()):
            if len(hist) <= max_per_user:
                continue
            hist.sort(key=lambda x: x["timestamp"])
            self._raw_interactions_by_user[uid] = hist[-max_per_user:]

    def _load_metadata(self, meta_path: str, keep_parent_asins: Optional[set[str]]) -> dict[str, dict]:
        loads = _json_loads()
        keep_all = keep_parent_asins is None
        remaining = set(keep_parent_asins) if keep_parent_asins is not None else set()
        if not keep_all and not remaining:
            print("[Dataset] Metadata rows kept: 0")
            return {}
        out: dict[str, dict] = {}

        total_bytes = None
        try:
            total_bytes = os.path.getsize(meta_path)
        except Exception:
            total_bytes = None

        with open(meta_path, "rb") as f, tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            desc="[Dataset] Metadata",
            mininterval=1.0,
        ) as pbar:
            for line in f:
                pbar.update(len(line))
                if not line.strip():
                    continue

                pid = None
                if not keep_all:
                    pid = _fast_extract_parent_asin(line)
                    if not pid or pid not in remaining:
                        continue

                try:
                    meta = loads(line)
                except Exception:
                    continue

                pid2 = meta.get("parent_asin")
                if not pid2:
                    continue
                pid2 = str(pid2)
                if not keep_all:
                    if pid2 not in remaining:
                        continue
                    remaining.remove(pid2)

                out[pid2] = meta
                if not keep_all and not remaining:
                    break

        if not keep_all and remaining:
            print(f"[Dataset] Metadata missing for {len(remaining)} needed items")

        print(f"[Dataset] Metadata rows kept: {len(out)}")
        return out

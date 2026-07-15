from __future__ import annotations

import hashlib
import json
import random
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def cfg_get(config: Any, path: str, default: Any = None) -> Any:
    current = config
    for key in path.split("."):
        if current is None:
            return default
        if isinstance(current, Mapping):
            current = current.get(key)
        elif hasattr(current, key):
            current = getattr(current, key)
        else:
            return default
    return default if current is None else current


def stable_hash_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def graded_relevance(rating: float | None) -> int:
    if rating is None:
        return 0
    value = float(rating)
    if value >= 4.0:
        return 4
    if value >= 3.0:
        return 3
    if value >= 2.0:
        return 2
    if value >= 1.0:
        return 1
    return 0


def make_candidate(item_id: Any, relevance: int, metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "relevance": int(relevance),
        "title": metadata.get("title", ""),
        "genres": list(metadata.get("genres", [])),
        "year": metadata.get("year"),
        "popularity": int(metadata.get("popularity", 0)),
    }


def build_target_ranking(candidates: list[dict[str, Any]]) -> dict[str, list[Any]]:
    def key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            -int(candidate["relevance"]),
            str(candidate.get("title") or "").strip().lower(),
            str(candidate.get("year") or ""),
            str(candidate.get("item_id") or ""),
        )

    ranking = sorted(candidates, key=key)
    return {
        "item_ids": [candidate["item_id"] for candidate in ranking],
        "relevance": [int(candidate["relevance"]) for candidate in ranking],
    }


def save_jsonl(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_sample(sample: Mapping[str, Any]) -> None:
    required = {"user_id", "history", "candidates", "target_ranking", "list_length", "split"}
    missing = sorted(required - set(sample))
    if missing:
        raise ValueError(f"Dataset sample is missing field(s): {missing}")
    candidates = sample["candidates"]
    if len(candidates) != int(sample["list_length"]):
        raise ValueError(f"Expected {sample['list_length']} candidates, found {len(candidates)}")
    item_ids = [candidate["item_id"] for candidate in candidates]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Duplicate items in candidate list")
    if not all(isinstance(candidate["relevance"], int) for candidate in candidates):
        raise ValueError("All candidate relevance labels must be integers")


def write_dataset_splits(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    save_jsonl(train, output / "train.jsonl")
    save_jsonl(validation, output / "val.jsonl")
    save_jsonl(test, output / "test.jsonl")
    print(f"[Dataset] Wrote train={len(train)}, val={len(validation)}, test={len(test)} to {output}")


class BaseDataset(ABC):
    def __init__(self, config: Any):
        self.config = config
        self.seed = int(cfg_get(config, "training.seed", cfg_get(config, "seed", 42)))
        self.item_metadata: dict[Any, dict[str, Any]] = {}
        self.user_histories: dict[Any, list[dict[str, Any]]] = {}

    @classmethod
    @abstractmethod
    def code(cls) -> str:
        raise NotImplementedError

    @abstractmethod
    def load_raw(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_item_metadata(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_user_histories(self) -> None:
        raise NotImplementedError


def parse_movie_title(raw_title: str) -> tuple[str, int | None]:
    match = re.search(r"\((\d{4})\)\s*$", str(raw_title))
    year = int(match.group(1)) if match else None
    return re.sub(r"\s*\(\d{4}\)\s*$", "", str(raw_title)).strip(), year


class MovieLens32MDataset(BaseDataset):
    @classmethod
    def code(cls) -> str:
        return "movielens32m"

    def __init__(self, config: Any):
        super().__init__(config)
        self.ratings = None
        self.movies = None

    def load_raw(self) -> None:
        import pandas as pd

        ratings_path = cfg_get(self.config, "paths.ratings")
        movies_path = cfg_get(self.config, "paths.movies")
        if not ratings_path or not movies_path:
            raise ValueError("MovieLens configuration requires paths.ratings and paths.movies.")
        self.ratings = pd.read_csv(ratings_path)
        self.movies = pd.read_csv(movies_path)
        print(f"[Dataset] Ratings rows: {len(self.ratings)}; movies rows: {len(self.movies)}")

    def build_item_metadata(self) -> None:
        from tqdm.auto import tqdm

        popularity = self.ratings["movieId"].value_counts().to_dict()
        for row in tqdm(self.movies.itertuples(index=False), desc="[Dataset] Movies"):
            title, year = parse_movie_title(row.title)
            self.item_metadata[int(row.movieId)] = {
                "title": title,
                "genres": row.genres.split("|") if isinstance(row.genres, str) else [],
                "year": year,
                "popularity": int(popularity.get(row.movieId, 0)),
            }

    def build_user_histories(self) -> None:
        from tqdm.auto import tqdm

        minimum = int(cfg_get(self.config, "dataset.min_user_interactions", 50))
        maximum_users = cfg_get(self.config, "training.max_users", 5000)
        maximum_per_user = cfg_get(self.config, "dataset.max_interactions_per_user")
        counts = self.ratings["userId"].value_counts()
        users = sorted(counts[counts >= minimum].index.tolist())
        if maximum_users is not None:
            users = users[: int(maximum_users)]
        frame = self.ratings[self.ratings["userId"].isin(users)].sort_values(["userId", "timestamp"])
        for user_id, group in tqdm(frame.groupby("userId", sort=True), desc="[Dataset] Users"):
            rows = list(group.itertuples(index=False))
            if maximum_per_user is not None:
                rows = rows[-int(maximum_per_user) :]
            history = []
            for row in rows:
                item_id = int(row.movieId)
                if item_id not in self.item_metadata:
                    continue
                metadata = self.item_metadata[item_id]
                rating = float(row.rating)
                history.append(
                    {
                        "item_id": item_id,
                        "relevance": graded_relevance(rating),
                        "title": metadata["title"],
                        "genres": list(metadata["genres"]),
                        "year": metadata["year"],
                        "popularity": metadata["popularity"],
                        "rating": rating,
                        "timestamp": int(row.timestamp),
                    }
                )
            if len(history) >= minimum:
                self.user_histories[int(user_id)] = history


def _json_loads(line: bytes | str) -> Any:
    try:
        import ujson

        return ujson.loads(line)
    except ImportError:
        return json.loads(line)


def select_users(user_counts: Mapping[str, int], minimum: int, maximum: int | None) -> list[str]:
    eligible = sorted(user_id for user_id, count in user_counts.items() if count >= minimum)
    return eligible if maximum is None else eligible[: int(maximum)]


def extract_categories(metadata: Mapping[str, Any]) -> list[str]:
    categories = []
    if isinstance(metadata.get("main_category"), str):
        categories.append(metadata["main_category"])
    if isinstance(metadata.get("categories"), list):
        categories.extend(value for value in metadata["categories"] if isinstance(value, str))
    return sorted(set(categories))


def fast_extract_parent_asin(line: bytes) -> str | None:
    key = b'"parent_asin"'
    index = line.find(key)
    if index < 0 or (index := line.find(b":", index + len(key))) < 0:
        return None
    index = line.find(b'"', index)
    end = line.find(b'"', index + 1) if index >= 0 else -1
    if index < 0 or end < 0:
        return None
    try:
        return line[index + 1 : end].decode("utf-8")
    except UnicodeDecodeError:
        return None


class AmazonBooksDataset(BaseDataset):
    @classmethod
    def code(cls) -> str:
        return "amazon_books"

    def __init__(self, config: Any):
        super().__init__(config)
        self.reviews_path = ""
        self.metadata_path = ""
        self.raw_interactions: dict[str, list[dict[str, Any]]] = {}
        self.popularity: dict[str, int] = {}
        self.selected_users: list[str] = []

    def load_raw(self) -> None:
        from tqdm.auto import tqdm

        self.reviews_path = str(cfg_get(self.config, "paths.reviews", ""))
        self.metadata_path = str(cfg_get(self.config, "paths.meta", ""))
        if not self.reviews_path or not self.metadata_path:
            raise ValueError("Amazon Books configuration requires paths.reviews and paths.meta.")
        user_counts: Counter[str] = Counter()
        popularity: Counter[str] = Counter()
        cache_path = self._ensure_reviews_cache() if cfg_get(self.config, "dataset.amazon.use_cache", True) else None
        if cache_path is not None:
            for batch in tqdm(self._iter_cache(cache_path, ["user_id", "parent_asin"]), desc="[Dataset] Stats"):
                columns = batch.to_pydict()
                user_counts.update(value for value in columns["user_id"] if value)
                popularity.update(value for value in columns["parent_asin"] if value)
        else:
            for row in tqdm(self._iter_review_json(), desc="[Dataset] Review stats"):
                user_counts[row["user_id"]] += 1
                popularity[row["parent_asin"]] += 1

        minimum = int(cfg_get(self.config, "dataset.min_user_interactions", 40))
        maximum_users = cfg_get(self.config, "training.max_users")
        self.selected_users = select_users(user_counts, minimum, maximum_users)
        selected = set(self.selected_users)
        interactions: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if cache_path is not None:
            batches = self._iter_cache(cache_path, ["user_id", "parent_asin", "rating", "timestamp"])
            for batch in tqdm(batches, desc="[Dataset] Selected reviews"):
                columns = batch.to_pydict()
                for user_id, parent_asin, rating, timestamp in zip(
                    columns["user_id"],
                    columns["parent_asin"],
                    columns["rating"],
                    columns["timestamp"],
                ):
                    if user_id in selected:
                        interactions[user_id].append(
                            {
                                "parent_asin": parent_asin,
                                "rating": float(rating),
                                "timestamp": int(timestamp),
                            }
                        )
        else:
            for row in tqdm(self._iter_review_json(), desc="[Dataset] Selected reviews"):
                if row["user_id"] in selected:
                    interactions[row["user_id"]].append(
                        {
                            "parent_asin": row["parent_asin"],
                            "rating": row["rating"],
                            "timestamp": row["timestamp"],
                        }
                    )
        maximum_per_user = cfg_get(self.config, "dataset.max_interactions_per_user")
        for user_id in self.selected_users:
            history = sorted(interactions[user_id], key=lambda value: value["timestamp"])
            if maximum_per_user is not None:
                history = history[-int(maximum_per_user) :]
            self.raw_interactions[user_id] = history
        self.popularity = dict(popularity)

    def _iter_review_json(self):
        with open(self.reviews_path, "rb") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    raw = _json_loads(line)
                except (ValueError, TypeError):
                    continue
                user_id = raw.get("user_id")
                parent_asin = raw.get("parent_asin")
                rating = raw.get("rating")
                timestamp = raw.get("sort_timestamp", raw.get("timestamp"))
                if user_id and parent_asin and rating is not None and timestamp is not None:
                    yield {
                        "user_id": str(user_id),
                        "parent_asin": str(parent_asin),
                        "rating": float(rating),
                        "timestamp": int(timestamp),
                    }

    def _cache_path(self) -> Path:
        root = cfg_get(self.config, "paths.cache_dir") or cfg_get(self.config, "paths.output_dir")
        directory = Path(root) / "cache" if root else Path(self.reviews_path).resolve().parent / ".cache"
        return directory / f"{Path(self.reviews_path).name}.minimal.parquet"

    def _ensure_reviews_cache(self) -> Path | None:
        try:
            import pyarrow as arrow
            import pyarrow.parquet as parquet
        except ImportError:
            return None
        path = self._cache_path()
        if path.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        schema = arrow.schema(
            [
                ("user_id", arrow.string()),
                ("parent_asin", arrow.string()),
                ("rating", arrow.float32()),
                ("timestamp", arrow.int64()),
            ]
        )
        writer = parquet.ParquetWriter(temporary, schema)
        chunk_size = int(cfg_get(self.config, "dataset.amazon.cache_chunk_size", 200_000))
        columns: dict[str, list[Any]] = {name: [] for name in schema.names}

        def flush() -> None:
            if not columns["user_id"]:
                return
            writer.write_table(arrow.table(columns, schema=schema))
            for values in columns.values():
                values.clear()

        try:
            for row in self._iter_review_json():
                for name in schema.names:
                    columns[name].append(row[name])
                if len(columns["user_id"]) >= chunk_size:
                    flush()
            flush()
        finally:
            writer.close()
        temporary.replace(path)
        return path

    @staticmethod
    def _iter_cache(path: Path, columns: list[str]):
        import pyarrow.parquet as parquet

        return parquet.ParquetFile(path).iter_batches(batch_size=200_000, columns=columns)

    def build_item_metadata(self) -> None:
        from tqdm.auto import tqdm

        required = {row["parent_asin"] for history in self.raw_interactions.values() for row in history}
        with open(self.metadata_path, "rb") as handle:
            for line in tqdm(handle, desc="[Dataset] Metadata"):
                parent_asin = fast_extract_parent_asin(line)
                if not parent_asin or parent_asin not in required:
                    continue
                try:
                    metadata = _json_loads(line)
                except (ValueError, TypeError):
                    continue
                self.item_metadata[parent_asin] = {
                    "title": metadata.get("title", ""),
                    "genres": extract_categories(metadata),
                    "year": None,
                    "popularity": int(self.popularity.get(parent_asin, 0)),
                }
                required.remove(parent_asin)
                if not required:
                    break

    def build_user_histories(self) -> None:
        minimum = int(cfg_get(self.config, "dataset.min_user_interactions", 40))
        for user_id in self.selected_users:
            history = []
            for interaction in self.raw_interactions.get(user_id, []):
                item_id = interaction["parent_asin"]
                if item_id not in self.item_metadata:
                    continue
                rating = float(interaction["rating"])
                history.append(
                    {
                        "item_id": item_id,
                        "rating": rating,
                        "relevance": graded_relevance(rating),
                        "timestamp": int(interaction["timestamp"]),
                        **self.item_metadata[item_id],
                    }
                )
            if len(history) >= minimum:
                self.user_histories[user_id] = history


class LightGCNRetriever:
    def __init__(self, config: Any):
        import numpy as np
        import torch

        self.config = config
        self.seed = int(cfg_get(config, "training.seed", 42))
        self.embedding_dim = int(cfg_get(config, "retrieval.embedding_dim", 128))
        self.num_layers = int(cfg_get(config, "retrieval.num_layers", 3))
        self.epochs = int(cfg_get(config, "retrieval.epochs", 100))
        self.learning_rate = float(cfg_get(config, "retrieval.learning_rate", 1e-3))
        self.regularization = float(cfg_get(config, "retrieval.reg", 1e-5))
        self.samples_per_epoch = int(cfg_get(config, "retrieval.edge_samples_per_epoch", 3_000_000))
        self.batch_size = int(cfg_get(config, "retrieval.batch_size", 8192))
        self.negatives = int(cfg_get(config, "retrieval.negatives_per_positive", 4))
        self.rejection_tries = int(cfg_get(config, "retrieval.neg_rejection_max_tries", 10))
        self.filter_seen = bool(cfg_get(config, "retrieval.filter_seen", True))
        self.maximum_k = int(cfg_get(config, "retrieval.k_max", 1000))
        self.edge_dropout = float(cfg_get(config, "retrieval.edge_dropout", 0.0))
        self.hard_negative_ratio = float(cfg_get(config, "retrieval.hard_negative_ratio", 0.5))
        self.hard_candidate_pool = int(cfg_get(config, "retrieval.hard_candidate_pool", 32))
        self.normalize_embeddings = bool(cfg_get(config, "retrieval.normalize_embeddings", True))
        deterministic = bool(cfg_get(config, "retrieval.deterministic", False))
        self.deterministic = deterministic
        use_cuda = bool(cfg_get(config, "retrieval.use_cuda", True)) and not deterministic
        self.use_amp = bool(cfg_get(config, "retrieval.use_amp", True)) and not deterministic
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.generator = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)

        self.user_to_index: dict[Any, int] = {}
        self.index_to_item: dict[int, Any] = {}
        self.item_to_index: dict[Any, int] = {}
        self.user_positive_sets: dict[int, set[int]] = {}
        self.user_seen_indices: dict[int, Any] = {}
        self.edge_users = None
        self.edge_items = None
        self.base_edge_users = None
        self.base_edge_items = None
        self.base_edge_weights = None
        self.user_to_item = None
        self.item_to_user = None
        self.user_embedding = None
        self.item_embedding = None
        self.user_factors = None
        self.item_factors = None
        self.num_users = 0
        self.num_items = 0

    def fit(self, interactions: Iterable[tuple[Any, Any]]) -> LightGCNRetriever:
        import numpy as np
        from scipy.sparse import csr_matrix
        from tqdm.auto import tqdm

        edges = list(dict.fromkeys(interactions))
        if not edges:
            raise ValueError("LightGCNRetriever requires at least one interaction")
        users = sorted({user for user, _ in edges})
        items = sorted({item for _, item in edges})
        self.user_to_index = {user: index for index, user in enumerate(users)}
        self.item_to_index = {item: index for index, item in enumerate(items)}
        self.index_to_item = {index: item for item, index in self.item_to_index.items()}
        self.num_users = len(users)
        self.num_items = len(items)
        rows = np.empty(len(edges), dtype=np.int64)
        columns = np.empty(len(edges), dtype=np.int64)
        positive_sets: dict[int, set[int]] = defaultdict(set)
        for edge_index, (raw_user, raw_item) in enumerate(tqdm(edges, desc="[LightGCN] Encoding edges")):
            user = self.user_to_index[raw_user]
            item = self.item_to_index[raw_item]
            rows[edge_index] = user
            columns[edge_index] = item
            positive_sets[user].add(item)
        self.edge_users = rows
        self.edge_items = columns
        self.user_positive_sets = dict(positive_sets)
        self.user_seen_indices = {
            user: np.asarray(sorted(seen), dtype=np.int64) for user, seen in positive_sets.items()
        }
        matrix = csr_matrix(
            (np.ones(len(edges), dtype=np.float32), (rows, columns)),
            shape=(self.num_users, self.num_items),
        )
        self._prepare_graph(matrix)
        self._refresh_adjacency()
        self._initialize_embeddings()
        self._train()
        self._materialize()
        return self

    def _prepare_graph(self, matrix: Any) -> None:
        import numpy as np

        matrix = matrix.tocsr()
        user_degree = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
        item_degree = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float32)
        user_degree[user_degree == 0] = 1.0
        item_degree[item_degree == 0] = 1.0
        coordinates = matrix.tocoo()
        self.base_edge_users = coordinates.row.astype(np.int64)
        self.base_edge_items = coordinates.col.astype(np.int64)
        self.base_edge_weights = (1.0 / np.sqrt(user_degree[coordinates.row] * item_degree[coordinates.col])).astype(
            np.float32
        )

    def _refresh_adjacency(self, keep_probability: float = 1.0) -> None:
        import numpy as np
        import torch

        users = self.base_edge_users
        items = self.base_edge_items
        weights = self.base_edge_weights
        if keep_probability < 1.0:
            keep = self.generator.random(len(users)) < keep_probability
            if not keep.any():
                keep[self.generator.integers(0, len(users))] = True
            users = users[keep]
            items = items[keep]
            weights = weights[keep] / keep_probability
        values = torch.tensor(weights, dtype=torch.float32, device=self.device)
        self.user_to_item = torch.sparse_coo_tensor(
            torch.tensor(np.vstack([users, items]), dtype=torch.int64, device=self.device),
            values,
            (self.num_users, self.num_items),
        ).coalesce()
        self.item_to_user = torch.sparse_coo_tensor(
            torch.tensor(np.vstack([items, users]), dtype=torch.int64, device=self.device),
            values,
            (self.num_items, self.num_users),
        ).coalesce()

    def _initialize_embeddings(self) -> None:
        import torch

        self.user_embedding = torch.nn.Embedding(self.num_users, self.embedding_dim, device=self.device)
        self.item_embedding = torch.nn.Embedding(self.num_items, self.embedding_dim, device=self.device)
        torch.nn.init.xavier_uniform_(self.user_embedding.weight)
        torch.nn.init.xavier_uniform_(self.item_embedding.weight)

    def _propagate(self):
        import torch

        user_layers = [self.user_embedding.weight]
        item_layers = [self.item_embedding.weight]
        current_users = user_layers[0]
        current_items = item_layers[0]
        for _ in range(self.num_layers):
            next_users = torch.sparse.mm(self.user_to_item, current_items)
            next_items = torch.sparse.mm(self.item_to_user, current_users)
            user_layers.append(next_users)
            item_layers.append(next_items)
            current_users, current_items = next_users, next_items
        return torch.stack(user_layers).mean(dim=0), torch.stack(item_layers).mean(dim=0)

    def _random_negatives(self, users: Any, count: int):
        import numpy as np

        negatives = self.generator.integers(0, self.num_items, size=(len(users), count), dtype=np.int64)
        for row, user in enumerate(users):
            positives = self.user_positive_sets[int(user)]
            for column in range(count):
                attempts = 0
                while int(negatives[row, column]) in positives:
                    negatives[row, column] = self.generator.integers(0, self.num_items)
                    attempts += 1
                    if attempts > self.rejection_tries and len(positives) >= self.num_items:
                        raise ValueError("A user has interacted with every retriever item; negatives are unavailable.")
        return negatives

    def _hard_negatives(self, users: Any, user_factors: Any, item_factors: Any, count: int):
        import numpy as np
        import torch

        pool_size = max(self.hard_candidate_pool, count)
        candidates = self.generator.integers(
            0,
            self.num_items,
            size=(len(users), pool_size),
            dtype=np.int64,
        )
        user_tensor = torch.from_numpy(users).to(self.device)
        candidate_tensor = torch.from_numpy(candidates).to(self.device)
        with torch.no_grad():
            scores = (user_factors[user_tensor].unsqueeze(1) * item_factors[candidate_tensor]).sum(dim=2)
        output = np.empty((len(users), count), dtype=np.int64)
        for row, user in enumerate(users):
            positives = self.user_positive_sets[int(user)]
            valid = [index for index, item in enumerate(candidates[row]) if int(item) not in positives]
            valid.sort(key=lambda index: float(scores[row, index]), reverse=True)
            picked = candidates[row, valid[:count]].tolist()
            if len(picked) < count:
                picked.extend(self._random_negatives(users[row : row + 1], count - len(picked))[0].tolist())
            output[row] = picked[:count]
        return output

    def _sample_negatives(self, users: Any, user_factors: Any, item_factors: Any):
        import numpy as np

        hard_count = max(0, min(self.negatives, round(self.negatives * self.hard_negative_ratio)))
        random_count = self.negatives - hard_count
        parts = []
        if hard_count:
            parts.append(self._hard_negatives(users, user_factors, item_factors, hard_count))
        if random_count:
            parts.append(self._random_negatives(users, random_count))
        return np.concatenate(parts, axis=1)

    def _train(self) -> None:
        import torch
        import torch.nn.functional as functional
        from tqdm.auto import tqdm

        optimizer = torch.optim.Adam(
            list(self.user_embedding.parameters()) + list(self.item_embedding.parameters()),
            lr=self.learning_rate,
        )
        edge_count = len(self.edge_users)
        progress = tqdm(range(self.epochs), desc="[LightGCN] Training")
        for _ in progress:
            self._refresh_adjacency(1.0 - self.edge_dropout if self.edge_dropout else 1.0)
            optimizer.zero_grad(set_to_none=True)
            all_users, all_items = self._propagate()
            sample_count = min(self.samples_per_epoch, edge_count)
            indices = self.generator.integers(0, edge_count, size=sample_count, dtype="int64")
            positive_users = self.edge_users[indices]
            positive_items = self.edge_items[indices]
            losses = []
            for start in range(0, sample_count, self.batch_size):
                end = min(start + self.batch_size, sample_count)
                users_array = positive_users[start:end]
                items_array = positive_items[start:end]
                negatives_array = self._sample_negatives(users_array, all_users, all_items)
                users = torch.from_numpy(users_array).to(self.device)
                items = torch.from_numpy(items_array).to(self.device)
                negatives = torch.from_numpy(negatives_array).to(self.device)
                user_vectors = all_users[users]
                item_vectors = all_items[items]
                negative_vectors = all_items[negatives]
                positive_scores = (user_vectors * item_vectors).sum(dim=1, keepdim=True)
                negative_scores = (user_vectors.unsqueeze(1) * negative_vectors).sum(dim=2)
                bpr = -functional.logsigmoid(positive_scores - negative_scores).mean()
                regularization = self.regularization * (
                    user_vectors.pow(2).sum(dim=1).mean()
                    + item_vectors.pow(2).sum(dim=1).mean()
                    + negative_vectors.pow(2).sum(dim=2).mean()
                )
                losses.append(bpr + regularization)
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=float(loss.detach().cpu()))

    def _materialize(self) -> None:
        import torch
        import torch.nn.functional as functional

        with torch.no_grad():
            self._refresh_adjacency()
            users, items = self._propagate()
            if self.normalize_embeddings:
                users = functional.normalize(users, dim=1)
                items = functional.normalize(items, dim=1)
        self.user_factors = users.detach().float().cpu().numpy()
        self.item_factors = items.detach().float().cpu().numpy()

    def retrieve(self, user_id: Any, k: int) -> list[Any]:
        import numpy as np

        if user_id not in self.user_to_index or k <= 0 or self.user_factors is None:
            return []
        user = self.user_to_index[user_id]
        scores = self.item_factors @ self.user_factors[user]
        if self.filter_seen:
            scores[self.user_seen_indices.get(user, [])] = -np.inf
        finite = np.isfinite(scores)
        limit = min(k, self.maximum_k, int(finite.sum()))
        if limit <= 0:
            return []
        indices = np.argpartition(-scores, limit - 1)[:limit]
        indices = indices[np.argsort(-scores[indices])]
        return [self.index_to_item[int(index)] for index in indices]


def split_user_histories(
    user_histories: Mapping[Any, list[dict[str, Any]]],
    history_length: int,
    train_percentage: float,
    validation_percentage: float,
    train_future_percentage: float,
) -> dict[Any, tuple[list[dict[str, Any]], ...]]:
    splits = {}
    for user_id in sorted(user_histories):
        history = sorted(user_histories[user_id], key=lambda value: value["timestamp"])
        train_count = int(len(history) * train_percentage)
        validation_count = int(len(history) * validation_percentage)
        test_count = len(history) - train_count - validation_count
        if train_count <= 0 or validation_count <= 0 or test_count <= 0:
            continue
        train = history[:train_count]
        validation = history[train_count : train_count + validation_count]
        test = history[train_count + validation_count :]
        future_count = max(1, int(len(train) * train_future_percentage))
        if len(train) - future_count < history_length:
            continue
        past_train = train[:-future_count]
        future_train = train[-future_count:]
        if future_train and validation and test:
            splits[user_id] = (past_train, future_train, validation, test)
    return splits


def build_train_interactions(
    splits: Mapping[Any, tuple[list[dict[str, Any]], ...]],
    minimum_rating: float | None,
) -> list[tuple[Any, Any]]:
    return [
        (user_id, interaction["item_id"])
        for user_id, (past_train, *_rest) in splits.items()
        for interaction in past_train
        if minimum_rating is None or float(interaction.get("rating", 0.0)) >= minimum_rating
    ]


def fill_candidates(
    candidates: list[dict[str, Any]],
    banned: set[Any],
    all_items: list[Any],
    metadata: Mapping[Any, Mapping[str, Any]],
    list_size: int,
    deterministic: bool,
    seed_key: str,
) -> None:
    available = [item for item in all_items if item not in banned]
    if not deterministic:
        random.Random(stable_hash_int(seed_key)).shuffle(available)
    for item_id in available:
        if len(candidates) >= list_size:
            break
        candidates.append(make_candidate(item_id, 0, metadata[item_id]))
        banned.add(item_id)
    if len(candidates) < list_size:
        raise ValueError(f"Only {len(candidates)} unique candidates are available for list size {list_size}.")


def append_sample(
    outputs: dict[str, list[dict[str, Any]]],
    split: str,
    user_id: Any,
    history: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    list_size: int,
) -> None:
    outputs[split].append(
        {
            "user_id": user_id,
            "history": history,
            "candidates": candidates,
            "target_ranking": build_target_ranking(candidates),
            "list_length": list_size,
            "split": split,
        }
    )


def _sampling_settings(dataset: BaseDataset) -> tuple[int, list[int], bool, dict[Any, tuple[list[dict], ...]]]:
    config = dataset.config
    history_length = min(
        int(cfg_get(config, "split.history_length", 20)),
        int(cfg_get(config, "reranking.max_history_items", 20)),
    )
    list_sizes = list(cfg_get(config, "sampling.list_sizes", [15, 25, 50]))
    deterministic = bool(cfg_get(config, "sampling.deterministic", True))
    splits = split_user_histories(
        dataset.user_histories,
        history_length,
        float(cfg_get(config, "split.train_pct", 0.7)),
        float(cfg_get(config, "split.val_pct", 0.1)),
        float(cfg_get(config, "split.train_future_pct", 0.2)),
    )
    return history_length, list_sizes, deterministic, splits


def sample_movielens(dataset: BaseDataset) -> tuple[list[dict], list[dict], list[dict]]:
    from tqdm.auto import tqdm

    history_length, list_sizes, deterministic, splits = _sampling_settings(dataset)
    config = dataset.config
    seed = int(cfg_get(config, "training.seed", 42))
    minimum_rating = float(cfg_get(config, "dataset.implicit_min_rating", 4.0))
    retrieval_pool = min(300, int(cfg_get(config, "retrieval.k_max", 1500)))
    retriever = LightGCNRetriever(config).fit(build_train_interactions(splits, minimum_rating))
    all_items = sorted(dataset.item_metadata)
    outputs: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for user_id in tqdm(sorted(splits), desc="[Sampling] Users"):
        past_train, future_train, future_validation, future_test = splits[user_id]
        retrieved = retriever.retrieve(user_id, retrieval_pool)
        if not retrieved:
            continue
        split_values = {
            "train": (past_train, future_train),
            "val": (past_train + future_train, future_validation),
            "test": (past_train + future_train + future_validation, future_test),
        }
        for split, (past, future) in split_values.items():
            history = past[-history_length:]
            history_ids = {item["item_id"] for item in history}
            all_positives = [
                (item["item_id"], int(item["relevance"]))
                for item in sorted(future, key=lambda value: (-value["rating"], -value["timestamp"]))
                if int(item["relevance"]) > 0 and item["item_id"] in dataset.item_metadata
            ]
            if not all_positives:
                continue
            if deterministic:
                positives = all_positives[:3]
            else:
                generator = random.Random(stable_hash_int(f"{seed}-{user_id}-{split}-pos"))
                draw = generator.random()
                positive_count = 1 if draw <= 0.25 else 2 if draw <= 0.70 else 3
                positive_count = min(positive_count, len(all_positives))
                weights = [max(relevance, 1) for _, relevance in all_positives]
                indices = list(
                    dict.fromkeys(generator.choices(range(len(all_positives)), weights=weights, k=positive_count))
                )
                remaining = [index for index in range(len(all_positives)) if index not in indices]
                generator.shuffle(remaining)
                indices.extend(remaining[: positive_count - len(indices)])
                positives = [all_positives[index] for index in indices]
            hard_pool = [item for item in retrieved[3:100] if item not in history_ids and item in dataset.item_metadata]
            for list_size in list_sizes:
                candidates = [
                    make_candidate(item_id, relevance, dataset.item_metadata[item_id])
                    for item_id, relevance in positives
                ]
                banned = history_ids | {item_id for item_id, _ in positives}
                minimum_hard = max(6, list_size // 3)
                local_pool = list(hard_pool)
                if not deterministic:
                    random.Random(stable_hash_int(f"{seed}-{user_id}-{split}-neg")).shuffle(local_pool)
                for item_id in local_pool:
                    if len(candidates) >= len(positives) + minimum_hard:
                        break
                    if item_id not in banned:
                        candidates.append(make_candidate(item_id, 0, dataset.item_metadata[item_id]))
                        banned.add(item_id)
                fill_candidates(
                    candidates,
                    banned,
                    all_items,
                    dataset.item_metadata,
                    list_size,
                    deterministic,
                    f"{seed}-{user_id}-{split}-fill",
                )
                append_sample(outputs, split, user_id, history, candidates, list_size)
    return outputs["train"], outputs["val"], outputs["test"]


def sample_amazon_books(dataset: BaseDataset) -> tuple[list[dict], list[dict], list[dict]]:
    from tqdm.auto import tqdm

    history_length, list_sizes, deterministic, splits = _sampling_settings(dataset)
    config = dataset.config
    seed = int(cfg_get(config, "training.seed", 42))
    minimum_positives = int(cfg_get(config, "sampling.min_future_positives", 1))
    require_retrieved = bool(cfg_get(config, "sampling.amazon.require_retrieved_positive", True))
    retriever = LightGCNRetriever(config).fit(build_train_interactions(splits, None))
    outputs: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    all_items = sorted(dataset.item_metadata)

    for user_id in tqdm(sorted(splits), desc="[Sampling] Users"):
        past_train, future_train, future_validation, future_test = splits[user_id]
        future_all = future_train + future_validation + future_test
        future_ids = {item["item_id"] for item in future_all}
        retrieved = retriever.retrieve(user_id, max(list_sizes) * 10)
        negative_pool = [item for item in retrieved if item not in future_ids and item in dataset.item_metadata]
        split_values = {
            "train": (past_train, future_train),
            "val": (past_train + future_train, future_validation),
            "test": (past_train + future_train + future_validation, future_test),
        }
        for split, (past, future) in split_values.items():
            history = past[-history_length:]
            if require_retrieved:
                local_future_ids = {item["item_id"] for item in future}
                positive_ids = [item for item in retrieved if item in local_future_ids][:minimum_positives]
            else:
                positive_ids = [
                    item["item_id"]
                    for item in sorted(future, key=lambda value: (-value["rating"], -value["timestamp"]))
                    if int(item["relevance"]) >= 2 and item["item_id"] in dataset.item_metadata
                ][:minimum_positives]
            if len(positive_ids) < minimum_positives:
                continue
            for list_size in list_sizes:
                candidates = []
                banned = set(positive_ids)
                for item_id in positive_ids:
                    rating = next(item["rating"] for item in future if item["item_id"] == item_id)
                    candidates.append(make_candidate(item_id, graded_relevance(rating), dataset.item_metadata[item_id]))
                local_pool = list(negative_pool)
                if not deterministic:
                    random.Random(stable_hash_int(f"{seed}-{user_id}-{split}-neg")).shuffle(local_pool)
                for item_id in local_pool:
                    if len(candidates) >= list_size:
                        break
                    if item_id not in banned:
                        candidates.append(make_candidate(item_id, 0, dataset.item_metadata[item_id]))
                        banned.add(item_id)
                fill_candidates(
                    candidates,
                    banned,
                    all_items,
                    dataset.item_metadata,
                    list_size,
                    deterministic,
                    f"{seed}-{user_id}-{split}-fill",
                )
                append_sample(outputs, split, user_id, history, candidates, list_size)
    return outputs["train"], outputs["val"], outputs["test"]


DATASETS: dict[str, type[BaseDataset]] = {
    "movielens": MovieLens32MDataset,
    "movielens32m": MovieLens32MDataset,
    "amazon": AmazonBooksDataset,
    "amazon_books": AmazonBooksDataset,
}


def build_dataset_splits(config: Any) -> tuple[list[dict], list[dict], list[dict]]:
    name = str(cfg_get(config, "dataset.name", "")).lower()
    if name not in DATASETS:
        raise ValueError(f"Unsupported dataset: {name}. Expected one of {sorted(DATASETS)}")
    dataset = DATASETS[name](config)
    dataset.load_raw()
    dataset.build_item_metadata()
    dataset.build_user_histories()
    splits = sample_amazon_books(dataset) if dataset.code() == "amazon_books" else sample_movielens(dataset)
    for split in splits:
        for sample in split:
            validate_sample(sample)
    return splits


__all__ = [
    "AmazonBooksDataset",
    "BaseDataset",
    "DATASETS",
    "LightGCNRetriever",
    "MovieLens32MDataset",
    "build_dataset_splits",
    "build_target_ranking",
    "cfg_get",
    "graded_relevance",
    "parse_movie_title",
    "save_jsonl",
    "split_user_histories",
    "validate_sample",
    "write_dataset_splits",
]

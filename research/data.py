from __future__ import annotations

import hashlib
import json
import random
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol


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


def candidate_record_issues(sample: Mapping[str, Any]) -> list[str]:
    """Return schema and contract issues for one generated candidate-list record."""
    issues = []
    try:
        validate_sample(sample)
    except Exception as error:
        issues.append(f"invalid_sample: {type(error).__name__}: {error}")
        return issues

    candidates = list(sample["candidates"])
    candidate_ids = [candidate.get("item_id") for candidate in candidates]
    history_ids = {item.get("item_id") for item in sample.get("history", [])}
    if any(item_id in history_ids for item_id in candidate_ids):
        issues.append("candidate_overlaps_history")
    if not any(int(candidate.get("relevance", 0)) > 0 for candidate in candidates):
        issues.append("missing_positive_candidate")
    for index, candidate in enumerate(candidates):
        title = candidate.get("title", candidate.get("name", ""))
        if title is None or not str(title).strip():
            issues.append(f"candidate_{index}_missing_title")
        relevance = candidate.get("relevance")
        if isinstance(relevance, bool) or not isinstance(relevance, int) or relevance < 0:
            issues.append(f"candidate_{index}_invalid_relevance")
    retrieval = sample.get("retrieval")
    if retrieval is not None:
        if not isinstance(retrieval, Mapping):
            issues.append("retrieval_provenance_invalid")
        elif not retrieval.get("backend"):
            issues.append("retrieval_provenance_missing_backend")
    return issues


def validate_candidate_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate generated candidate-list records and return an aggregate report."""
    issue_counts: Counter[str] = Counter()
    total = 0
    valid = 0
    for index, record in enumerate(records):
        total += 1
        issues = candidate_record_issues(record)
        if issues:
            issue_counts.update(issues)
            issue_counts.update({f"record_{index}_invalid": 1})
        else:
            valid += 1
    return {
        "records": total,
        "valid_records": valid,
        "invalid_records": total - valid,
        "issues": dict(sorted(issue_counts.items())),
    }


def retrieval_provenance(config: Any) -> dict[str, Any]:
    """Describe the first-stage retriever configuration stored with candidate samples."""
    backend = retrieval_backend(config)
    provenance = {
        "backend": backend,
        "model": cfg_get(config, "retrieval.model"),
        "k_max": cfg_get(config, "retrieval.k_max"),
        "filter_seen": cfg_get(config, "retrieval.filter_seen", True),
        "seed": cfg_get(config, "retrieval.seed", cfg_get(config, "training.seed", 42)),
    }
    if backend == RECBOLE_BACKEND:
        provenance.update(
            {
                "recbole_dataset_name": cfg_get(config, "retrieval.recbole_dataset_name"),
                "recbole_dir": cfg_get(config, "retrieval.recbole_dir"),
            }
        )
    return {key: value for key, value in provenance.items() if value is not None}


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


class FirstStageRetriever(Protocol):
    """Minimal first-stage retrieval contract used by candidate-list sampling."""

    def fit(self, interactions: Iterable[tuple[Any, Any]]) -> FirstStageRetriever:
        """Fit the retriever from user-item interactions."""

    def retrieve(self, user_id: Any, k: int) -> list[Any]:
        """Return up to k candidate item IDs for a user."""


class RecBoleRetriever:
    """RecBole-backed first-stage retriever for research candidate generation."""

    def __init__(self, config: Any):
        validate_recbole_retrieval_config(config)
        self.config = config
        self.model_name = str(cfg_get(config, "retrieval.model")).strip()
        self.seed = int(cfg_get(config, "retrieval.seed", cfg_get(config, "training.seed", 42)))
        self.maximum_k = int(cfg_get(config, "retrieval.k_max", 1000))
        self.filter_seen = bool(cfg_get(config, "retrieval.filter_seen", True))
        self.dataset_name = str(cfg_get(config, "retrieval.recbole_dataset_name", "invarirank_recbole"))
        self.raw_user_by_token: dict[str, Any] = {}
        self.raw_item_by_token: dict[str, Any] = {}
        self.user_token_by_raw: dict[Any, str] = {}
        self.item_token_by_raw: dict[Any, str] = {}
        self.config_object = None
        self.dataset = None
        self.model = None
        self.test_data = None

    def fit(self, interactions: Iterable[tuple[Any, Any]]) -> RecBoleRetriever:
        edges = list(dict.fromkeys(interactions))
        if not edges:
            raise ValueError("RecBoleRetriever requires at least one interaction")
        self._build_token_maps(edges)
        self._write_atomic_interactions(edges)
        self._fit_recbole()
        return self

    def retrieve(self, user_id: Any, k: int) -> list[Any]:
        if k <= 0 or self.dataset is None or self.model is None or self.test_data is None:
            return []
        user_token = self.user_token_by_raw.get(user_id)
        if user_token is None:
            return []
        try:
            internal_user = self.dataset.token2id(self.dataset.uid_field, [user_token])
        except (KeyError, ValueError, TypeError):
            return []

        from recbole.utils.case_study import full_sort_topk

        limit = min(int(k), self.maximum_k, max(int(getattr(self.dataset, "item_num", 1)) - 1, 0))
        if limit <= 0:
            return []
        device = self.config_object["device"] if self.config_object is not None else None
        _, internal_items = full_sort_topk(internal_user, self.model, self.test_data, k=limit, device=device)
        external_items = self.dataset.id2token(self.dataset.iid_field, internal_items.detach().cpu())
        output = []
        for token in _flatten_recbole_tokens(external_items):
            raw_item = self.raw_item_by_token.get(str(token))
            if raw_item is not None:
                output.append(raw_item)
        return output

    def _build_token_maps(self, edges: list[tuple[Any, Any]]) -> None:
        users = sorted({user for user, _ in edges}, key=str)
        items = sorted({item for _, item in edges}, key=str)
        self.user_token_by_raw = {user: _recbole_token(user) for user in users}
        self.item_token_by_raw = {item: _recbole_token(item) for item in items}
        if len(set(self.user_token_by_raw.values())) != len(self.user_token_by_raw):
            raise ValueError("RecBole user tokens are not unique after string conversion.")
        if len(set(self.item_token_by_raw.values())) != len(self.item_token_by_raw):
            raise ValueError("RecBole item tokens are not unique after string conversion.")
        self.raw_user_by_token = {token: user for user, token in self.user_token_by_raw.items()}
        self.raw_item_by_token = {token: item for item, token in self.item_token_by_raw.items()}

    def _write_atomic_interactions(self, edges: list[tuple[Any, Any]]) -> None:
        dataset_dir = self._dataset_dir()
        dataset_dir.mkdir(parents=True, exist_ok=True)
        inter_path = dataset_dir / f"{self.dataset_name}.inter"
        with inter_path.open("w", encoding="utf-8") as handle:
            handle.write("user_id:token\titem_id:token\ttimestamp:float\n")
            for timestamp, (user_id, item_id) in enumerate(edges, start=1):
                handle.write(
                    f"{self.user_token_by_raw[user_id]}\t{self.item_token_by_raw[item_id]}\t{float(timestamp)}\n"
                )

    def _fit_recbole(self) -> None:
        from recbole.config import Config
        from recbole.data import create_dataset, data_preparation
        from recbole.utils import get_model, get_trainer, init_seed

        config = Config(
            model=self.model_name,
            dataset=self.dataset_name,
            config_dict=self._recbole_config(),
        )
        init_seed(config["seed"], config["reproducibility"])
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
        model_dataset = getattr(train_data, "_dataset", getattr(train_data, "dataset", dataset))
        model = get_model(config["model"])(config, model_dataset).to(config["device"])
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        trainer.fit(
            train_data,
            valid_data,
            saved=bool(cfg_get(self.config, "retrieval.recbole_save_model", False)),
            show_progress=bool(cfg_get(self.config, "retrieval.show_progress", True)),
        )
        model.eval()
        self.config_object = config
        self.dataset = dataset
        self.model = model
        self.test_data = test_data

    def _recbole_config(self) -> dict[str, Any]:
        use_cuda = bool(cfg_get(self.config, "retrieval.use_cuda", True))
        config = {
            "data_path": str(self._data_path()),
            "field_separator": "\t",
            "USER_ID_FIELD": "user_id",
            "ITEM_ID_FIELD": "item_id",
            "TIME_FIELD": "timestamp",
            "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
            "epochs": int(cfg_get(self.config, "retrieval.epochs", 100)),
            "train_batch_size": int(cfg_get(self.config, "retrieval.batch_size", 8192)),
            "eval_batch_size": int(cfg_get(self.config, "retrieval.eval_batch_size", 8192)),
            "seed": self.seed,
            "reproducibility": bool(cfg_get(self.config, "retrieval.deterministic", True)),
            "gpu_id": "0" if use_cuda else "",
            "checkpoint_dir": str(self._checkpoint_dir()),
            "eval_args": {
                "split": {"RS": [8, 1, 1]},
                "group_by": "user",
                "order": "RO",
                "mode": "full",
            },
            "metrics": ["Recall"],
            "topk": [10],
            "valid_metric": "Recall@10",
            "show_progress": bool(cfg_get(self.config, "retrieval.show_progress", True)),
        }
        if (learning_rate := cfg_get(self.config, "retrieval.learning_rate")) is not None:
            config["learning_rate"] = float(learning_rate)
        if (embedding_size := cfg_get(self.config, "retrieval.embedding_dim")) is not None:
            config["embedding_size"] = int(embedding_size)
        if (regularization := cfg_get(self.config, "retrieval.reg")) is not None:
            config["reg_weight"] = float(regularization)
        if (layers := cfg_get(self.config, "retrieval.num_layers")) is not None:
            config["n_layers"] = int(layers)
        if (negatives := cfg_get(self.config, "retrieval.negatives_per_positive")) is not None:
            config["train_neg_sample_args"] = {"distribution": "uniform", "sample_num": int(negatives)}
        extra = cfg_get(self.config, "retrieval.recbole_config", {})
        if extra:
            if not isinstance(extra, Mapping):
                raise TypeError("data.retrieval.recbole_config must be a mapping when provided.")
            config.update(dict(extra))
        return config

    def _data_path(self) -> Path:
        return self._recbole_root()

    def _dataset_dir(self) -> Path:
        return self._data_path() / self.dataset_name

    def _checkpoint_dir(self) -> Path:
        return self._recbole_root() / "checkpoints"

    def _recbole_root(self) -> Path:
        configured = cfg_get(self.config, "retrieval.recbole_dir")
        if configured:
            return Path(configured)
        cache_dir = cfg_get(self.config, "paths.cache_dir")
        if cache_dir:
            return Path(cache_dir) / "recbole"
        output_dir = cfg_get(self.config, "paths.output_dir", "data/processed")
        return Path(output_dir) / "recbole"


RECBOLE_BACKEND = "recbole"
RETRIEVER_BACKENDS = frozenset({RECBOLE_BACKEND})


def retrieval_backend(config: Any) -> str:
    """Resolve the configured first-stage retrieval backend."""
    return str(cfg_get(config, "retrieval.backend", RECBOLE_BACKEND)).lower()


def build_first_stage_retriever(config: Any) -> FirstStageRetriever:
    backend = retrieval_backend(config)
    if backend == RECBOLE_BACKEND:
        return RecBoleRetriever(config)
    raise ValueError(f"Unsupported retrieval backend: {backend}. Expected one of {sorted(RETRIEVER_BACKENDS)}")


def validate_recbole_retrieval_config(config: Any) -> None:
    model = cfg_get(config, "retrieval.model")
    if model is None:
        raise ValueError("RecBole retrieval requires data.retrieval.model, for example 'LightGCN'.")
    if not str(model).strip():
        raise ValueError("RecBole retrieval model must be non-empty.")
    if not bool(cfg_get(config, "retrieval.filter_seen", True)):
        raise ValueError("RecBole retrieval currently requires data.retrieval.filter_seen: true.")


def _recbole_token(value: Any) -> str:
    token = str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()
    if not token:
        raise ValueError("RecBole user and item IDs must be non-empty after string conversion.")
    return token


def _flatten_recbole_tokens(values: Any) -> list[str]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (str, bytes)):
        return [values.decode("utf-8") if isinstance(values, bytes) else values]
    if isinstance(values, Iterable):
        output = []
        for value in values:
            output.extend(_flatten_recbole_tokens(value))
        return output
    return [str(values)]


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
    retrieval: Mapping[str, Any] | None = None,
) -> None:
    sample = {
        "user_id": user_id,
        "history": history,
        "candidates": candidates,
        "target_ranking": build_target_ranking(candidates),
        "list_length": list_size,
        "split": split,
    }
    if retrieval is not None:
        sample["retrieval"] = dict(retrieval)
    outputs[split].append(sample)


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
    retriever = build_first_stage_retriever(config).fit(build_train_interactions(splits, minimum_rating))
    provenance = retrieval_provenance(config)
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
                append_sample(outputs, split, user_id, history, candidates, list_size, retrieval=provenance)
    return outputs["train"], outputs["val"], outputs["test"]


def sample_amazon_books(dataset: BaseDataset) -> tuple[list[dict], list[dict], list[dict]]:
    from tqdm.auto import tqdm

    history_length, list_sizes, deterministic, splits = _sampling_settings(dataset)
    config = dataset.config
    seed = int(cfg_get(config, "training.seed", 42))
    minimum_positives = int(cfg_get(config, "sampling.min_future_positives", 1))
    require_retrieved = bool(cfg_get(config, "sampling.amazon.require_retrieved_positive", True))
    retriever = build_first_stage_retriever(config).fit(build_train_interactions(splits, None))
    provenance = retrieval_provenance(config)
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
                append_sample(outputs, split, user_id, history, candidates, list_size, retrieval=provenance)
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
    "FirstStageRetriever",
    "MovieLens32MDataset",
    "RECBOLE_BACKEND",
    "RETRIEVER_BACKENDS",
    "RecBoleRetriever",
    "build_first_stage_retriever",
    "build_dataset_splits",
    "build_target_ranking",
    "candidate_record_issues",
    "cfg_get",
    "graded_relevance",
    "parse_movie_title",
    "retrieval_provenance",
    "save_jsonl",
    "split_user_histories",
    "validate_candidate_records",
    "validate_sample",
    "validate_recbole_retrieval_config",
    "write_dataset_splits",
]

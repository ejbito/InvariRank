from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.data.interactions import build_id_mappings, validate_interactions
from experiments.evaluation.split import split_interactions
from experiments.utils.io import ensure_dir, write_json
from experiments.utils.progress import progress

InteractionReader = Callable[[str | Path], pd.DataFrame]
MetadataReader = Callable[[str | Path], Any]
MetadataNormalizer = Callable[[Any, dict[str, int]], pd.DataFrame]

REVIEW_FILE_CANDIDATES = [
    "reviews.jsonl",
    "reviews.jsonl.gz",
    "reviews.json",
    "reviews.json.gz",
]
METADATA_FILE_CANDIDATES = [
    "metadata.jsonl",
    "metadata.jsonl.gz",
    "meta.jsonl",
    "meta.jsonl.gz",
    "metadata.json",
    "metadata.json.gz",
]
YEAR_PATTERN = re.compile(r"\((\d{4})\)\s*$")


def prepare_dataset(
    *,
    name: str,
    raw_dir: str | Path,
    processed_dir: str | Path,
    read_interactions: InteractionReader,
    read_metadata: MetadataReader,
    normalize_metadata: MetadataNormalizer,
    min_rating: float = 4.0,
    min_user_interactions: int = 3,
    split_strategy: str = "temporal_ratio",
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    show_progress: bool = True,
    write_empty_items: bool = False,
    extra_stats: Callable[[Mapping[str, Any]], dict[str, int]] | None = None,
) -> dict[str, int]:
    processed_path = ensure_dir(processed_dir)
    state: dict[str, Any] = {}
    steps = [
        "read interactions",
        "filter eligible users",
        "map ids",
        "split interactions",
        "read metadata",
        "write files",
    ]

    for step in progress(steps, desc=f"Preparing {name}", total=len(steps), enabled=show_progress):
        if step == "read interactions":
            state["raw_interactions"] = read_interactions(raw_dir)
        elif step == "filter eligible users":
            state["interactions"] = state["raw_interactions"].groupby("user_id").filter(
                lambda group: len(group) >= min_user_interactions
            )
        elif step == "map ids":
            interactions, user_mapping, item_mapping = build_id_mappings(state["interactions"])
            state.update(
                interactions=interactions,
                positives=interactions[interactions["rating"] >= min_rating].copy(),
                user_mapping=user_mapping,
                item_mapping=item_mapping,
            )
        elif step == "split interactions":
            train, val, test = split_interactions(
                state["interactions"],
                strategy=split_strategy,
                min_user_interactions=min_user_interactions,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
            )
            retriever_train, train_queries = make_reranker_training_split(train)
            state.update(
                train=train,
                retriever_train=retriever_train,
                train_queries=train_queries,
                val=val,
                test=test,
            )
        elif step == "read metadata":
            metadata = read_metadata(raw_dir)
            state["items"] = normalize_metadata(metadata, state["item_mapping"])
        elif step == "write files":
            write_processed_dataset(processed_path, state, write_empty_items=write_empty_items)

    stats = {
        "num_interactions": len(state["interactions"]),
        "num_positive_interactions": len(state["positives"]),
        "num_train": len(state["train"]),
        "num_retriever_train": len(state["retriever_train"]),
        "num_train_queries": len(state["train_queries"]),
        "num_val": len(state["val"]),
        "num_test": len(state["test"]),
        "num_users": len(state["user_mapping"]),
        "num_items": len(state["item_mapping"]),
    }
    if extra_stats is not None:
        stats.update(extra_stats(state))
    write_json(stats, processed_path / "stats.json")
    return stats


def write_processed_dataset(
    processed_dir: str | Path,
    state: Mapping[str, Any],
    *,
    write_empty_items: bool = False,
) -> None:
    processed_path = Path(processed_dir)
    state["interactions"].to_csv(processed_path / "interactions.csv", index=False)
    state["train"].to_csv(processed_path / "train.csv", index=False)
    state["retriever_train"].to_csv(processed_path / "retriever_train.csv", index=False)
    state["train_queries"].to_csv(processed_path / "train_queries.csv", index=False)
    state["val"].to_csv(processed_path / "val.csv", index=False)
    state["test"].to_csv(processed_path / "test.csv", index=False)

    items = state["items"]
    if write_empty_items or not items.empty:
        items.to_csv(processed_path / "items.csv", index=False)

    users = pd.DataFrame(
        [{"user_id": mapped, "raw_user_id": raw} for raw, mapped in state["user_mapping"].items()]
    ).sort_values("user_id")
    users.to_csv(processed_path / "users.csv", index=False)

    write_json(state["user_mapping"], processed_path / "user_mapping.json")
    write_json(state["item_mapping"], processed_path / "item_mapping.json")


def make_reranker_training_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve each eligible user's final training interaction as a reranker target."""
    ordered = train.sort_values(["user_id", "timestamp", "item_id"])
    context_parts = []
    query_parts = []
    for _, group in ordered.groupby("user_id", sort=False):
        if len(group) < 2:
            context_parts.append(group)
            continue
        context_parts.append(group.iloc[:-1])
        query_parts.append(group.iloc[-1:])
    empty = ordered.iloc[0:0].copy()
    context = pd.concat(context_parts, ignore_index=True) if context_parts else empty.copy()
    queries = pd.concat(query_parts, ignore_index=True) if query_parts else empty.copy()
    return context, queries


def prepare_amazon(
    raw_dir: str | Path,
    processed_dir: str | Path,
    reviews_file: str | None = None,
    metadata_file: str | None = None,
    min_rating: float = 4.0,
    min_user_interactions: int = 3,
    split_strategy: str = "temporal_ratio",
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    show_progress: bool = True,
) -> dict[str, int]:
    return prepare_dataset(
        name="Amazon",
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        read_interactions=lambda source: read_amazon_reviews(source, reviews_file=reviews_file),
        read_metadata=lambda source: read_amazon_metadata(source, metadata_file=metadata_file),
        normalize_metadata=normalize_amazon_metadata,
        min_rating=min_rating,
        min_user_interactions=min_user_interactions,
        split_strategy=split_strategy,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        show_progress=show_progress,
        write_empty_items=True,
        extra_stats=lambda state: {"num_items_with_metadata": len(state["items"])},
    )


def read_amazon_reviews(
    raw_dir: str | Path,
    reviews_file: str | None = None,
) -> pd.DataFrame:
    path = _resolve_file(raw_dir, reviews_file, REVIEW_FILE_CANDIDATES)
    reviews = pd.read_json(path, lines=True, compression="infer")
    reviews = reviews.rename(columns={"parent_asin": "item_id"})
    required = ["user_id", "item_id", "rating", "timestamp"]
    missing = [column for column in required if column not in reviews.columns]
    if missing:
        raise ValueError(f"Amazon reviews are missing required columns: {missing}")
    reviews = reviews[required].dropna(subset=["user_id", "item_id", "rating", "timestamp"])
    reviews["user_id"] = reviews["user_id"].astype(str)
    reviews["item_id"] = reviews["item_id"].astype(str)
    reviews["rating"] = reviews["rating"].astype(float)
    reviews["timestamp"] = reviews["timestamp"].astype("int64")
    return validate_interactions(reviews)


def read_amazon_metadata(
    raw_dir: str | Path,
    metadata_file: str | None = None,
) -> pd.DataFrame:
    path = _resolve_file(raw_dir, metadata_file, METADATA_FILE_CANDIDATES)
    metadata = pd.read_json(path, lines=True, compression="infer")
    if "parent_asin" not in metadata.columns:
        raise ValueError("Amazon metadata is missing required column: parent_asin")
    return metadata


def normalize_amazon_metadata(
    metadata: pd.DataFrame,
    item_mapping: dict[str, int],
) -> pd.DataFrame:
    records = []
    for record in metadata.to_dict(orient="records"):
        raw_item_id = str(record.get("parent_asin"))
        if raw_item_id not in item_mapping:
            continue
        records.append(
            {
                "item_id": int(item_mapping[raw_item_id]),
                "raw_item_id": raw_item_id,
                "title": _clean_scalar(record.get("title")),
                "main_category": _clean_scalar(record.get("main_category")),
                "categories": _json_list(record.get("categories")),
                "brand_or_store": _clean_scalar(record.get("store")),
                "description": _json_list(record.get("description")),
                "features": _json_list(record.get("features")),
                "price": _clean_number(record.get("price")),
                "average_rating": _clean_number(record.get("average_rating")),
                "rating_number": _clean_number(record.get("rating_number")),
            }
        )
    columns = [
        "item_id",
        "raw_item_id",
        "title",
        "main_category",
        "categories",
        "brand_or_store",
        "description",
        "features",
        "price",
        "average_rating",
        "rating_number",
    ]
    return pd.DataFrame(records, columns=columns).sort_values("item_id").reset_index(drop=True)


def prepare_movielens(
    raw_dir: str | Path,
    processed_dir: str | Path,
    min_rating: float = 4.0,
    min_user_interactions: int = 3,
    split_strategy: str = "temporal_ratio",
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    show_progress: bool = True,
) -> dict[str, int]:
    return prepare_dataset(
        name="MovieLens",
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        read_interactions=read_movielens_ratings,
        read_metadata=read_movielens_movies,
        normalize_metadata=_normalize_movielens_metadata,
        min_rating=min_rating,
        min_user_interactions=min_user_interactions,
        split_strategy=split_strategy,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        show_progress=show_progress,
    )


def read_movielens_ratings(raw_dir: str | Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    ratings_csv = raw_dir / "ratings.csv"
    if ratings_csv.exists():
        ratings = pd.read_csv(ratings_csv)
        ratings = ratings.rename(columns={"userId": "user_id", "movieId": "item_id"})
        return validate_interactions(ratings[["user_id", "item_id", "rating", "timestamp"]])

    ratings_dat = raw_dir / "ratings.dat"
    if ratings_dat.exists():
        ratings = pd.read_csv(
            ratings_dat,
            sep="::",
            engine="python",
            names=["user_id", "item_id", "rating", "timestamp"],
        )
        return validate_interactions(ratings)

    u_data = raw_dir / "u.data"
    if u_data.exists():
        ratings = pd.read_csv(
            u_data,
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
        )
        return validate_interactions(ratings)

    raise FileNotFoundError(
        f"No MovieLens ratings file found in {raw_dir}. Expected ratings.csv, ratings.dat, or u.data."
    )


def read_movielens_movies(raw_dir: str | Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    movies_csv = raw_dir / "movies.csv"
    if movies_csv.exists():
        return pd.read_csv(movies_csv)

    movies_dat = raw_dir / "movies.dat"
    if movies_dat.exists():
        return pd.read_csv(
            movies_dat,
            sep="::",
            engine="python",
            names=["movieId", "title", "genres"],
            encoding="latin-1",
        )

    u_item = raw_dir / "u.item"
    if u_item.exists():
        genre_columns = [
            "unknown",
            "Action",
            "Adventure",
            "Animation",
            "Children",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ]
        columns = [
            "movieId",
            "title",
            "release_date",
            "video_release_date",
            "imdb_url",
            *genre_columns,
        ]
        movies = pd.read_csv(u_item, sep="|", names=columns, encoding="latin-1")
        movies["genres"] = movies[genre_columns].apply(
            lambda row: "|".join(column for column in genre_columns if row[column] == 1),
            axis=1,
        )
        return movies[["movieId", "title", "genres"]]

    return pd.DataFrame(columns=["movieId", "title", "genres"])


def extract_release_year(title: str) -> int | None:
    match = YEAR_PATTERN.search(str(title))
    return int(match.group(1)) if match else None


def clean_title(title: str) -> str:
    return YEAR_PATTERN.sub("", str(title)).strip()


def normalize_movie_metadata(movies: pd.DataFrame, item_mapping: dict[str, int]) -> pd.DataFrame:
    required = {"movieId", "title", "genres"}
    missing = required - set(movies.columns)
    if missing:
        raise ValueError(f"Movie metadata is missing required columns: {sorted(missing)}")

    normalized = movies.copy()
    normalized["raw_item_id"] = normalized["movieId"].astype(str)
    normalized = normalized[normalized["raw_item_id"].isin(item_mapping)].copy()
    normalized["item_id"] = normalized["raw_item_id"].map(item_mapping).astype(int)
    normalized["release_year"] = normalized["title"].map(extract_release_year)
    normalized["title"] = normalized["title"].map(clean_title)
    normalized["genres"] = normalized["genres"].replace("(no genres listed)", "")
    return normalized[["item_id", "raw_item_id", "title", "genres", "release_year"]].sort_values("item_id")


def _normalize_movielens_metadata(movies: pd.DataFrame, item_mapping: dict[str, int]) -> pd.DataFrame:
    if movies.empty:
        return pd.DataFrame()
    return normalize_movie_metadata(movies, item_mapping)


def _resolve_file(raw_dir: str | Path, explicit_file: str | None, candidates: list[str]) -> Path:
    raw_dir = Path(raw_dir)
    if explicit_file:
        path = raw_dir / explicit_file
        if not path.exists():
            raise FileNotFoundError(f"Configured file does not exist: {path}")
        return path
    for name in candidates:
        path = raw_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No Amazon file found in {raw_dir}. Tried: {candidates}")


def _json_list(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "[]"
    if isinstance(value, list):
        return json.dumps([item for item in value if item is not None])
    return json.dumps([value])


def _clean_scalar(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)


def _clean_number(value: Any) -> float | int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        value = value.replace("$", "").replace(",", "").strip()
        if not value:
            return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return int(numeric) if numeric.is_integer() else numeric


__all__ = [
    "clean_title",
    "extract_release_year",
    "normalize_amazon_metadata",
    "normalize_movie_metadata",
    "prepare_amazon",
    "prepare_dataset",
    "prepare_movielens",
    "read_amazon_metadata",
    "read_amazon_reviews",
    "read_movielens_movies",
    "read_movielens_ratings",
    "write_processed_dataset",
]

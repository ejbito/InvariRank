from __future__ import annotations

import re

import pandas as pd
from tqdm.auto import tqdm

from .base import BaseDataset
from .utils import cfg_get, graded_relevance


class MovieLens32MDataset(BaseDataset):
    @classmethod
    def code(cls) -> str:
        return "movielens32m"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.df_ratings = None
        self.df_movies = None

    def load_raw(self) -> None:
        ratings_path = cfg_get(self.cfg, "paths.ratings")
        movies_path = cfg_get(self.cfg, "paths.movies")
        print("[Dataset] Loading MovieLens ratings")
        self.df_ratings = pd.read_csv(ratings_path)
        print(f"[Dataset] Ratings rows: {len(self.df_ratings)}")
        print("[Dataset] Loading MovieLens movies")
        self.df_movies = pd.read_csv(movies_path)
        print(f"[Dataset] Movies rows: {len(self.df_movies)}")

    def build_item_metadata(self) -> None:
        popularity = self.df_ratings["movieId"].value_counts().to_dict()
        for row in tqdm(self.df_movies.itertuples(index=False), desc="[Dataset] Movies"):
            title, year = parse_movie_title(row.title)
            self.item_metadata[int(row.movieId)] = {
                "title": title,
                "genres": row.genres.split("|") if isinstance(row.genres, str) else [],
                "year": year,
                "popularity": int(popularity.get(row.movieId, 0)),
            }
        print(f"[Dataset] Item metadata: {len(self.item_metadata)} items")

    def build_user_histories(self) -> None:
        min_interactions = int(cfg_get(self.cfg, "dataset.min_user_interactions", 50))
        max_users = cfg_get(self.cfg, "training.max_users", 5000)
        max_users = int(max_users) if max_users is not None else None
        max_per_user = cfg_get(self.cfg, "dataset.max_interactions_per_user", None)
        max_per_user = int(max_per_user) if max_per_user is not None else None

        counts = self.df_ratings["userId"].value_counts()
        users = sorted(counts[counts >= min_interactions].index.tolist())
        if max_users is not None:
            users = users[:max_users]

        df = self.df_ratings[self.df_ratings["userId"].isin(users)].sort_values(["userId", "timestamp"])
        print(f"[Dataset] Candidate users: {len(users)}")

        for uid, group in tqdm(df.groupby("userId", sort=True), desc="[Dataset] Users"):
            rows = list(group.itertuples(index=False))
            if max_per_user is not None and len(rows) > max_per_user:
                rows = rows[-max_per_user:]

            history = []
            for row in rows:
                movie_id = int(row.movieId)
                if movie_id not in self.item_metadata:
                    continue
                meta = self.item_metadata[movie_id]
                rating = float(row.rating)
                history.append(
                    {
                        "item_id": movie_id,
                        "relevance": int(graded_relevance(rating)),
                        "title": meta["title"],
                        "genres": list(meta["genres"]),
                        "year": meta["year"],
                        "popularity": meta["popularity"],
                        "rating": rating,
                        "timestamp": int(row.timestamp),
                    }
                )

            if len(history) >= min_interactions:
                self.user_histories[int(uid)] = history

        print(f"[Dataset] User histories: {len(self.user_histories)} users")


def parse_movie_title(raw_title: str) -> tuple[str, int | None]:
    match = re.search(r"\((\d{4})\)\s*$", str(raw_title))
    year = int(match.group(1)) if match else None
    title = re.sub(r"\s*\(\d{4}\)\s*$", "", str(raw_title)).strip()
    return title, year

import random

import pandas as pd
from tqdm.auto import tqdm

from .config_utils import cfg_get
from .samplers import sample_movielens
from .utils import graded_relevance


class MovieLensDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.seed = int(cfg_get(self.cfg, "training.seed", 42))
        self.rng = random.Random(self.seed)

        self.df_ratings = None
        self.df_movies = None
        self.item_metadata: dict[int, dict] = {}
        self.user_histories: dict[int, list[dict]] = {}

    def load_raw(self):
        ratings_path = cfg_get(self.cfg, "paths.ratings")
        movies_path = cfg_get(self.cfg, "paths.movies")
        print("[Dataset] Loading MovieLens ratings")
        self.df_ratings = pd.read_csv(ratings_path)
        print(f"[Dataset] Ratings rows: {len(self.df_ratings)}")
        print("[Dataset] Loading MovieLens movies")
        self.df_movies = pd.read_csv(movies_path)
        print(f"[Dataset] Movies rows: {len(self.df_movies)}")

    def build_item_metadata(self):
        pop = self.df_ratings["movieId"].value_counts().to_dict()
        for r in tqdm(self.df_movies.itertuples(index=False), desc="[Dataset] Movies"):
            self.item_metadata[int(r.movieId)] = {
                "title": r.title,
                "genres": r.genres.split("|") if isinstance(r.genres, str) else [],
                "year": None,
                "popularity": int(pop.get(r.movieId, 0)),
            }
        print(f"[Dataset] Item metadata: {len(self.item_metadata)} items")

    def build_user_histories(self):
        min_i = int(cfg_get(self.cfg, "dataset.min_user_interactions", 50))
        max_u = int(cfg_get(self.cfg, "training.max_users", 5000))
        max_per_user = cfg_get(self.cfg, "dataset.max_interactions_per_user", None)
        max_per_user = int(max_per_user) if max_per_user is not None else None

        counts = self.df_ratings["userId"].value_counts()
        users = sorted(counts[counts >= min_i].index.tolist())
        if max_u is not None:
            users = users[:max_u]

        df = self.df_ratings[self.df_ratings["userId"].isin(users)]
        df = df.sort_values(["userId", "timestamp"])

        print(f"[Dataset] Candidate users: {len(users)}")
        for uid, g in tqdm(df.groupby("userId", sort=True), desc="[Dataset] Users"):
            hist = []
            rows = list(g.itertuples(index=False))
            if max_per_user is not None and len(rows) > max_per_user:
                rows = rows[-max_per_user:]

            for r in rows:
                mid = int(r.movieId)
                if mid not in self.item_metadata:
                    continue
                meta = self.item_metadata[mid]
                rating = float(r.rating)
                hist.append(
                    {
                        "item_id": mid,
                        "relevance": int(graded_relevance(rating)),
                        "title": meta["title"],
                        "genres": list(meta["genres"]),
                        "year": meta["year"],
                        "popularity": meta["popularity"],
                        "rating": rating,
                        "timestamp": int(r.timestamp),
                    }
                )

            if len(hist) >= min_i:
                self.user_histories[int(uid)] = hist
        print(f"[Dataset] User histories: {len(self.user_histories)} users")

    def generate_samples(self):
        return sample_movielens(self)

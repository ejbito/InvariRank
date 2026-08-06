from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from time import perf_counter

import pandas as pd

from experiments.data.interactions import validate_interactions
from experiments.utils.io import read_pickle, write_pickle


class BaseRetriever(ABC):
    def __init__(self) -> None:
        self.seen_items_: dict[int, set[int]] = {}
        self.n_users_: int = 0
        self.n_items_: int = 0
        self.training_stats_: dict = {}

    def _prepare_fit(self, interactions: pd.DataFrame) -> pd.DataFrame:
        interactions = validate_interactions(interactions)
        if interactions.empty:
            raise ValueError("Cannot fit a retriever on empty interactions.")
        self.n_users_ = int(interactions["user_id"].max()) + 1
        self.n_items_ = int(interactions["item_id"].max()) + 1
        self.seen_items_ = (
            interactions.groupby("user_id")["item_id"].apply(lambda values: set(map(int, values)))
        ).to_dict()
        return interactions

    def _start_training_stats(self, interactions: pd.DataFrame) -> float:
        self.training_stats_ = {
            "retriever": type(self).__name__,
            "num_interactions": int(len(interactions)),
            "num_users": int(self.n_users_),
            "num_items": int(self.n_items_),
            "loss_history": [],
        }
        return perf_counter()

    def _finish_training_stats(self, started_at: float) -> None:
        self.training_stats_["fit_seconds"] = perf_counter() - started_at

    def _filter_seen(self, user_id: int, ranked_items: list[int], exclude_seen: bool) -> list[int]:
        if not exclude_seen:
            return ranked_items
        seen = self.seen_items_.get(int(user_id), set())
        return [item_id for item_id in ranked_items if item_id not in seen]

    @abstractmethod
    def fit(self, interactions: pd.DataFrame) -> "BaseRetriever":
        raise NotImplementedError

    @abstractmethod
    def recommend(
        self,
        user_ids: list[int],
        k: int = 100,
        exclude_seen: bool = True,
    ) -> dict[int, list[int]]:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        write_pickle(self, Path(path) / "model.pkl")

    @classmethod
    def load(cls, path: str | Path) -> "BaseRetriever":
        model = read_pickle(Path(path) / "model.pkl")
        if not isinstance(model, cls):
            raise TypeError(f"Expected saved {cls.__name__}, got {type(model).__name__}")
        return model

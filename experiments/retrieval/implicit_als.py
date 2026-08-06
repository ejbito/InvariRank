from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from experiments.data.interactions import user_item_matrix
from experiments.retrieval.base import BaseRetriever


class ImplicitALSRetriever(BaseRetriever):
    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 20,
        alpha: float = 40.0,
        show_progress: bool = True,
    ) -> None:
        super().__init__()
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.show_progress = show_progress
        self.user_factors_: np.ndarray | None = None
        self.item_factors_: np.ndarray | None = None
        self.user_items_ = None

    def fit(self, interactions: pd.DataFrame) -> "ImplicitALSRetriever":
        interactions = self._prepare_fit(interactions)
        started_at = self._start_training_stats(interactions)
        try:
            from implicit.als import AlternatingLeastSquares
        except ImportError as exc:
            raise ImportError(
                "ImplicitALSRetriever requires the optional 'implicit' package. "
                "Install it with: pip install -e \".[experiments]\""
            ) from exc

        self.user_items_ = user_item_matrix(interactions, self.n_users_, self.n_items_)
        confidence = (self.user_items_ * self.alpha).astype("float32")
        model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
        )
        try:
            model.fit(confidence, show_progress=False)
        except TypeError:
            model.fit(confidence)
        self.user_factors_ = model.user_factors.astype("float32")
        self.item_factors_ = model.item_factors.astype("float32")
        self.training_stats_["loss_history_note"] = (
            "Implicit ALS loss history is not captured; the external implicit package "
            "does not expose a stable per-iteration loss API through this wrapper."
        )
        self._finish_training_stats(started_at)
        return self

    def recommend(
        self,
        user_ids: list[int],
        k: int = 100,
        exclude_seen: bool = True,
    ) -> dict[int, list[int]]:
        if self.user_factors_ is None or self.item_factors_ is None:
            raise RuntimeError("ImplicitALSRetriever must be fitted before calling recommend.")

        recommendations = {}
        item_ids = np.arange(self.n_items_)
        show_progress = getattr(self, "show_progress", True)
        for user_id in tqdm(user_ids, desc="ALS recommendations", disable=not show_progress):
            if not 0 <= int(user_id) < len(self.user_factors_):
                recommendations[int(user_id)] = []
                continue
            scores = self.item_factors_.dot(self.user_factors_[int(user_id)])
            ranked = np.lexsort((item_ids, -scores)).astype(int).tolist()
            ranked = self._filter_seen(int(user_id), ranked, exclude_seen)
            recommendations[int(user_id)] = ranked[:k]
        return recommendations

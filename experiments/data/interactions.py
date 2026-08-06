from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import sparse

REQUIRED_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]


def validate_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in interactions.columns]
    if missing:
        raise ValueError(f"Interactions are missing required columns: {missing}")
    return interactions.copy()


def load_interactions(path: str | Path) -> pd.DataFrame:
    return validate_interactions(pd.read_csv(path))


def build_id_mappings(
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    interactions = validate_interactions(interactions)
    users = sorted(interactions["user_id"].unique().tolist())
    items = sorted(interactions["item_id"].unique().tolist())
    user_mapping = {str(raw_id): idx for idx, raw_id in enumerate(users)}
    item_mapping = {str(raw_id): idx for idx, raw_id in enumerate(items)}

    mapped = interactions.copy()
    mapped["raw_user_id"] = mapped["user_id"]
    mapped["raw_item_id"] = mapped["item_id"]
    mapped["user_id"] = mapped["raw_user_id"].astype(str).map(user_mapping).astype(int)
    mapped["item_id"] = mapped["raw_item_id"].astype(str).map(item_mapping).astype(int)
    return mapped, user_mapping, item_mapping


def user_item_matrix(
    interactions: pd.DataFrame,
    n_users: int | None = None,
    n_items: int | None = None,
    value_column: str | None = None,
) -> sparse.csr_matrix:
    interactions = validate_interactions(interactions)
    if interactions.empty:
        rows = n_users or 0
        cols = n_items or 0
        return sparse.csr_matrix((rows, cols), dtype="float32")

    n_users = n_users or int(interactions["user_id"].max()) + 1
    n_items = n_items or int(interactions["item_id"].max()) + 1
    data = (
        interactions[value_column].astype("float32").to_numpy()
        if value_column
        else [1.0] * len(interactions)
    )
    return sparse.csr_matrix(
        (data, (interactions["user_id"].to_numpy(), interactions["item_id"].to_numpy())),
        shape=(n_users, n_items),
        dtype="float32",
    )

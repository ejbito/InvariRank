from __future__ import annotations

import pandas as pd

from experiments.data.interactions import validate_interactions


def leave_one_out_split(
    interactions: pd.DataFrame,
    min_user_interactions: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interactions = validate_interactions(interactions)
    filtered = interactions.groupby("user_id").filter(
        lambda group: len(group) >= min_user_interactions
    )
    ordered = filtered.sort_values(["user_id", "timestamp", "item_id"]).copy()
    ranks = ordered.groupby("user_id").cumcount(ascending=False)

    test = ordered[ranks == 0].reset_index(drop=True)
    val = ordered[ranks == 1].reset_index(drop=True)
    train = ordered[ranks >= 2].reset_index(drop=True)
    return train, val, test


def temporal_ratio_split(
    interactions: pd.DataFrame,
    min_user_interactions: int = 3,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronologically split each user while keeping every segment non-empty."""
    interactions = validate_interactions(interactions)
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than one.")
    required = max(3, int(min_user_interactions))
    filtered = interactions.groupby("user_id").filter(lambda group: len(group) >= required)
    ordered = filtered.sort_values(["user_id", "timestamp", "item_id"]).copy()

    train_parts = []
    val_parts = []
    test_parts = []
    for _, group in ordered.groupby("user_id", sort=False):
        count = len(group)
        train_end = max(1, int(count * train_ratio))
        val_count = max(1, int(count * val_ratio))
        if train_end + val_count >= count:
            train_end = max(1, count - val_count - 1)
        val_end = train_end + val_count
        train_parts.append(group.iloc[:train_end])
        val_parts.append(group.iloc[train_end:val_end])
        test_parts.append(group.iloc[val_end:])

    empty = ordered.iloc[0:0].copy()
    concatenate = lambda parts: pd.concat(parts, ignore_index=True) if parts else empty.copy()
    return concatenate(train_parts), concatenate(val_parts), concatenate(test_parts)


def split_interactions(
    interactions: pd.DataFrame,
    *,
    strategy: str,
    min_user_interactions: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if strategy == "leave_one_out":
        return leave_one_out_split(interactions, min_user_interactions=min_user_interactions)
    if strategy in {"temporal_ratio", "temporal_70_10_20"}:
        return temporal_ratio_split(
            interactions,
            min_user_interactions=min_user_interactions,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
    raise ValueError(f"Unsupported split strategy: {strategy}")


__all__ = ["leave_one_out_split", "split_interactions", "temporal_ratio_split"]

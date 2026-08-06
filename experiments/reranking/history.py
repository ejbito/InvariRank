from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_user_histories(
    processed_dir: str | Path,
    max_history_items: int = 20,
    split: str = "test",
) -> dict[int, list[dict[str, Any]]]:
    processed_dir = Path(processed_dir)
    if split == "train":
        context_path = processed_dir / "retriever_train.csv"
        context = pd.read_csv(context_path if context_path.exists() else processed_dir / "train.csv")
    elif split == "val":
        context = pd.read_csv(processed_dir / "train.csv")
    elif split == "test":
        parts = [pd.read_csv(processed_dir / "train.csv")]
        val_path = processed_dir / "val.csv"
        if val_path.exists():
            parts.append(pd.read_csv(val_path))
        context = pd.concat(parts, ignore_index=True)
    else:
        raise ValueError("split must be one of: train, val, test")
    items = pd.read_csv(processed_dir / "items.csv") if (processed_dir / "items.csv").exists() else pd.DataFrame()
    metadata = {}
    if not items.empty:
        for record in items.to_dict(orient="records"):
            metadata[int(record["item_id"])] = _format_history_item(record)

    histories = {}
    ordered = context.sort_values(["user_id", "timestamp", "item_id"], ascending=[True, False, True])
    for user_id, group in ordered.groupby("user_id"):
        records = []
        for interaction in group.head(max_history_items).to_dict(orient="records"):
            item_id = int(interaction["item_id"])
            record = dict(metadata.get(item_id, {"item_id": str(item_id)}))
            record["rating"] = float(interaction["rating"])
            records.append(record)
        histories[int(user_id)] = records
    return histories


def _format_history_item(record: dict[str, Any]) -> dict[str, Any]:
    item_id = record.get("raw_item_id", record.get("item_id"))
    formatted = {"item_id": str(item_id)}
    for key, value in record.items():
        if key in {"item_id", "raw_item_id"}:
            continue
        if pd.isna(value) if not isinstance(value, list) else False:
            continue
        if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
            try:
                import json

                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        elif key == "genres" and isinstance(value, str):
            value = [genre for genre in value.split("|") if genre]
        elif key == "release_year":
            value = int(value)
        formatted[key] = value
    return formatted

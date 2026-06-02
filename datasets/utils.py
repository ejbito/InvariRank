from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str | Path, *, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cfg_get(cfg: Any, path: str, default=None):
    cur = cfg
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key)
        elif hasattr(cur, key):
            cur = getattr(cur, key)
        else:
            try:
                cur = cur[key]
            except Exception:
                return default
    return default if cur is None else cur


def graded_relevance(rating: float | None) -> int:
    if rating is None:
        return 0
    rating = float(rating)
    if rating >= 4.0:
        return 4
    if rating >= 3.0:
        return 3
    if rating >= 2.0:
        return 2
    if rating >= 1.0:
        return 1
    return 0


def make_candidate(item_id, relevance: int, meta: dict) -> dict:
    return {
        "item_id": item_id,
        "relevance": int(relevance),
        "title": meta.get("title", ""),
        "genres": list(meta.get("genres", [])),
        "year": meta.get("year"),
        "popularity": int(meta.get("popularity", 0)),
    }


def build_target_ranking(candidates: list[dict], rng: random.Random | None = None) -> dict:
    rel_groups: dict[int, list[dict]] = {}
    for candidate in candidates:
        rel_groups.setdefault(int(candidate["relevance"]), []).append(candidate)

    ranking = []

    def alpha_key(candidate: dict):
        title = (candidate.get("title") or "").strip().lower()
        year = str(candidate.get("year") or "").strip()
        item_id = str(candidate.get("item_id") or "")
        return (title, year, item_id)

    for rel in sorted(rel_groups.keys(), reverse=True):
        group = rel_groups[rel]
        if rng is not None:
            rng.shuffle(group)
        else:
            group = sorted(group, key=alpha_key)
        ranking.extend(group)

    return {
        "item_ids": [candidate["item_id"] for candidate in ranking],
        "relevance": [candidate["relevance"] for candidate in ranking],
    }


def save_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_lengths(lengths: Iterable[int]) -> dict[str, float]:
    values = list(lengths)
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {"min": float(min(values)), "max": float(max(values)), "mean": float(sum(values) / len(values))}


def stable_hash_int(text: str) -> int:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def json_loads():
    try:
        import ujson  # type: ignore

        return ujson.loads
    except Exception:
        return json.loads

import random
from collections.abc import Iterable


def graded_relevance(rating: float) -> int:
    """
    Map raw ratings to graded relevance labels.

    MovieLens / Amazon convention:
      4-5 -> strong positive
      3   -> weak positive
      unseen -> handled elsewhere
      <3  -> negative
    """
    if rating >= 4.0:
        return 4
    elif rating >= 3.0:
        return 3
    elif rating >= 2.0:
        return 2
    elif rating >= 1.0:
        return 1
    else:
        return 0


def build_target_ranking(candidates: list[dict], rng: random.Random | None = None):
    """
    Build a target ranking for listwise supervision.

    Items are sorted by relevance (descending).
    If rng is provided, ties are randomly shuffled with that RNG.
    If rng is None, ties are deterministically sorted.
    """
    rel_groups = {}
    for c in candidates:
        rel_groups.setdefault(c["relevance"], []).append(c)

    ranking = []

    def _alpha_key(c):
        title = (c.get("title") or "").strip().lower()
        year = str(c.get("year") or "").strip()
        item_id = str(c.get("item_id") or "")
        return (title, year, item_id)

    for rel in sorted(rel_groups.keys(), reverse=True):
        group = rel_groups[rel]
        if rng is not None:
            rng.shuffle(group)
        else:
            group = sorted(group, key=_alpha_key)
        ranking.extend(group)

    return {
        "item_ids": [c["item_id"] for c in ranking],
        "relevance": [c["relevance"] for c in ranking],
    }


def summarize_lengths(lengths: Iterable[int]) -> dict[str, float]:
    values = list(lengths)
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def make_candidate(item_id, relevance: int, meta: dict) -> dict:
    return {
        "item_id": item_id,
        "relevance": int(relevance),
        "title": meta.get("title", ""),
        "genres": list(meta.get("genres", [])),
        "year": meta.get("year"),
        "popularity": int(meta.get("popularity", 0)),
    }

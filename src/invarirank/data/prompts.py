from __future__ import annotations

from typing import Any


def format_user_history(history: list[dict[str, Any]] | None) -> str:
    lines: list[str] = []
    for item in history or []:
        title = item.get("title", "")
        year = item.get("year", "")
        rating = item.get("rating", "")
        genres = item.get("genres", [])

        line = f"title: {title}"
        if year:
            line += f" ({year})"
        if rating:
            line += f", rating: {rating}"
        if genres:
            line += f", genres: {', '.join(map(str, genres))}"
        lines.append(line)
    return "\n".join(lines)


def format_candidate_item(item: dict[str, Any]) -> str:
    title = item.get("title", item.get("name", ""))
    year = item.get("year", "")
    genres = item.get("genres", item.get("categories", []))

    text = str(title)
    if year:
        text += f" ({year})"
    if genres:
        text += f", genres: {', '.join(map(str, genres))}"
    return text


def build_prompt(sample: dict[str, Any], permutation: list[int], cfg: Any) -> str:
    history_text = format_user_history(sample.get("history"))
    parts = [
        cfg.span_start_token,
        "Given the user's interaction history, rank the candidate items according to the user's preferences.",
        "",
        history_text,
        cfg.span_end_token,
        "",
    ]

    candidates = sample["candidates"]
    for idx in permutation:
        parts.append(cfg.item_start_token)
        parts.append(format_candidate_item(candidates[idx]))
        parts.append(cfg.item_end_token)
        parts.append("")

    return "\n".join(parts)


def extract_relevance_labels(sample: dict[str, Any], permutation: list[int]) -> list[int]:
    return [int(sample["candidates"][idx].get("relevance", 0)) for idx in permutation]


def candidate_id(item: dict[str, Any], fallback: int) -> str:
    for key in ("item_id", "id", "asin", "movie_id"):
        if key in item:
            return str(item[key])
    return str(fallback)

from __future__ import annotations

from typing import Any

from .base import candidate_id, format_candidate_item, format_user_history, load_template


def build_prompt(sample: dict[str, Any], permutation: list[int], cfg: Any) -> str:
    template = load_template(getattr(cfg, "prompt_template", None), "rankgpt")
    instruction = template.get("instruction", "Rank the candidate items for the user.")
    history_header = template.get("history_header", "User history:")
    candidates_header = template.get("candidates_header", "Candidate items:")
    response_instruction = template.get("response_instruction", "Return the item IDs in ranked order.")

    parts = [instruction, ""]
    history_text = format_user_history(sample.get("history"), template)
    if history_header:
        parts.append(history_header)
    if history_text:
        parts.append(history_text)
    parts.append("")
    if candidates_header:
        parts.append(candidates_header)

    candidates = sample["candidates"]
    for rank, idx in enumerate(permutation, start=1):
        item = candidates[idx]
        parts.append(f"[{rank}] {candidate_id(item, idx)}: {format_candidate_item(item, template)}")

    parts.extend(["", response_instruction])
    return "\n".join(parts)

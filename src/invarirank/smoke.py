from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import require_fields
from .data import build_prompt, extract_relevance_labels, sample_permutation
from .modeling import (
    SpanExtractor,
    build_attention_mask,
    build_position_ids,
    load_model_for_ranking,
    load_tokenizer,
    resolve_dtype,
    select_device,
)
from .ranking import run_ranking_pipeline
from .utils import load_jsonl


def smoke_validate_preflight(cfg: Any) -> dict[str, Any]:
    require_fields(cfg, ["model_name", "data_path", "output_dir"])
    tokenizer = load_tokenizer(cfg)
    samples = load_jsonl(cfg.data_path)
    if not samples:
        raise ValueError(f"No samples found in {cfg.data_path}")

    sample = samples[0]
    perm = sample_permutation(len(sample["candidates"]), deterministic=True, seed=int(cfg.seed))
    prompt = build_prompt(sample, perm, cfg)
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=int(cfg.max_seq_length),
    )

    span_info = SpanExtractor(tokenizer, cfg)(enc["input_ids"])
    if len(span_info.candidate_spans) != len(sample["candidates"]):
        raise ValueError(
            f"Expected {len(sample['candidates'])} candidate spans; found {len(span_info.candidate_spans)}."
        )

    dtype = resolve_dtype(getattr(cfg, "dtype", "float32"))
    attn = build_attention_mask(enc["attention_mask"], span_info, cfg, dtype)
    pos = build_position_ids(enc["input_ids"], span_info, cfg)
    relevance = extract_relevance_labels(sample, perm)

    return {
        "prompt_chars": len(prompt),
        "tokens": int(enc["input_ids"].shape[-1]),
        "candidate_spans": span_info.candidate_spans,
        "attention_mask_shape": tuple(attn.shape),
        "position_ids_shape": None if pos is None else tuple(pos.shape),
        "permutation": perm,
        "relevance": relevance,
    }


def smoke_run_model(cfg: Any) -> list[dict[str, Any]]:
    records = run_ranking_pipeline(cfg)
    if not records:
        raise ValueError("Smoke ranking produced no records.")
    first = records[0]
    permutations = first.get("permutations", []) or []
    if not permutations:
        raise ValueError("Smoke ranking produced no permutation records.")
    ranking = permutations[0].get("output_ranking", {}).get("candidate_indices", []) or []
    if not ranking:
        raise ValueError("Smoke ranking produced an empty output ranking.")
    return records


def smoke_load_model_only(cfg: Any) -> None:
    tokenizer = load_tokenizer(cfg)
    device = select_device(getattr(cfg, "device", "cpu"))
    model = load_model_for_ranking(cfg, tokenizer, device)
    del model


def smoke_output_path(cfg: Any) -> Path:
    return Path(getattr(cfg, "ranked_lists_path", Path(cfg.output_dir) / "ranked_lists.json"))

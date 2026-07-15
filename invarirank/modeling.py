from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


def resolve_dtype(dtype_name: str):
    import torch

    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return aliases[dtype_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype_name}") from exc


def select_device(requested: str):
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def load_tokenizer(cfg: Any):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        token=os.environ.get("HF_TOKEN"),
        trust_remote_code=bool(getattr(cfg, "trust_remote_code", False)),
    )
    special_tokens = [
        cfg.span_start_token,
        cfg.span_end_token,
        cfg.item_start_token,
        cfg.item_end_token,
    ]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    validate_special_tokens(tokenizer, cfg)
    return tokenizer


def validate_special_tokens(tokenizer: Any, cfg: Any) -> None:
    tokens = [
        cfg.span_start_token,
        cfg.span_end_token,
        cfg.item_start_token,
        cfg.item_end_token,
    ]
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    missing = []
    split = []
    for token in tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0 or token_id == unknown_id:
            missing.append(token)
        encoded = tokenizer(token, add_special_tokens=False)["input_ids"]
        if len(encoded) != 1:
            split.append(token)
    if missing:
        raise ValueError(f"Special token(s) missing from tokenizer vocabulary: {missing}")
    if split:
        raise ValueError(f"Special token(s) do not tokenize as single tokens: {split}")


def load_base_model(cfg: Any, tokenizer: Any, device: Any):
    from transformers import AutoModelForCausalLM

    dtype = resolve_dtype(getattr(cfg, "dtype", "bfloat16"))
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        dtype=dtype,
        token=os.environ.get("HF_TOKEN"),
        trust_remote_code=bool(getattr(cfg, "trust_remote_code", False)),
    )
    model.resize_token_embeddings(len(tokenizer))
    return model.to(device)


def build_lora_model(cfg: Any, tokenizer: Any, device: Any):
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    model = load_base_model(cfg, tokenizer, device)
    resume = getattr(cfg, "resume_checkpoint_path", None)
    if resume:
        return PeftModel.from_pretrained(model, resume, is_trainable=True)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(cfg.lora_r),
        lora_alpha=int(cfg.lora_alpha),
        lora_dropout=float(cfg.lora_dropout),
        target_modules=list(cfg.lora_target_modules),
    )
    return get_peft_model(model, lora_config)


def load_model_for_ranking(cfg: Any, tokenizer: Any, device: Any):
    model = load_base_model(cfg, tokenizer, device)
    adapter_path = getattr(cfg, "adapter_path", None) or getattr(cfg, "checkpoint_path", None)
    if adapter_path:
        adapter_path = Path(adapter_path)
        if adapter_path.exists() and adapter_path.is_dir() and not (adapter_path / "adapter_config.json").exists():
            raise ValueError(
                f"Configured adapter_path exists but is not a PEFT adapter directory: {adapter_path}. "
                "Remove or unset adapter_path to use the original base model for zero-shot ranking."
            )
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model


def model_dtype(model: Any):
    return next(model.parameters()).dtype


@dataclass(frozen=True)
class SpanInfo:
    span_start: int
    span_end: int
    candidate_spans: list[tuple[int, int]]


class SpanExtractor:
    def __init__(self, tokenizer: Any, cfg: Any):
        self.span_start_id = tokenizer.convert_tokens_to_ids(cfg.span_start_token)
        self.span_end_id = tokenizer.convert_tokens_to_ids(cfg.span_end_token)
        self.item_start_id = tokenizer.convert_tokens_to_ids(cfg.item_start_token)
        self.item_end_id = tokenizer.convert_tokens_to_ids(cfg.item_end_token)
        self.unknown_id = getattr(tokenizer, "unk_token_id", None)

    def __call__(self, input_ids: Any) -> SpanInfo:
        ids = input_ids[0].tolist() if getattr(input_ids, "ndim", 1) == 2 else input_ids.tolist()
        required = [self.span_start_id, self.span_end_id, self.item_start_id, self.item_end_id]
        if any(token_id is None or token_id < 0 or token_id == self.unknown_id for token_id in required):
            raise ValueError("Special token IDs missing from tokenizer. Add special tokens before tokenizing.")

        try:
            span_start = ids.index(self.span_start_id)
            span_end = ids.index(self.span_end_id) + 1
        except ValueError as exc:
            raise ValueError("Shared [SPAN] markers were not found in input_ids.") from exc

        candidate_spans: list[tuple[int, int]] = []
        cursor = span_end
        while True:
            try:
                item_start = ids.index(self.item_start_id, cursor)
                item_end = ids.index(self.item_end_id, item_start + 1)
            except ValueError:
                break
            candidate_spans.append((item_start, item_end + 1))
            cursor = item_end + 1

        if not candidate_spans:
            raise ValueError("No [ITEM] candidate spans found.")
        return SpanInfo(span_start=span_start, span_end=span_end, candidate_spans=candidate_spans)


class AttentionMaskMode(str, Enum):
    CAUSAL = "causal"
    BLOCK = "block"


class PositionIdMode(str, Enum):
    STANDARD = "standard"
    SHARED = "shared"


def parse_attention_mask_mode(value: str | AttentionMaskMode) -> AttentionMaskMode:
    try:
        return AttentionMaskMode(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported attention mask mode: {value}") from exc


def parse_position_id_mode(value: str | PositionIdMode) -> PositionIdMode:
    try:
        return PositionIdMode(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported position ID mode: {value}") from exc


def make_4d_causal_mask_from_2d(attention_2d: Any, dtype: Any) -> Any:
    import torch

    batch, sequence_length = attention_2d.shape
    device = attention_2d.device
    lower_triangle = torch.tril(torch.ones((sequence_length, sequence_length), device=device, dtype=torch.bool))
    key_allowed = attention_2d.bool().unsqueeze(1).unsqueeze(1)
    allowed = lower_triangle.view(1, 1, sequence_length, sequence_length) & key_allowed
    blocked = torch.finfo(dtype).min
    mask = torch.zeros((batch, 1, sequence_length, sequence_length), device=device, dtype=dtype)
    return mask.masked_fill(~allowed, blocked)


def make_span_item_block_mask(
    attention_2d: Any,
    span_info: SpanInfo,
    dtype: Any,
    *,
    span_causal: bool = True,
) -> Any:
    import torch

    batch, sequence_length = attention_2d.shape
    if batch != 1:
        raise ValueError("Block mask construction currently expects batch_size=1.")

    device = attention_2d.device
    allowed = torch.zeros((sequence_length, sequence_length), dtype=torch.bool, device=device)
    span_start, span_end = span_info.span_start, span_info.span_end
    span_length = span_end - span_start
    if span_causal:
        allowed[span_start:span_end, span_start:span_end] = torch.tril(
            torch.ones((span_length, span_length), device=device, dtype=torch.bool)
        )
    else:
        allowed[span_start:span_end, span_start:span_end] = True

    for candidate_start, candidate_end in span_info.candidate_spans:
        allowed[candidate_start:candidate_end, span_start:span_end] = True
        allowed[candidate_start:candidate_end, candidate_start:candidate_end] = True

    allowed &= attention_2d[0].bool().unsqueeze(0)
    blocked = torch.finfo(dtype).min
    mask = torch.zeros((1, 1, sequence_length, sequence_length), device=device, dtype=dtype)
    return mask.masked_fill(~allowed, blocked)


def build_attention_mask(attention_2d: Any, span_info: SpanInfo, cfg: Any, dtype: Any) -> Any:
    mode = parse_attention_mask_mode(getattr(cfg, "attention_mask", "causal"))
    if mode is AttentionMaskMode.CAUSAL:
        return make_4d_causal_mask_from_2d(attention_2d, dtype)
    return make_span_item_block_mask(
        attention_2d,
        span_info,
        dtype,
        span_causal=bool(getattr(cfg, "span_causal", True)),
    )


def make_shared_position_ids(input_ids: Any, span_info: SpanInfo) -> Any:
    import torch

    _, sequence_length = input_ids.shape
    positions = torch.zeros(sequence_length, dtype=torch.long, device=input_ids.device)
    span_start, span_end = span_info.span_start, span_info.span_end
    span_length = span_end - span_start
    positions[span_start:span_end] = torch.arange(span_length, device=input_ids.device)
    for candidate_start, candidate_end in span_info.candidate_spans:
        candidate_length = candidate_end - candidate_start
        positions[candidate_start:candidate_end] = torch.arange(
            span_length,
            span_length + candidate_length,
            device=input_ids.device,
        )
    return positions.unsqueeze(0)


def build_position_ids(input_ids: Any, span_info: SpanInfo, cfg: Any) -> Any | None:
    mode = parse_position_id_mode(getattr(cfg, "position_ids", "standard"))
    if mode is PositionIdMode.STANDARD:
        return None
    return make_shared_position_ids(input_ids, span_info)


def validate_candidate_count(span_info: SpanInfo, expected: int) -> None:
    observed = len(span_info.candidate_spans)
    if observed != expected:
        raise ValueError(f"Expected {expected} candidate spans, found {observed}.")


class MeanLogProbListwiseScorer:
    def __init__(self, backbone: Any, tokenizer: Any, cfg: Any):
        import torch.nn as nn

        class _Scorer(nn.Module):
            def __init__(self, outer: MeanLogProbListwiseScorer):
                super().__init__()
                self.outer = outer
                self.backbone = outer.backbone

            def forward(self, input_ids: Any, attention_mask: Any):
                return self.outer(input_ids, attention_mask)

        self.backbone = backbone
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.span_extractor = SpanExtractor(tokenizer, cfg)
        self.module = _Scorer(self)

    def to(self, *args: Any, **kwargs: Any):
        self.module.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        self.module.train(mode)
        return self

    def eval(self):
        self.module.eval()
        return self

    def parameters(self):
        return self.module.parameters()

    def __call__(self, input_ids: Any, attention_mask: Any):
        import torch
        import torch.nn.functional as functional

        span_info = self.span_extractor(input_ids)
        dtype = next(self.backbone.parameters()).dtype
        attention = build_attention_mask(attention_mask, span_info, self.cfg, dtype)
        position_ids = build_position_ids(input_ids, span_info, self.cfg)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention,
            position_ids=position_ids,
            use_cache=False,
        )
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        log_probabilities = functional.log_softmax(logits.float(), dim=-1)
        token_log_probabilities = torch.gather(
            log_probabilities,
            dim=-1,
            index=labels.unsqueeze(-1),
        ).squeeze(-1)

        scores = []
        for start, end in span_info.candidate_spans:
            shifted_start = max(start - 1, 0)
            shifted_end = max(end - 1, shifted_start + 1)
            scores.append(token_log_probabilities[0, shifted_start:shifted_end].mean())
        return torch.stack(scores)


def align_scores_to_shared_candidates(scores_list: list[Any], permutations: list[list[int]]):
    visible_sets = [set(permutation[: scores.numel()]) for scores, permutation in zip(scores_list, permutations)]
    shared = set.intersection(*visible_sets) if visible_sets else set()
    if len(shared) < 2:
        return None, None

    shared_list = sorted(shared)
    aligned = []
    for scores, permutation in zip(scores_list, permutations):
        positions = {candidate: index for index, candidate in enumerate(permutation[: scores.numel()])}
        aligned.append(scores[[positions[candidate] for candidate in shared_list]])
    return aligned, shared_list


def build_permutation_rank_record(
    *,
    sample_index: int,
    user_id: str,
    candidate_ids: list[str],
    permutation: list[int],
    scores: Any,
    relevance: list[int],
) -> dict[str, Any]:
    score_values = [float(value) for value in scores.detach().cpu().tolist()]
    rows = []
    for input_position, candidate_index in enumerate(permutation[: len(score_values)]):
        rows.append(
            {
                "candidate_index": int(candidate_index),
                "candidate_id": candidate_ids[candidate_index],
                "input_position": int(input_position),
                "score": score_values[input_position],
                "relevance": int(relevance[input_position]),
            }
        )
    return {
        "sample_index": sample_index,
        "user_id": user_id,
        "permutation": permutation,
        "ranking": sorted(rows, key=lambda row: row["score"], reverse=True),
    }


def build_rank_record(batch: dict[str, Any], scores_list: list[Any], permutations: list[list[int]]) -> dict[str, Any]:
    candidates = [copy.deepcopy(candidate) for candidate in batch.get("candidates", [])]
    candidate_ids = list(batch.get("candidate_ids", []))
    if not candidate_ids:
        candidate_ids = [str(index) for index in range(len(candidates))]

    record = {
        "sample_index": int(batch.get("sample_index", -1)),
        "user_id": batch.get("user_id"),
        "split": batch.get("split"),
        "list_length": int(batch.get("list_length", len(candidates))),
        "num_items": int(batch.get("num_items", len(candidates))),
        "history": copy.deepcopy(batch.get("history", [])),
        "candidates": candidates,
        "permutations": [],
    }
    relevance_sequences = batch.get("relevance", [])
    for permutation_index, (scores, permutation) in enumerate(zip(scores_list, permutations)):
        tensor = scores.detach().float().cpu()
        limit = min(int(tensor.numel()), len(permutation))
        visible_permutation = [int(value) for value in permutation[:limit]]
        visible_scores = [float(value) for value in tensor[:limit].tolist()]
        visible_relevance = []
        if permutation_index < len(relevance_sequences):
            visible_relevance = [int(value) for value in relevance_sequences[permutation_index][:limit]]

        input_item_ids = [candidate_ids[index] if index < len(candidate_ids) else None for index in visible_permutation]
        ranking_pairs = sorted(
            zip(visible_permutation, input_item_ids, visible_scores),
            key=lambda value: value[2],
            reverse=True,
        )
        record["permutations"].append(
            {
                "permutation_index": int(permutation_index),
                "input": {
                    "candidate_indices": visible_permutation,
                    "item_ids": input_item_ids,
                    "relevance": visible_relevance,
                },
                "scores": visible_scores,
                "output_ranking": {
                    "candidate_indices": [index for index, _, _ in ranking_pairs],
                    "item_ids": [item_id for _, item_id, _ in ranking_pairs],
                    "scores": [score for _, _, score in ranking_pairs],
                },
            }
        )
    return record


__all__ = [
    "AttentionMaskMode",
    "MeanLogProbListwiseScorer",
    "PositionIdMode",
    "SpanExtractor",
    "SpanInfo",
    "align_scores_to_shared_candidates",
    "build_attention_mask",
    "build_lora_model",
    "build_model_for_ranking",
    "build_permutation_rank_record",
    "build_position_ids",
    "build_rank_record",
    "load_base_model",
    "load_model_for_ranking",
    "load_tokenizer",
    "make_4d_causal_mask_from_2d",
    "make_shared_position_ids",
    "make_span_item_block_mask",
    "model_dtype",
    "parse_attention_mask_mode",
    "parse_position_id_mode",
    "resolve_dtype",
    "select_device",
    "validate_candidate_count",
    "validate_special_tokens",
]

# Backwards-friendly spelling for callers that prefer a build verb.
build_model_for_ranking = load_model_for_ranking

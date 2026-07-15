"""Generated-output ranking backend for research baselines."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

from invarirank import RankedItem, RankingResult, RankingSample, Reranker

from .prompts import PROMPT_VERSIONS, build_research_prompt, parse_generated_ranking


@dataclass(frozen=True)
class GeneratedRerankerConfig:
    output_count: int | None = None
    max_length: int = 4096
    max_new_tokens: int | None = None
    do_sample: bool = False
    num_beams: int = 1
    temperature: float = 1.0
    incomplete_output: str = "append_input_order"
    allow_fenced_json: bool = True
    use_chat_template: bool = True
    seed: int = 42
    batch_size: int = 1
    top_one_generation: bool = False

    def __post_init__(self) -> None:
        if self.output_count is not None and self.output_count < 1:
            raise ValueError("output_count must be positive or omitted.")
        if self.max_length < 1 or self.batch_size < 1:
            raise ValueError("max_length and batch_size must be positive.")
        if self.max_new_tokens is not None and self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive or omitted for automatic sizing.")
        if self.num_beams < 1:
            raise ValueError("num_beams must be at least one.")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if self.incomplete_output not in {"append_input_order", "error"}:
            raise ValueError(f"Unsupported incomplete-output policy: {self.incomplete_output}")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> GeneratedRerankerConfig:
        if "prompt" in values and values["prompt"] != "rankgpt":
            raise ValueError("Generated research methods only support the RankGPT prompt.")
        known = {field.name for field in fields(cls)}
        data = dict(values)
        return cls(**{key: value for key, value in data.items() if key in known})


class GeneratedRankingReranker(Reranker):
    """Turn a generated JSON ranking into the shared complete ranking contract."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: GeneratedRerankerConfig | Mapping[str, Any] | None = None,
        *,
        device: Any | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = (
            config
            if isinstance(config, GeneratedRerankerConfig)
            else GeneratedRerankerConfig.from_mapping(config or {})
        )
        self.device = device if device is not None else _model_device(model)
        if hasattr(self.model, "eval"):
            self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        config: GeneratedRerankerConfig | Mapping[str, Any] | None = None,
        adapter_path: str | None = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        trust_remote_code: bool = False,
    ) -> GeneratedRankingReranker:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        dtype_value = getattr(torch, dtype, None)
        if dtype_value is None:
            raise ValueError(f"Unsupported dtype: {dtype}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=os.environ.get("HF_TOKEN"),
            trust_remote_code=trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype_value,
            token=os.environ.get("HF_TOKEN"),
            trust_remote_code=trust_remote_code,
        )
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        return cls(model.to(resolved_device), tokenizer, config, device=resolved_device)

    def rank(
        self,
        sample: RankingSample | Mapping[str, Any],
        *,
        permutation: Sequence[int] | None = None,
    ) -> RankingResult:
        return self.rank_many([(sample, permutation)])[0]

    def rank_many(
        self,
        requests: Sequence[tuple[RankingSample | Mapping[str, Any], Sequence[int] | None]],
        *,
        batch_size: int | None = None,
    ) -> list[RankingResult]:
        """Rank independent requests in padded generation batches."""
        if not requests:
            return []
        size = int(batch_size or self.config.batch_size)
        if size < 1:
            raise ValueError("batch_size must be positive.")
        results: list[RankingResult] = []
        for start in range(0, len(requests), size):
            results.extend(self._rank_batch(requests[start : start + size]))
        return results

    def _rank_batch(
        self,
        requests: Sequence[tuple[RankingSample | Mapping[str, Any], Sequence[int] | None]],
    ) -> list[RankingResult]:
        import torch

        prepared: list[tuple[RankingSample, list[int]]] = []
        prompts = []
        for sample, permutation in requests:
            ranking_sample = sample if isinstance(sample, RankingSample) else RankingSample.from_dict(sample)
            resolved_permutation = _permutation(permutation, len(ranking_sample.candidates))
            prompt = build_research_prompt(
                ranking_sample.to_dict(),
                resolved_permutation,
                output_count=self._output_count(len(resolved_permutation)),
            )
            prepared.append((ranking_sample, resolved_permutation))
            prompts.append(self._render_chat_prompt(prompt))
        tokenize_kwargs: dict[str, Any] = {
            "return_tensors": "pt",
            "truncation": True,
            "max_length": self.config.max_length,
        }
        if len(prompts) > 1:
            tokenize_kwargs["padding"] = True
        encoded = self.tokenizer(prompts if len(prompts) > 1 else prompts[0], **tokenize_kwargs)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        padded_input_length = int(input_ids.shape[-1])
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens
            or _automatic_max_new_tokens(self._output_count(len(prepared[0][1])) or len(prepared[0][1])),
            "do_sample": self.config.do_sample,
            "num_beams": self.config.num_beams,
        }
        if self.config.do_sample:
            generation_kwargs["temperature"] = self.config.temperature
            torch.manual_seed(self.config.seed)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None and eos_token_id is not None:
            generation_kwargs["pad_token_id"] = eos_token_id

        started = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
        latency = time.perf_counter() - started
        batch_results = []
        for row, (ranking_sample, resolved_permutation) in enumerate(prepared):
            generated_ids = outputs[row, padded_input_length:]
            raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            parsed = parse_generated_ranking(
                raw_output,
                resolved_permutation,
                expected_count=self._output_count(len(resolved_permutation)),
                incomplete_output=self.config.incomplete_output,
                allow_fenced_json=self.config.allow_fenced_json,
            )
            input_positions = {candidate: position for position, candidate in enumerate(resolved_permutation)}
            count = len(parsed.order)
            items = tuple(
                RankedItem(
                    candidate_index=index,
                    item_id=_candidate_id(ranking_sample.candidates[index], index),
                    score=float(count - rank),
                    input_position=input_positions[index],
                    relevance=_relevance(ranking_sample.candidates[index]),
                    candidate=dict(ranking_sample.candidates[index]),
                )
                for rank, index in enumerate(parsed.order)
            )
            input_tokens = int(attention_mask[row].sum().item()) if attention_mask is not None else padded_input_length
            output_count = self._output_count(len(resolved_permutation))
            prompt_version = (
                "rankgpt-top1-json-v1"
                if output_count == 1
                else "rankgpt-topk-json-v1"
                if output_count is not None
                else PROMPT_VERSIONS["rankgpt"]
            )
            metadata = {
                "method": "generated",
                "output_backend": "generate",
                "prompt_family": "rankgpt",
                "prompt_version": prompt_version,
                "top_one_generation": self.config.top_one_generation,
                "output_count": output_count,
                "raw_output": raw_output,
                "input_tokens": input_tokens,
                "generated_tokens": int(generated_ids.numel()),
                "latency_seconds": float(latency) / len(prepared),
                "generation_calls": 1,
                "generation_batches": 1.0 / len(prepared),
                "generation_config": generation_kwargs,
                "incomplete_output": self.config.incomplete_output,
                "retry_policy": "none",
                "retry_count": 0,
                **parsed.metadata(),
                "unknown_label_count": len(parsed.unknown_labels),
                "duplicate_label_count": len(parsed.duplicate_labels),
                "missing_label_count": len(parsed.unreturned_labels),
            }
            batch_results.append(
                RankingResult(
                    user_id=ranking_sample.user_id,
                    items=items,
                    permutation=tuple(resolved_permutation),
                    split=ranking_sample.split,
                    metadata=metadata,
                )
            )
        return batch_results

    def _output_count(self, candidate_count: int) -> int | None:
        if self.config.top_one_generation:
            return 1
        if self.config.output_count is None or self.config.output_count >= candidate_count:
            return None
        return self.config.output_count

    def _render_chat_prompt(self, prompt: str) -> str:
        if (
            self.config.use_chat_template
            and getattr(self.tokenizer, "chat_template", None)
            and hasattr(self.tokenizer, "apply_chat_template")
        ):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt


def _automatic_max_new_tokens(candidate_count: int) -> int:
    return max(32, 8 * candidate_count + 16)


def _model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return "cpu"


def _permutation(permutation: Sequence[int] | None, count: int) -> list[int]:
    resolved = list(range(count)) if permutation is None else [int(value) for value in permutation]
    if len(resolved) != count or set(resolved) != set(range(count)):
        raise ValueError(f"permutation must contain every candidate index from 0 to {count - 1} exactly once.")
    return resolved


def _candidate_id(candidate: Mapping[str, Any], fallback: int) -> str:
    for key in ("item_id", "id", "asin", "movie_id"):
        if key in candidate:
            return str(candidate[key])
    return str(fallback)


def _relevance(candidate: Mapping[str, Any]) -> int | None:
    value = candidate.get("relevance")
    return None if value is None else int(value)


__all__ = ["GeneratedRankingReranker", "GeneratedRerankerConfig"]

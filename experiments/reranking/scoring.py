from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from invarirank import InvariRankReranker, RerankerConfig
from invarirank.contracts import RankedItem, RankingResult, RankingSample, Reranker

from experiments.reranking.parsers import parse_rankgpt_output
from experiments.reranking.prompt_builder import (
    RANKGPT_PROMPT_VERSION,
    build_rankgpt_prompt,
    validate_prompt,
)

SCORING_OPTIONS = ("generation", "marker_logprob")


@dataclass(frozen=True)
class ScoringConfig:
    scoring: str = "marker_logprob"
    prompt: str = "marker"
    model_name_or_path: str = "meta-llama/Llama-3.2-3B-Instruct"
    adapter_path: str | None = None
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = False
    max_length: int = 4096
    max_new_tokens: int | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    num_beams: int = 1
    parser_repair: str = "append_input_order"
    seed: int = 42
    output_count: int | None = None
    architecture: str = "lft"

    def __post_init__(self) -> None:
        if self.scoring not in SCORING_OPTIONS:
            raise ValueError(f"Unknown scoring '{self.scoring}'. Valid scoring options: {list(SCORING_OPTIONS)}")
        validate_prompt(self.prompt)
        if self.scoring == "generation" and self.prompt != "rankgpt":
            raise ValueError("scoring='generation' requires prompt='rankgpt'.")
        if self.scoring == "marker_logprob" and self.prompt != "marker":
            raise ValueError("scoring='marker_logprob' requires prompt='marker'.")
        if self.architecture not in {"lft", "invarirank"}:
            raise ValueError("architecture must be either 'lft' or 'invarirank'.")


def load_scorer(config: ScoringConfig) -> Reranker:
    if config.scoring == "generation":
        return GeneratedOutputScorer.from_pretrained(config)
    return MarkerLogProbScorer.from_pretrained(config)


class GeneratedOutputScorer(Reranker):
    def __init__(self, model: Any, tokenizer: Any, config: ScoringConfig, *, device: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        if hasattr(model, "eval"):
            model.eval()

    @classmethod
    def from_pretrained(cls, config: ScoringConfig) -> GeneratedOutputScorer:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = torch.device(_resolve_device(config.device))
        dtype_name = _resolve_dtype(config.torch_dtype)
        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            raise ValueError(f"Unsupported torch dtype: {config.torch_dtype}")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            token=os.environ.get("HF_TOKEN"),
            trust_remote_code=config.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=dtype,
            token=os.environ.get("HF_TOKEN"),
            trust_remote_code=config.trust_remote_code,
        )
        if config.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, config.adapter_path)
        return cls(model.to(device), tokenizer, config, device=device)

    def rank(
        self,
        sample: RankingSample | Mapping[str, Any],
        *,
        permutation: Sequence[int] | None = None,
    ) -> RankingResult:
        return self.rank_many([(sample, permutation)], batch_size=1)[0]

    def rank_many(
        self,
        samples: Sequence[
            RankingSample | Mapping[str, Any] | tuple[RankingSample | Mapping[str, Any], Sequence[int] | None]
        ],
        *,
        permutations: Sequence[Sequence[int] | None] | None = None,
        batch_size: int = 1,
    ) -> list[RankingResult]:
        requests = _normalize_requests(samples, permutations)
        results = []
        for start in range(0, len(requests), max(1, int(batch_size))):
            results.extend(self._rank_batch(requests[start : start + max(1, int(batch_size))]))
        return results

    def _rank_batch(
        self,
        requests: Sequence[tuple[RankingSample | Mapping[str, Any], Sequence[int] | None]],
    ) -> list[RankingResult]:
        import torch

        prepared = []
        prompts = []
        for sample, permutation in requests:
            ranking_sample = sample if isinstance(sample, RankingSample) else RankingSample.from_dict(sample)
            resolved = _permutation(permutation, len(ranking_sample.candidates))
            prompts.append(
                build_rankgpt_prompt(
                    ranking_sample.to_dict(),
                    resolved,
                    output_count=self.config.output_count,
                )
            )
            prepared.append((ranking_sample, resolved))

        rendered = [self._render_chat_prompt(prompt) for prompt in prompts]
        encoded = self.tokenizer(
            rendered if len(rendered) > 1 else rendered[0],
            return_tensors="pt",
            padding=len(rendered) > 1,
            truncation=True,
            max_length=self.config.max_length,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens
            or _automatic_max_new_tokens(self.config.output_count or len(prepared[0][1])),
            "do_sample": self.config.temperature > 0,
            "num_beams": self.config.num_beams,
        }
        if self.config.temperature > 0:
            generation_kwargs["temperature"] = self.config.temperature
            generation_kwargs["top_p"] = self.config.top_p
            torch.manual_seed(self.config.seed)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            generation_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

        started = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)
        latency = (time.perf_counter() - started) / len(prepared)
        input_length = int(input_ids.shape[-1])

        results = []
        for row, (sample, resolved) in enumerate(prepared):
            generated_ids = outputs[row, input_length:]
            raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            parsed = parse_rankgpt_output(
                raw_output,
                resolved,
                expected_count=self.config.output_count,
                incomplete_output=self.config.parser_repair,
            )
            results.append(_result_from_order(sample, resolved, parsed.order, metadata={
                "scoring": "generation",
                "prompt": "rankgpt",
                "output_backend": "generate",
                "prompt_family": "rankgpt",
                "prompt_version": RANKGPT_PROMPT_VERSION,
                "prompt_text": prompts[row],
                "raw_output": raw_output,
                "generated_tokens": int(generated_ids.numel()),
                "latency_seconds": latency,
                "generation_config": generation_kwargs,
                **parsed.metadata(),
            }))
        return results

    def _render_chat_prompt(self, prompt: str) -> str:
        if getattr(self.tokenizer, "chat_template", None) and hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt


class MarkerLogProbScorer:
    @classmethod
    def from_pretrained(cls, config: ScoringConfig) -> Reranker:
        runtime_values = {
            "device": _resolve_device(config.device),
            "dtype": _resolve_dtype(config.torch_dtype),
            "max_length": config.max_length,
            "trust_remote_code": config.trust_remote_code,
        }
        if config.adapter_path and (Path(config.adapter_path) / "invarirank_config.json").is_file():
            return InvariRankReranker.from_pretrained(
                config.adapter_path,
                config=runtime_values,
            )

        reranker_config = RerankerConfig.for_method(
            config.architecture,
            {
                **runtime_values,
                "prompt_template": "invarirank",
            },
        )
        return InvariRankReranker.from_pretrained(
            config.model_name_or_path,
            config=reranker_config,
            adapter_path=config.adapter_path,
        )


def _result_from_order(
    sample: RankingSample,
    permutation: Sequence[int],
    order: Sequence[int],
    *,
    metadata: Mapping[str, Any],
) -> RankingResult:
    input_positions = {candidate: position for position, candidate in enumerate(permutation)}
    count = len(order)
    items = tuple(
        RankedItem(
            candidate_index=index,
            item_id=_candidate_id(sample.candidates[index], index),
            score=float(count - rank),
            input_position=input_positions[index],
            relevance=_relevance(sample.candidates[index]),
            candidate=dict(sample.candidates[index]),
        )
        for rank, index in enumerate(order)
    )
    return RankingResult(
        user_id=sample.user_id,
        items=items,
        permutation=tuple(permutation),
        split=sample.split,
        metadata=dict(metadata),
    )


def _normalize_requests(
    samples: Sequence[
        RankingSample | Mapping[str, Any] | tuple[RankingSample | Mapping[str, Any], Sequence[int] | None]
    ],
    permutations: Sequence[Sequence[int] | None] | None,
) -> list[tuple[RankingSample | Mapping[str, Any], Sequence[int] | None]]:
    values = list(samples)
    if permutations is not None:
        if len(permutations) != len(values):
            raise ValueError("permutations must contain one entry per sample.")
        if any(isinstance(value, tuple) for value in values):
            raise ValueError("Do not combine request tuples with the permutations argument.")
        return list(zip(values, permutations, strict=True))  # type: ignore[arg-type]
    return [(value[0], value[1]) if isinstance(value, tuple) else (value, None) for value in values]


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


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(dtype: str | None) -> str:
    if dtype and dtype != "auto":
        return dtype
    try:
        import torch
    except ImportError:
        return "float32"
    return "bfloat16" if torch.cuda.is_available() else "float32"


def _automatic_max_new_tokens(candidate_count: int) -> int:
    return max(32, 8 * candidate_count + 16)

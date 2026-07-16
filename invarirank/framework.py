from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

FINE_TUNED_METHODS = frozenset({"lft", "invarirank"})


@dataclass
class RankingSample:
    """A user context and the retrieved candidates to rerank."""

    user_id: str
    candidates: list[dict[str, Any]]
    history: list[dict[str, Any]] = field(default_factory=list)
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.user_id = str(self.user_id)
        self.history = [dict(item) for item in self.history]
        self.candidates = [dict(item) for item in self.candidates]
        self.metadata = dict(self.metadata)
        if not self.candidates:
            raise ValueError("RankingSample requires at least one candidate.")

    @classmethod
    def from_dict(cls, sample: Mapping[str, Any]) -> RankingSample:
        if "candidates" not in sample:
            raise ValueError("Ranking sample is missing required field: candidates")
        known = {"user_id", "history", "candidates", "split"}
        metadata = {key: value for key, value in sample.items() if key not in known}
        return cls(
            user_id=str(sample.get("user_id", "")),
            history=list(sample.get("history") or []),
            candidates=list(sample["candidates"]),
            split=sample.get("split"),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        sample = dict(self.metadata)
        sample.update(
            {
                "user_id": self.user_id,
                "history": [dict(item) for item in self.history],
                "candidates": [dict(item) for item in self.candidates],
            }
        )
        if self.split is not None:
            sample["split"] = self.split
        return sample


@dataclass(frozen=True)
class RankedItem:
    """One candidate in the final output order."""

    candidate_index: int
    item_id: str
    score: float
    input_position: int
    relevance: int | None = None
    candidate: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "item_id": self.item_id,
            "score": self.score,
            "input_position": self.input_position,
            "relevance": self.relevance,
            "candidate": dict(self.candidate),
        }


@dataclass(frozen=True)
class RankingResult:
    """A complete ranking plus the input permutation used to score it."""

    user_id: str
    items: tuple[RankedItem, ...]
    permutation: tuple[int, ...]
    split: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        candidate_indices = [item.candidate_index for item in self.items]
        if len(self.permutation) != len(set(self.permutation)):
            raise ValueError("RankingResult input permutation contains duplicate candidates.")
        if len(candidate_indices) != len(set(candidate_indices)):
            raise ValueError("RankingResult contains duplicate candidates.")
        if len(candidate_indices) != len(self.permutation) or set(candidate_indices) != set(self.permutation):
            raise ValueError("RankingResult must contain every candidate in the input permutation exactly once.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "split": self.split,
            "permutation": list(self.permutation),
            "items": [item.to_dict() for item in self.items],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RerankerConfig:
    """Validated configuration for framework-level inference."""

    model_name: str | None = None
    adapter_path: str | None = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    max_length: int = 4096
    prompt_template: str | None = "invarirank"
    span_start_token: str = "[SPAN]"
    span_end_token: str = "[/SPAN]"
    item_start_token: str = "[ITEM]"
    item_end_token: str = "[/ITEM]"
    attention_mask: str = "block"
    position_ids: str = "shared"
    span_causal: bool = True
    extras: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extras", dict(self.extras or {}))
        if self.max_length <= 0:
            raise ValueError("max_length must be greater than zero.")
        if self.attention_mask not in {"block", "causal"}:
            raise ValueError(f"Unsupported attention_mask: {self.attention_mask}")
        if self.position_ids not in {"shared", "standard"}:
            raise ValueError(f"Unsupported position_ids: {self.position_ids}")
        structural_tokens = (
            self.span_start_token,
            self.span_end_token,
            self.item_start_token,
            self.item_end_token,
        )
        if any(not token for token in structural_tokens):
            raise ValueError("Structural marker tokens must be non-empty.")
        if len(set(structural_tokens)) != len(structural_tokens):
            raise ValueError("Structural marker tokens must be distinct.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> RerankerConfig:
        data = dict(values)
        if "max_seq_length" in data and "max_length" not in data:
            data["max_length"] = data.pop("max_seq_length")
        known = {item.name for item in fields(cls)} - {"extras"}
        kwargs = {key: data.pop(key) for key in list(data) if key in known}
        return cls(**kwargs, extras=data)

    @classmethod
    def for_method(cls, method: str, values: Mapping[str, Any]) -> RerankerConfig:
        """Build one of the framework-owned fine-tuned reranker presets."""
        architectures = {
            "lft": ("causal", "standard"),
            "invarirank": ("block", "shared"),
        }
        if method not in architectures:
            raise ValueError(f"Unsupported framework method: {method}. Expected one of {sorted(architectures)}")
        resolved = dict(values)
        resolved["attention_mask"], resolved["position_ids"] = architectures[method]
        resolved["prompt_template"] = "invarirank"
        return cls.from_mapping(resolved)

    def to_dict(self) -> dict[str, Any]:
        """Return a round-trippable configuration mapping with flattened extras."""
        values = asdict(self)
        extras = values.pop("extras")
        return {**extras, **values}

    def save_json(self, path: str | Path) -> None:
        """Save this configuration as human-readable JSON."""
        _save_json_mapping(self.to_dict(), path)

    @classmethod
    def from_json(cls, path: str | Path) -> RerankerConfig:
        """Load and validate a configuration from JSON."""
        return cls.from_mapping(_load_json_mapping(path))

    def to_namespace(self, **overrides: Any) -> SimpleNamespace:
        values = dict(self.extras)
        values.update(asdict(self))
        values.pop("extras", None)
        values["max_seq_length"] = values.pop("max_length")
        values.update(overrides)
        return SimpleNamespace(**values)


class Reranker(ABC):
    """Common contract implemented by framework and research rerankers."""

    @abstractmethod
    def rank(
        self,
        sample: RankingSample | Mapping[str, Any],
        *,
        permutation: Sequence[int] | None = None,
    ) -> RankingResult:
        """Score and order every candidate in a sample."""

    def rank_many(
        self,
        samples: Sequence[
            RankingSample | Mapping[str, Any] | tuple[RankingSample | Mapping[str, Any], Sequence[int] | None]
        ],
        *,
        permutations: Sequence[Sequence[int] | None] | None = None,
        batch_size: int = 1,
    ) -> list[RankingResult]:
        """Rank requests in order, safely falling back to repeated single calls."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        requests = _normalize_rank_requests(samples, permutations)
        return [self.rank(sample, permutation=permutation) for sample, permutation in requests]


class InvariRankReranker(Reranker):
    """Framework facade backed by the current tested InvariRank implementation."""

    def __init__(
        self,
        backbone: Any,
        tokenizer: Any,
        config: RerankerConfig | Mapping[str, Any] | None = None,
        *,
        device: Any | None = None,
    ) -> None:
        from .modeling import MeanLogProbListwiseScorer, select_device

        self.config = _coerce_config(config)
        self.device = device if device is not None else select_device(self.config.device)
        self.tokenizer = tokenizer
        self.backbone = backbone
        self._legacy_config = self.config.to_namespace(device=str(self.device))
        self.scorer = MeanLogProbListwiseScorer(backbone, tokenizer, self._legacy_config).to(self.device)
        self.scorer.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        config: RerankerConfig | Mapping[str, Any] | None = None,
        adapter_path: str | None = None,
    ) -> InvariRankReranker:
        from .modeling import load_model_for_ranking, load_tokenizer, select_device

        framework_config = _coerce_config(config)
        cfg = framework_config.to_namespace(
            model_name=model_name,
            adapter_path=adapter_path if adapter_path is not None else framework_config.adapter_path,
        )
        device = select_device(cfg.device)
        cfg.device = str(device)
        tokenizer = load_tokenizer(cfg)
        backbone = load_model_for_ranking(cfg, tokenizer, device)
        resolved_config = RerankerConfig.from_mapping(vars(cfg))
        return cls(backbone, tokenizer, resolved_config, device=device)

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
        batch_size: int = 8,
    ) -> list[RankingResult]:
        """Rank recommendation samples in padded model-forward batches."""
        import torch

        from .prompts import build_prompt

        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        requests = _normalize_rank_requests(samples, permutations)
        prepared: list[tuple[RankingSample, list[int], str]] = []
        for sample, permutation in requests:
            ranking_sample = sample if isinstance(sample, RankingSample) else RankingSample.from_dict(sample)
            resolved = _validate_permutation(permutation, len(ranking_sample.candidates))
            prompt = build_prompt(ranking_sample.to_dict(), resolved, self._legacy_config)
            prepared.append((ranking_sample, resolved, prompt))

        results: list[RankingResult] = []
        self.scorer.eval()
        for start in range(0, len(prepared), batch_size):
            chunk = prepared[start : start + batch_size]
            encoded = self.tokenizer(
                [prompt for _, _, prompt in chunk],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            with torch.no_grad():
                score_batch = [
                    scores.detach().float().cpu() for scores in self.scorer.score_batch(input_ids, attention_mask)
                ]
            results.extend(
                _build_ranking_result(ranking_sample, resolved, scores)
                for (ranking_sample, resolved, _), scores in zip(chunk, score_batch, strict=True)
            )
        return results


def _coerce_config(config: RerankerConfig | Mapping[str, Any] | None) -> RerankerConfig:
    if config is None:
        return RerankerConfig()
    if isinstance(config, RerankerConfig):
        return config
    if isinstance(config, Mapping):
        return RerankerConfig.from_mapping(config)
    raise TypeError("config must be a RerankerConfig, mapping, or None.")


def _save_json_mapping(values: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    try:
        payload = json.dumps(dict(values), indent=2, sort_keys=True) + "\n"
    except TypeError as exc:
        raise TypeError(f"Configuration contains a value that cannot be serialized to JSON: {output}") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")


def _load_json_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        values = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON configuration: {source}") from exc
    if not isinstance(values, dict):
        raise ValueError(f"JSON configuration must contain an object: {source}")
    return values


def _validate_permutation(permutation: Sequence[int] | None, num_candidates: int) -> list[int]:
    resolved = list(range(num_candidates)) if permutation is None else [int(index) for index in permutation]
    if len(resolved) != num_candidates or set(resolved) != set(range(num_candidates)):
        raise ValueError(f"permutation must contain every candidate index from 0 to {num_candidates - 1} exactly once.")
    return resolved


def _normalize_rank_requests(
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

    requests = []
    for value in values:
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError("Rank request tuples must contain (sample, permutation).")
            requests.append((value[0], value[1]))
        else:
            requests.append((value, None))
    return requests


def _build_ranking_result(sample: RankingSample, permutation: list[int], scores: Any) -> RankingResult:
    from .prompts import candidate_id

    if int(scores.numel()) != len(sample.candidates):
        raise ValueError(
            f"The tokenized prompt produced {scores.numel()} candidate scores for {len(sample.candidates)} candidates. "
            "Increase max_length or shorten the history/candidate text."
        )
    ranked_items = []
    for input_position, candidate_index in enumerate(permutation):
        candidate = sample.candidates[candidate_index]
        relevance = candidate.get("relevance")
        ranked_items.append(
            RankedItem(
                candidate_index=candidate_index,
                item_id=candidate_id(candidate, candidate_index),
                score=float(scores[input_position].item()),
                input_position=input_position,
                relevance=None if relevance is None else int(relevance),
                candidate=dict(candidate),
            )
        )
    ranked_items.sort(key=lambda item: (-item.score, item.input_position))
    return RankingResult(
        user_id=sample.user_id,
        items=tuple(ranked_items),
        permutation=tuple(permutation),
        split=sample.split,
        metadata={
            "method": "invarirank",
            "output_backend": "span_logprob",
            "prompt_family": "invarirank_marker",
            "prompt_version": "invarirank-marker-v1",
        },
    )

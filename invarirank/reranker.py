from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .config import (
    FRAMEWORK_METADATA_NAME,
    INTERACTION_WEIGHTS_NAME,
    INVARIRANK_CONFIG_NAME,
    SAVED_FORMAT_VERSION,
    RerankerConfig,
    _load_json_mapping,
    _save_json_mapping,
)
from .contracts import RankedItem, RankingResult, RankingSample, Reranker, _normalize_rank_requests


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
        from .modeling import (
            MeanLogProbListwiseScorer,
            select_device,
            validate_model_architecture,
            validate_special_tokens,
        )

        self.config = _coerce_config(config)
        self.device = device if device is not None else select_device(self.config.device)
        self.tokenizer = tokenizer
        self.backbone = backbone
        self._legacy_config = self.config.to_namespace(device=str(self.device))
        validate_special_tokens(tokenizer, self._legacy_config)
        validate_model_architecture(backbone)
        self.scorer = MeanLogProbListwiseScorer(backbone, tokenizer, self._legacy_config).to(self.device)
        self.scorer.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str | Path,
        *,
        config: RerankerConfig | Mapping[str, Any] | None = None,
        adapter_path: str | None = None,
    ) -> InvariRankReranker:
        from .modeling import load_model_for_ranking, load_tokenizer, select_device

        model_path = Path(model_name)
        if model_path.is_dir() and _looks_like_saved_directory(model_path):
            if adapter_path is not None:
                raise ValueError("adapter_path must not be supplied when loading a saved InvariRank directory.")
            return cls._from_saved_directory(model_path, config=config)

        framework_config = _coerce_config(config)
        if framework_config.interaction != "none":
            raise ValueError(
                "Interaction-aware variants must be loaded from a complete saved InvariRank directory containing "
                "interaction weights. Use Trainer.from_pretrained(...) to initialize a new trainable variant."
            )
        cfg = framework_config.to_namespace(
            model_name=str(model_name),
            adapter_path=adapter_path if adapter_path is not None else framework_config.adapter_path,
        )
        device = select_device(cfg.device)
        cfg.device = str(device)
        tokenizer = load_tokenizer(cfg)
        backbone = load_model_for_ranking(cfg, tokenizer, device)
        resolved_config = RerankerConfig.from_mapping(vars(cfg))
        return cls(backbone, tokenizer, resolved_config, device=device)

    @classmethod
    def _from_saved_directory(
        cls,
        path: Path,
        *,
        config: RerankerConfig | Mapping[str, Any] | None,
    ) -> InvariRankReranker:
        from .modeling import load_model_for_ranking, load_tokenizer, select_device

        saved_config, metadata = _validate_saved_directory(path)
        framework_config = _merge_saved_config(saved_config, config)
        artifact_type = metadata["artifact_type"]
        base_model_name = metadata.get("base_model_name")
        runtime_model_name = str(base_model_name) if artifact_type == "adapter" else str(path)
        cfg = framework_config.to_namespace(
            model_name=runtime_model_name,
            base_model_name=runtime_model_name,
            tokenizer_name=str(path),
            adapter_path=str(path) if artifact_type == "adapter" else None,
            checkpoint_path=None,
        )
        device = select_device(cfg.device)
        cfg.device = str(device)
        tokenizer = load_tokenizer(cfg)
        backbone = load_model_for_ranking(cfg, tokenizer, device)
        resolved_values = framework_config.to_dict()
        resolved_values.update({"adapter_path": None, "device": str(device)})
        if resolved_values.get("model_name") is None:
            resolved_values["model_name"] = saved_config.model_name
        resolved_config = RerankerConfig.from_mapping(resolved_values)
        reranker = cls(backbone, tokenizer, resolved_config, device=device)
        reranker._load_interaction_weights(path)
        return reranker

    def save_pretrained(self, path: str | Path) -> None:
        """Save model or adapter weights, tokenizer, configuration, and format metadata."""
        output = Path(path)
        if output.exists() and not output.is_dir():
            raise ValueError(f"Saved InvariRank path exists and is not a directory: {output}")
        backbone_save = getattr(self.backbone, "save_pretrained", None)
        tokenizer_save = getattr(self.tokenizer, "save_pretrained", None)
        if not callable(backbone_save):
            raise TypeError("The reranker backbone does not implement save_pretrained().")
        if not callable(tokenizer_save):
            raise TypeError("The reranker tokenizer does not implement save_pretrained().")

        artifact_type = "adapter" if _is_adapter_backbone(self.backbone) else "model"
        base_model_name = _adapter_base_model_name(self.backbone) or self.config.model_name
        if artifact_type == "adapter" and not base_model_name:
            raise ValueError("Cannot save an adapter without its base model name in the model or RerankerConfig.")

        output.mkdir(parents=True, exist_ok=True)
        backbone_save(output)
        tokenizer_save(output)

        interaction_model = self.scorer.interaction_model
        interaction_path = output / INTERACTION_WEIGHTS_NAME
        if interaction_model is None:
            if interaction_path.exists():
                interaction_path.unlink()
        else:
            from safetensors.torch import save_file

            state = {
                name: tensor.detach().cpu().contiguous()
                for name, tensor in interaction_model.state_dict().items()
            }
            save_file(state, str(interaction_path))

        config_values = self.config.to_dict()
        for runtime_key in ("base_model_name", "checkpoint_path", "tokenizer_name"):
            config_values.pop(runtime_key, None)
        config_values["adapter_path"] = None
        if artifact_type == "adapter":
            config_values["model_name"] = str(base_model_name)
        saved_config = RerankerConfig.from_mapping(config_values)
        saved_config.save_json(output / INVARIRANK_CONFIG_NAME)
        _save_json_mapping(
            {
                "artifact_type": artifact_type,
                "base_model_name": str(base_model_name) if base_model_name else None,
                "format_version": SAVED_FORMAT_VERSION,
                "framework": "invarirank",
                "package_version": _package_version(),
            },
            output / FRAMEWORK_METADATA_NAME,
        )
        _validate_saved_directory(output)

    def _load_interaction_weights(self, path: Path) -> None:
        interaction_model = self.scorer.interaction_model
        weights_path = path / INTERACTION_WEIGHTS_NAME
        if interaction_model is None:
            if weights_path.is_file():
                raise ValueError(
                    f"Saved artifact {path} contains interaction weights but config interaction is 'none'."
                )
            return
        if not weights_path.is_file():
            raise ValueError(f"Saved interaction-aware InvariRank artifact is missing: {INTERACTION_WEIGHTS_NAME}")
        from safetensors.torch import load_file

        state = load_file(str(weights_path), device="cpu")
        try:
            interaction_model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ValueError(f"Interaction weights are incompatible with the saved configuration in {path}.") from exc

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
            _validate_recommendation_sample(ranking_sample)
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
                    scores.detach().float().cpu()
                    for scores in self.scorer.score_batch(
                        input_ids,
                        attention_mask,
                        expected_candidate_counts=[len(sample.candidates) for sample, _, _ in chunk],
                    )
                ]
            results.extend(
                _build_ranking_result(
                    ranking_sample,
                    resolved,
                    scores,
                    prompt_text=prompt,
                    method=self.config.method,
                    interaction=self.config.interaction,
                )
                for (ranking_sample, resolved, prompt), scores in zip(chunk, score_batch, strict=True)
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


def _looks_like_saved_directory(path: Path) -> bool:
    return (path / INVARIRANK_CONFIG_NAME).exists() or (path / FRAMEWORK_METADATA_NAME).exists()


def _validate_saved_directory(path: Path) -> tuple[RerankerConfig, dict[str, Any]]:
    if not path.is_dir():
        raise ValueError(f"Saved InvariRank path is not a directory: {path}")
    required = [INVARIRANK_CONFIG_NAME, FRAMEWORK_METADATA_NAME, "tokenizer_config.json"]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete saved InvariRank directory {path}; missing: {', '.join(missing)}")
    tokenizer_assets = ("tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json", "vocab.txt")
    if not _directory_has_any(path, tokenizer_assets):
        raise ValueError(f"Incomplete saved InvariRank directory {path}; missing tokenizer vocabulary files.")

    metadata = _load_json_mapping(path / FRAMEWORK_METADATA_NAME)
    if metadata.get("framework") != "invarirank":
        raise ValueError(f"Incompatible framework metadata in {path}: expected framework 'invarirank'.")
    format_version = metadata.get("format_version")
    if format_version not in {1, SAVED_FORMAT_VERSION}:
        raise ValueError(
            f"Unsupported saved InvariRank format version in {path}: {metadata.get('format_version')!r}; "
            f"expected one of {[1, SAVED_FORMAT_VERSION]}."
        )
    if not isinstance(metadata.get("package_version"), str) or not metadata["package_version"]:
        raise ValueError(f"Saved InvariRank metadata is missing package_version: {path}")
    artifact_type = metadata.get("artifact_type")
    if artifact_type not in {"adapter", "model"}:
        raise ValueError(f"Unsupported saved InvariRank artifact type in {path}: {artifact_type!r}.")

    saved_config = RerankerConfig.from_json(path / INVARIRANK_CONFIG_NAME)
    interaction_path = path / INTERACTION_WEIGHTS_NAME
    if format_version == 1 and saved_config.interaction != "none":
        raise ValueError("Saved format version 1 does not support interaction-aware InvariRank artifacts.")
    if saved_config.interaction == "none" and interaction_path.is_file():
        raise ValueError(
            f"Saved artifact {path} contains {INTERACTION_WEIGHTS_NAME} but config interaction is 'none'."
        )
    if saved_config.interaction != "none" and not interaction_path.is_file():
        raise ValueError(
            f"Incomplete interaction-aware InvariRank artifact {path}; missing: {INTERACTION_WEIGHTS_NAME}"
        )
    if artifact_type == "adapter":
        if not (path / "adapter_config.json").is_file():
            raise ValueError(f"Incomplete saved InvariRank adapter directory {path}; missing: adapter_config.json")
        if not _directory_has_any(path, ("adapter_model.safetensors", "adapter_model.bin")):
            raise ValueError(f"Incomplete saved InvariRank adapter directory {path}; missing adapter weights.")
        base_model_name = metadata.get("base_model_name")
        if not isinstance(base_model_name, str) or not base_model_name:
            raise ValueError(f"Saved InvariRank adapter metadata is missing base_model_name: {path}")
        if saved_config.model_name and saved_config.model_name != base_model_name:
            raise ValueError(
                f"Saved InvariRank adapter base model mismatch in {path}: "
                f"config has {saved_config.model_name!r}, metadata has {base_model_name!r}."
            )
    else:
        if not (path / "config.json").is_file():
            raise ValueError(f"Incomplete saved InvariRank model directory {path}; missing: config.json")
        model_weights = (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
        has_sharded_weights = any(path.glob("model-*.safetensors")) or any(path.glob("pytorch_model-*.bin"))
        if not _directory_has_any(path, model_weights) and not has_sharded_weights:
            raise ValueError(f"Incomplete saved InvariRank model directory {path}; missing model weights.")
    return saved_config, metadata


def _directory_has_any(path: Path, names: Sequence[str]) -> bool:
    return any((path / name).is_file() for name in names)


def _merge_saved_config(
    saved: RerankerConfig,
    override: RerankerConfig | Mapping[str, Any] | None,
) -> RerankerConfig:
    if override is None:
        return saved
    values = saved.to_dict()
    if isinstance(override, RerankerConfig):
        values.update(override.to_dict())
    elif isinstance(override, Mapping):
        values.update(override)
    else:
        raise TypeError("config must be a RerankerConfig, mapping, or None.")
    values.update({"adapter_path": None, "model_name": saved.model_name})
    return RerankerConfig.from_mapping(values)


def _is_adapter_backbone(backbone: Any) -> bool:
    return bool(getattr(backbone, "peft_config", None))


def _adapter_base_model_name(backbone: Any) -> str | None:
    peft_config = getattr(backbone, "peft_config", None)
    configs = peft_config.values() if isinstance(peft_config, Mapping) else [peft_config]
    for config in configs:
        value = getattr(config, "base_model_name_or_path", None)
        if value:
            return str(value)
    return None


def _package_version() -> str:
    try:
        return version("invarirank")
    except PackageNotFoundError:
        return "0.1.0"


def _validate_permutation(permutation: Sequence[int] | None, num_candidates: int) -> list[int]:
    resolved = list(range(num_candidates)) if permutation is None else list(permutation)
    if any(isinstance(index, bool) or not isinstance(index, int) for index in resolved):
        raise TypeError("permutation indices must be integers.")
    if len(resolved) != num_candidates or set(resolved) != set(range(num_candidates)):
        raise ValueError(f"permutation must contain every candidate index from 0 to {num_candidates - 1} exactly once.")
    return resolved


def _validate_recommendation_sample(sample: RankingSample) -> None:
    from .prompts import candidate_id

    candidate_ids = [candidate_id(candidate, index) for index, candidate in enumerate(sample.candidates)]
    indices_by_id: dict[str, list[int]] = {}
    for index, item_id in enumerate(candidate_ids):
        indices_by_id.setdefault(item_id, []).append(index)
    duplicates = {item_id: indices for item_id, indices in indices_by_id.items() if len(indices) > 1}
    if duplicates:
        details = ", ".join(f"{item_id!r} at indices {indices}" for item_id, indices in duplicates.items())
        raise ValueError(f"Candidate item IDs must be unique; duplicate IDs: {details}.")

    for index, candidate in enumerate(sample.candidates):
        title = candidate.get("title", candidate.get("name", ""))
        if title is None or not str(title).strip():
            raise ValueError(f"Candidate at index {index} must have a non-empty title or name.")


def _build_ranking_result(
    sample: RankingSample,
    permutation: list[int],
    scores: Any,
    *,
    prompt_text: str | None = None,
    method: str = "invarirank",
    interaction: str = "none",
) -> RankingResult:
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
            "method": method,
            "interaction": interaction,
            "output_backend": "span_logprob",
            "prompt_family": "invarirank_marker",
            "prompt_version": "invarirank-marker-v1",
            "prompt_text": prompt_text,
        },
    )


__all__ = ["InvariRankReranker"]

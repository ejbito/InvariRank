from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

FINE_TUNED_METHODS = frozenset({"lft", "invarirank"})
INVARIRANK_CONFIG_NAME = "invarirank_config.json"
FRAMEWORK_METADATA_NAME = "framework_metadata.json"
SAVED_FORMAT_VERSION = 1


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


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    train_num_permutations: int = 1
    eval_num_permutations: int = 10
    val_perms_deterministic: bool = True
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lambda_rank: float = 1.0
    lambda_perm: float = 0.0
    permutation_loss: str = "kl"
    num_epochs: int | None = None
    total_optimizer_steps: int | None = 500
    save_every_steps: int | None = None
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    extras: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extras", dict(self.extras or {}))
        if self.train_num_permutations < 1 or self.eval_num_permutations < 1:
            raise ValueError("Permutation counts must be at least one.")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least one.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")
        if self.num_epochs is None and self.total_optimizer_steps is None:
            raise ValueError("Set num_epochs or total_optimizer_steps.")
        if self.permutation_loss not in {"kl", "symkl", "jeffreys"}:
            raise ValueError(f"Unsupported permutation_loss: {self.permutation_loss}")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> TrainingConfig:
        data = dict(values)
        known = {item.name for item in fields(cls)} - {"extras"}
        kwargs = {key: data.pop(key) for key in list(data) if key in known}
        if "lora_target_modules" in kwargs:
            kwargs["lora_target_modules"] = tuple(kwargs["lora_target_modules"])
        return cls(**kwargs, extras=data)

    def to_dict(self) -> dict[str, Any]:
        """Return a round-trippable, JSON-compatible configuration mapping."""
        values = asdict(self)
        extras = values.pop("extras")
        values["lora_target_modules"] = list(values["lora_target_modules"])
        return {**extras, **values}

    def save_json(self, path: str | Path) -> None:
        """Save this configuration as human-readable JSON."""
        _save_json_mapping(self.to_dict(), path)

    @classmethod
    def from_json(cls, path: str | Path) -> TrainingConfig:
        """Load and validate a configuration from JSON."""
        return cls.from_mapping(_load_json_mapping(path))

    def to_namespace(self, **overrides: Any) -> SimpleNamespace:
        values = dict(self.extras)
        values.update(asdict(self))
        values.pop("extras", None)
        values["lora_target_modules"] = list(values["lora_target_modules"])
        values.update(overrides)
        return SimpleNamespace(**values)


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


__all__ = [
    "FINE_TUNED_METHODS",
    "FRAMEWORK_METADATA_NAME",
    "INVARIRANK_CONFIG_NAME",
    "RerankerConfig",
    "SAVED_FORMAT_VERSION",
    "TrainingConfig",
]

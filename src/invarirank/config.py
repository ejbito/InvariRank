from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .utils import dict_to_namespace, ensure_dir, namespace_to_dict


DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "device": "cuda",
    "dtype": "bfloat16",
    "max_seq_length": 4096,
    "span_start_token": "[SPAN]",
    "span_end_token": "[/SPAN]",
    "item_start_token": "[ITEM]",
    "item_end_token": "[/ITEM]",
    "attention_mask": "block",
    "position_ids": "shared",
    "span_causal": True,
    "train_num_permutations": 1,
    "eval_num_permutations": 10,
    "val_perms_deterministic": True,
    "gradient_accumulation_steps": 16,
    "learning_rate": 5e-5,
    "weight_decay": 0.0,
    "warmup_steps": 10,
    "max_grad_norm": 1.0,
    "lambda_rank": 1.0,
    "lambda_perm": 0.0,
    "permutation_loss": "kl",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "save_every_steps": 250,
}


def _load_raw_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError:
            data = _load_simple_yaml(text)
        else:
            data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain an object at top level: {path}")
    return data


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the flat YAML subset used by the repository's default configs."""
    data: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise RuntimeError("Install PyYAML to parse nested or advanced YAML configs.")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid YAML line: {raw_line}")
        data[key] = _parse_simple_yaml_value(value)
    return data


def _parse_simple_yaml_value(value: str) -> Any:
    if value == "":
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_simple_yaml_value(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(\d+\.\d*|\d*\.\d+)([eE][-+]?\d+)?|[-+]?\d+[eE][-+]?\d+", value):
        return float(value)
    return value


def _merge_defaults(data: dict[str, Any]) -> dict[str, Any]:
    merged = dict(DEFAULTS)
    merged.update(data)
    return merged


def _resolve_path(value: str, base_dir: Path) -> str:
    if not value:
        return value
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def resolve_config_paths(data: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    path_keys = {
        "train_path",
        "val_path",
        "data_path",
        "run_dir",
        "output_dir",
        "checkpoint_dir",
        "checkpoint_path",
        "resume_checkpoint_path",
        "adapter_path",
        "ranked_lists_path",
        "metrics_path",
    }
    resolved = dict(data)
    for key in path_keys:
        if isinstance(resolved.get(key), str):
            resolved[key] = _resolve_path(resolved[key], base_dir)
    return resolved


def make_config(data: dict[str, Any], *, base_dir: str | Path | None = None) -> SimpleNamespace:
    base = Path(base_dir or ".").resolve()
    merged = resolve_config_paths(_merge_defaults(data), base)

    run_dir = merged.get("run_dir") or merged.get("output_dir")
    if run_dir:
        ensure_dir(run_dir)
        merged.setdefault("checkpoint_dir", str(Path(run_dir) / "checkpoints"))
        ensure_dir(merged["checkpoint_dir"])

    return dict_to_namespace(merged)


def load_config(path: str | Path) -> SimpleNamespace:
    path = Path(path).resolve()
    return make_config(_load_raw_config(path), base_dir=path.parent)


def dump_config(cfg: SimpleNamespace, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(namespace_to_dict(cfg), f, ensure_ascii=False, indent=2)


def require_fields(cfg: SimpleNamespace, fields: list[str]) -> None:
    missing = [field for field in fields if not hasattr(cfg, field) or getattr(cfg, field) in {None, ""}]
    if missing:
        raise ValueError(f"Missing required config field(s): {', '.join(missing)}")

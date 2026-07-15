from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from invarirank import (
    FINE_TUNED_METHODS,
    RerankerConfig,
    Trainer,
    TrainingConfig,
)

from .baselines import (
    Stella,
    fit_stella_calibrator,
    load_backbone_method,
    load_ranking_backend,
    output_prompt_identity,
    resolve_output_backend,
    stella_provenance,
)
from .data import build_dataset_splits, write_dataset_splits
from .evaluation import evaluate

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "research" / "configs" / "paper.yaml"
DEFAULT_CANDIDATES_CONFIG = ROOT / "research" / "configs" / "candidates.yaml"
DEFAULT_TRAIN_CONFIG = ROOT / "research" / "configs" / "train.yaml"
DEFAULT_RANK_CONFIG = ROOT / "research" / "configs" / "rank.yaml"
DEFAULT_EVALUATE_CONFIG = ROOT / "research" / "configs" / "evaluate.yaml"
PATH_KEYS = {
    "adapter_path",
    "cache_dir",
    "data_path",
    "data_dir",
    "meta",
    "metrics_path",
    "movies",
    "output_dir",
    "ranked_lists_path",
    "ratings",
    "reviews",
    "run_dir",
    "train_path",
    "transition_matrix_output",
    "transition_matrix_path",
    "probe_path",
    "val_path",
}
TRAINABLE_METHODS = set(FINE_TUNED_METHODS)
REPRODUCTION_STAGES = ("candidates", "train", "rank", "evaluate")


@dataclass(frozen=True)
class CandidateArtifacts:
    train_path: str
    validation_path: str
    test_path: str
    counts: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_path": self.train_path,
            "validation_path": self.validation_path,
            "test_path": self.test_path,
            "counts": dict(self.counts),
        }


@dataclass(frozen=True)
class TrainingArtifacts:
    checkpoint_path: str
    result: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"checkpoint_path": self.checkpoint_path, "result": dict(self.result)}


@dataclass(frozen=True)
class RankingArtifacts:
    ranked_lists_path: str
    num_records: int

    def to_dict(self) -> dict[str, Any]:
        return {"ranked_lists_path": self.ranked_lists_path, "num_records": self.num_records}


@dataclass(frozen=True)
class EvaluationArtifacts:
    metrics_path: str | None
    report: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"metrics_path": self.metrics_path, "report": dict(self.report)}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Research config must contain an object: {config_path}")
    data = _resolve_config_references(data, config_path.parent)
    project_root = Path(data.get("project_root", ROOT))
    if not project_root.is_absolute():
        project_root = (ROOT / project_root).resolve()
    return _resolve_paths(data, project_root)


def _resolve_config_references(value: Any, config_dir: Path, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {name: _resolve_config_references(item, config_dir, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_resolve_config_references(item, config_dir, key) for item in value]
    if isinstance(value, str) and key == "config" and value:
        path = Path(value)
        return str(path if path.is_absolute() else (config_dir / path).resolve())
    return value


def _resolve_paths(value: Any, base_dir: Path, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {name: _resolve_paths(item, base_dir, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(item, base_dir, key) for item in value]
    if isinstance(value, str) and key in PATH_KEYS and value:
        path = Path(value)
        return str(path if path.is_absolute() else (base_dir / path).resolve())
    return value


def to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_namespace(item) for item in value]
    return value


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(value: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def write_yaml(value: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=True)


def section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{name}' must be an object.")
    return dict(value)


def generate_candidates(config: Mapping[str, Any], *, output_dir: str | None = None) -> dict[str, int]:
    data = section(config, "data") or dict(config)
    namespace = to_namespace(data)
    train, validation, test = build_dataset_splits(namespace)
    destination = output_dir or data.get("paths", {}).get("output_dir") or data.get("output_dir")
    if not destination:
        raise ValueError("Data config requires paths.output_dir or output_dir.")
    write_dataset_splits(train, validation, test, destination)
    return {"train": len(train), "validation": len(validation), "test": len(test)}


def train(
    config: Mapping[str, Any],
    *,
    model_name: str | None = None,
    output_dir: str | None = None,
    method: str = "invarirank",
) -> dict[str, Any]:
    values = section(config, "training") or dict(config)
    if method not in TRAINABLE_METHODS:
        raise ValueError(f"Method '{method}' does not require training.")
    method_options = _method_options(config, method)
    values.update({key: value for key, value in method_options.items() if key in {"run_dir", "output_dir"}})
    selected_model = model_name or values.get("model_name")
    if not selected_model:
        raise ValueError("Training config requires model_name.")
    train_path = values.get("train_path")
    validation_path = values.get("val_path")
    destination = output_dir or values.get("run_dir") or values.get("output_dir")
    if not train_path or not validation_path or not destination:
        raise ValueError("Training config requires train_path, val_path, and run_dir/output_dir.")
    train_samples = load_jsonl(train_path)
    validation_samples = load_jsonl(validation_path)
    if values.get("train_max_samples") is not None:
        train_samples = train_samples[: int(values["train_max_samples"])]
    if values.get("val_max_samples") is not None:
        validation_samples = validation_samples[: int(values["val_max_samples"])]
    trainer = Trainer.from_pretrained(
        selected_model,
        train_samples,
        validation_samples,
        reranker_config=RerankerConfig.for_method(method, values),
        training_config=TrainingConfig.from_mapping(values),
    )
    return trainer.train(output_dir=destination)


def deterministic_permutation(
    candidate_count: int,
    sample_index: int,
    permutation_index: int,
    seed: int = 0,
) -> list[int]:
    permutation = list(range(candidate_count))
    random.Random(seed * 1_000_003 + sample_index * 1009 + permutation_index).shuffle(permutation)
    return permutation


def result_permutation_record(result: Any, permutation_index: int) -> dict[str, Any]:
    by_index = {item.candidate_index: item for item in result.items}
    input_items = [by_index[index] for index in result.permutation]
    return {
        "permutation_index": permutation_index,
        "input": {
            "candidate_indices": list(result.permutation),
            "item_ids": [item.item_id for item in input_items],
            "relevance": [item.relevance or 0 for item in input_items],
        },
        "scores": [item.score for item in input_items],
        "output_ranking": {
            "candidate_indices": [item.candidate_index for item in result.items],
            "item_ids": [item.item_id for item in result.items],
            "scores": [item.score for item in result.items],
        },
        "metadata": dict(result.metadata),
    }


def rank(
    config: Mapping[str, Any],
    *,
    model_name: str | None = None,
    method: str = "invarirank",
    num_samples: int | None = None,
    permutations: int | None = None,
    backend: str | None = None,
) -> list[dict[str, Any]]:
    values = section(config, "ranking") or dict(config)
    values["generation"] = section(config, "generation")
    selected_model = model_name or values.get("model_name")
    data_path = values.get("data_path")
    if not selected_model or not data_path:
        raise ValueError("Ranking config requires model_name and data_path.")
    samples = load_jsonl(data_path)
    limit = num_samples if num_samples is not None else values.get("ranking_num_samples")
    if limit is not None:
        samples = samples[: int(limit)]
    method_options = _method_options(config, method)
    if backend is not None:
        method_options["backend"] = backend
    data_values = section(config, "data")
    dataset_values = data_values.get("dataset", {})
    if isinstance(dataset_values, Mapping):
        method_options.setdefault("dataset", dataset_values.get("name"))
    resolved_backend = resolve_output_backend(method, values, method_options)
    if method == "stella" and not method_options.get("transition_matrix_path"):
        probe_path = method_options.get("probe_path")
        if not probe_path:
            raise ValueError("STELLA requires methods.stella.probe_path when no transition matrix is provided.")
        probe_samples = load_jsonl(probe_path)
        if not probe_samples:
            raise ValueError("STELLA probing data is empty.")
        candidate_count = len(probe_samples[0].get("candidates", []))
        method_options["candidate_count"] = candidate_count
        base = load_ranking_backend(
            "stella",
            selected_model,
            values,
            method_options,
            backend=resolved_backend,
        )
        calibrator = fit_stella_calibrator(
            base,
            probe_samples,
            ensemble_steps=int(method_options.get("ensemble_steps", 5)),
            max_samples=method_options.get("probe_samples"),
            smoothing=float(method_options.get("smoothing", 1.0)),
            seed=int(method_options.get("seed", 42)),
            provenance=stella_provenance(
                "stella",
                selected_model,
                values,
                method_options,
                backend=resolved_backend,
                candidate_count=candidate_count,
            ),
            batch_size=int(method_options.get("batch_size", 1)),
        )
        matrix_output = method_options.get("transition_matrix_output")
        if matrix_output:
            calibrator.save(matrix_output)
        reranker = Stella(
            base,
            calibrator,
            max_updates=int(method_options.get("max_updates", 10)),
            aggregate_count=int(method_options.get("aggregate_count", 3)),
            seed=int(method_options.get("seed", 42)),
            batch_size=int(method_options.get("batch_size", 1)),
        )
    else:
        reranker = load_backbone_method(method, selected_model, values, method_options)
    permutation_count = int(permutations or values.get("eval_num_permutations", 1))
    seed = int(values.get("seed", 0))
    records = []
    for sample_index, sample in enumerate(samples):
        candidate_count = len(sample.get("candidates", []))
        if candidate_count == 0:
            continue
        permutation_records = []
        for permutation_index in range(permutation_count):
            permutation = deterministic_permutation(candidate_count, sample_index, permutation_index, seed)
            result = reranker.rank(sample, permutation=permutation)
            permutation_records.append(result_permutation_record(result, permutation_index))
        records.append(
            {
                "sample_index": sample_index,
                "method": method,
                "backend": resolved_backend,
                "user_id": sample.get("user_id"),
                "split": sample.get("split", "test"),
                "list_length": int(sample.get("list_length", candidate_count)),
                "num_items": candidate_count,
                "history": sample.get("history", []),
                "candidates": sample["candidates"],
                "permutations": permutation_records,
            }
        )
    output = method_options.get("ranked_lists_path") or method_options.get("output_path")
    if not output and method_options.get("output_dir"):
        output = str(Path(method_options["output_dir"]) / "ranked_lists.json")
    if not output:
        output = values.get("ranked_lists_path")
    if not output:
        output_dir = values.get("output_dir", ROOT / "runs" / "eval" / method)
        output = str(Path(output_dir) / "ranked_lists.json")
    write_json(records, output)
    return records


def _method_options(config: Mapping[str, Any], method: str) -> dict[str, Any]:
    methods = config.get("methods", {})
    if not isinstance(methods, Mapping):
        raise ValueError("Config section 'methods' must be an object.")
    options = methods.get(method, {})
    if not isinstance(options, Mapping):
        raise ValueError(f"Method config '{method}' must be an object.")
    return dict(options)


def evaluate_records(
    config: Mapping[str, Any],
    *,
    ranked_lists_path: str | None = None,
    output_path: str | None = None,
    top_k: Sequence[int] | None = None,
) -> dict[str, Any]:
    values = section(config, "evaluation") or dict(config)
    ranking_values = section(config, "ranking") or dict(config)
    source = ranked_lists_path or values.get("ranked_lists_path") or ranking_values.get("ranked_lists_path")
    if not source:
        raise ValueError("Evaluation requires ranked_lists_path.")
    text = Path(source).read_text(encoding="utf-8").strip()
    try:
        records = json.loads(text) if text else []
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(records, dict):
        records = [records]
    report = evaluate(records, top_k=tuple(top_k or values.get("top_k", [5, 10])))
    report["ranked_lists_path"] = str(source)
    report["efficiency"] = aggregate_efficiency(records)
    report["generation"] = aggregate_generation_validity(records)
    destination = output_path or values.get("metrics_path")
    if destination:
        write_json(report, destination)
    return report


def aggregate_efficiency(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    forward_passes = []
    method_counts: dict[str, int] = {}
    generation_calls = 0
    generation_batches = 0.0
    input_tokens = 0
    generated_tokens = 0
    latency_seconds = 0.0
    for record in records:
        for permutation in record.get("permutations", []):
            metadata = permutation.get("metadata", {})
            passes = int(metadata.get("forward_passes", 1))
            forward_passes.append(passes)
            method = str(metadata.get("method", record.get("method", "unknown")))
            method_counts[method] = method_counts.get(method, 0) + 1
            generation_calls += int(metadata.get("generation_calls", 0))
            generation_batches += float(metadata.get("generation_batches", metadata.get("generation_calls", 0)))
            input_tokens += int(metadata.get("input_tokens", 0))
            generated_tokens += int(metadata.get("generated_tokens", 0))
            latency_seconds += float(metadata.get("latency_seconds", 0.0))
    total = sum(forward_passes)
    return {
        "num_rankings": len(forward_passes),
        "total_forward_passes": total,
        "mean_forward_passes_per_ranking": (float(total / len(forward_passes)) if forward_passes else 0.0),
        "max_forward_passes_per_ranking": max(forward_passes, default=0),
        "rankings_by_method": method_counts,
        "generation_calls": generation_calls,
        "generation_batches": generation_batches,
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "generation_latency_seconds": latency_seconds,
    }


def aggregate_generation_validity(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = []
    unknown_labels = 0
    duplicate_labels = 0
    missing_labels = 0
    for record in records:
        for permutation in record.get("permutations", []):
            metadata = permutation.get("metadata", {})
            if metadata.get("output_backend") != "generate":
                continue
            values = metadata.get("parse_statuses")
            if isinstance(values, list):
                statuses.extend(str(value) for value in values)
            else:
                statuses.append(str(metadata.get("parse_status", "unknown")))
            unknown_labels += int(metadata.get("unknown_label_count", len(metadata.get("unknown_labels", []))))
            duplicate_labels += int(metadata.get("duplicate_label_count", len(metadata.get("duplicate_labels", []))))
            missing_labels += int(metadata.get("missing_label_count", len(metadata.get("missing_labels", []))))
    total = len(statuses)
    valid = statuses.count("valid")
    repaired = statuses.count("repaired")
    return {
        "num_generated_outputs": total,
        "valid_outputs": valid,
        "repaired_outputs": repaired,
        "failed_outputs": total - valid - repaired,
        "valid_output_rate": float(valid / total) if total else 0.0,
        "repaired_output_rate": float(repaired / total) if total else 0.0,
        "unknown_labels": unknown_labels,
        "duplicate_labels": duplicate_labels,
        "missing_labels": missing_labels,
        "unknown_labels_per_output": float(unknown_labels / total) if total else 0.0,
        "duplicate_labels_per_output": float(duplicate_labels / total) if total else 0.0,
        "missing_labels_per_output": float(missing_labels / total) if total else 0.0,
    }


def run_candidates_stage(
    config: Mapping[str, Any],
    *,
    output_dir: str | None = None,
) -> CandidateArtifacts:
    counts = generate_candidates(config) if output_dir is None else generate_candidates(config, output_dir=output_dir)
    data = section(config, "data") or dict(config)
    destination = output_dir or data.get("paths", {}).get("output_dir") or data.get("output_dir")
    if not destination:
        raise ValueError("Candidates stage requires an output directory.")
    directory = Path(destination)
    return CandidateArtifacts(
        train_path=str(directory / "train.jsonl"),
        validation_path=str(directory / "val.jsonl"),
        test_path=str(directory / "test.jsonl"),
        counts=counts,
    )


def run_train_stage(
    config: Mapping[str, Any],
    *,
    method: str = "invarirank",
    model_name: str | None = None,
    output_dir: str | None = None,
    train_path: str | None = None,
    validation_path: str | None = None,
) -> TrainingArtifacts:
    resolved = copy.deepcopy(dict(config))
    training_values = resolved.setdefault("training", {})
    if train_path is not None:
        training_values["train_path"] = train_path
    if validation_path is not None:
        training_values["val_path"] = validation_path
    train_kwargs: dict[str, Any] = {"method": method}
    if model_name is not None:
        train_kwargs["model_name"] = model_name
    if output_dir is not None:
        train_kwargs["output_dir"] = output_dir
    result = train(resolved, **train_kwargs)
    options = _method_options(resolved, method)
    destination = (
        output_dir or options.get("run_dir") or training_values.get("run_dir") or training_values.get("output_dir")
    )
    checkpoint = options.get("adapter_path")
    if not checkpoint and destination:
        checkpoint = str(Path(destination) / "checkpoints" / "final")
    if not checkpoint:
        raise ValueError("Train stage could not determine its final checkpoint path.")
    return TrainingArtifacts(checkpoint_path=str(checkpoint), result=result)


def run_rank_stage(
    config: Mapping[str, Any],
    *,
    method: str = "invarirank",
    model_name: str | None = None,
    num_samples: int | None = None,
    permutations: int | None = None,
    backend: str | None = None,
    data_path: str | None = None,
    adapter_path: str | None = None,
) -> RankingArtifacts:
    resolved = copy.deepcopy(dict(config))
    ranking_values = resolved.setdefault("ranking", {})
    if data_path is not None:
        ranking_values["data_path"] = data_path
    if adapter_path is not None:
        resolved.setdefault("methods", {}).setdefault(method, {})["adapter_path"] = adapter_path
    rank_kwargs: dict[str, Any] = {"method": method}
    if model_name is not None:
        rank_kwargs["model_name"] = model_name
    if num_samples is not None:
        rank_kwargs["num_samples"] = num_samples
    if permutations is not None:
        rank_kwargs["permutations"] = permutations
    if backend is not None:
        rank_kwargs["backend"] = backend
    records = rank(resolved, **rank_kwargs)
    options = _method_options(resolved, method)
    output = options.get("ranked_lists_path") or options.get("output_path")
    if not output and options.get("output_dir"):
        output = str(Path(options["output_dir"]) / "ranked_lists.json")
    if not output:
        output = ranking_values.get("ranked_lists_path")
    if not output:
        output = str(Path(ranking_values.get("output_dir", ROOT / "runs" / "eval" / method)) / "ranked_lists.json")
    return RankingArtifacts(ranked_lists_path=str(output), num_records=len(records))


def run_evaluate_stage(
    config: Mapping[str, Any],
    *,
    ranked_lists_path: str | None = None,
    output_path: str | None = None,
    top_k: Sequence[int] | None = None,
) -> EvaluationArtifacts:
    evaluate_kwargs: dict[str, Any] = {}
    if ranked_lists_path is not None:
        evaluate_kwargs["ranked_lists_path"] = ranked_lists_path
    if output_path is not None:
        evaluate_kwargs["output_path"] = output_path
    if top_k is not None:
        evaluate_kwargs["top_k"] = top_k
    report = evaluate_records(config, **evaluate_kwargs)
    evaluation_values = section(config, "evaluation") or dict(config)
    destination = output_path or evaluation_values.get("metrics_path")
    return EvaluationArtifacts(metrics_path=None if destination is None else str(destination), report=report)


def run_pipeline(*, method: str = "invarirank", dry_run: bool = False) -> dict[str, Any]:
    stage_paths = {
        "candidates": str(DEFAULT_CANDIDATES_CONFIG),
        "train": str(DEFAULT_TRAIN_CONFIG),
        "rank": str(DEFAULT_RANK_CONFIG),
        "evaluate": str(DEFAULT_EVALUATE_CONFIG),
    }
    stage_configs = {stage: load_config(path) for stage, path in stage_paths.items()}

    stages: dict[str, Any] = {}
    if dry_run:
        for stage in REPRODUCTION_STAGES:
            status = "not_applicable" if stage == "train" and method not in TRAINABLE_METHODS else "planned"
            stages[stage] = {"status": status, "config": stage_paths[stage]}
        return {"method": method, "dry_run": True, "stages": stages}

    candidates = run_candidates_stage(stage_configs["candidates"])
    stages["candidates"] = {"status": "completed", **candidates.to_dict()}
    checkpoint_path = None
    if method in TRAINABLE_METHODS:
        trained = run_train_stage(
            stage_configs["train"],
            method=method,
            train_path=candidates.train_path,
            validation_path=candidates.validation_path,
        )
        checkpoint_path = trained.checkpoint_path
        stages["train"] = {"status": "completed", **trained.to_dict()}
    else:
        stages["train"] = {"status": "not_applicable"}
    ranked = run_rank_stage(
        stage_configs["rank"],
        method=method,
        data_path=candidates.test_path,
        adapter_path=checkpoint_path,
    )
    stages["rank"] = {"status": "completed", **ranked.to_dict()}
    evaluated = run_evaluate_stage(
        stage_configs["evaluate"],
        ranked_lists_path=ranked.ranked_lists_path,
        output_path=str(Path(ranked.ranked_lists_path).with_name("metrics.json")),
    )
    stages["evaluate"] = {"status": "completed", **evaluated.to_dict()}
    return {"method": method, "dry_run": False, "stages": stages}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(output.get(key), Mapping):
            output[key] = deep_merge(output[key], value)
        else:
            output[key] = copy.deepcopy(value)
    return output


def _slug(value: Any) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return slug or "value"


def _matrix_entries(value: Any, name: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"reproduction.{name} must be a non-empty object.")
    output = {}
    for alias, options in value.items():
        if isinstance(options, str):
            options = {"model_name": options}
        if not isinstance(options, Mapping):
            raise ValueError(f"reproduction.{name}.{alias} must be an object or model name.")
        output[str(alias)] = dict(options)
    return output


def _selected(values: Sequence[Any], requested: Sequence[Any] | None) -> list[Any]:
    if requested is None:
        return list(values)
    requested_strings = {str(value) for value in requested}
    selected = [value for value in values if str(value) in requested_strings]
    missing = requested_strings - {str(value) for value in selected}
    if missing:
        raise ValueError(f"Unknown reproduction selection: {sorted(missing)}")
    return selected


def expand_experiments(
    config: Mapping[str, Any],
    *,
    datasets: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    list_sizes: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    reproduction = section(config, "reproduction")
    dataset_entries = _matrix_entries(reproduction.get("datasets"), "datasets")
    model_entries = _matrix_entries(reproduction.get("models"), "models")
    method_values = reproduction.get("methods")
    if not isinstance(method_values, Sequence) or isinstance(method_values, str) or not method_values:
        raise ValueError("reproduction.methods must be a non-empty list.")
    seed_values = reproduction.get("seeds", [42])
    size_values = reproduction.get("list_sizes")
    if not isinstance(seed_values, Sequence) or isinstance(seed_values, str) or not seed_values:
        raise ValueError("reproduction.seeds must be a non-empty list.")
    if not isinstance(size_values, Sequence) or isinstance(size_values, str) or not size_values:
        raise ValueError("reproduction.list_sizes must be a non-empty list.")

    selected_datasets = _selected(list(dataset_entries), datasets)
    selected_models = _selected(list(model_entries), models)
    selected_methods = _selected([str(value) for value in method_values], methods)
    selected_seeds = [int(value) for value in _selected([int(value) for value in seed_values], seeds)]
    selected_sizes = [int(value) for value in _selected([int(value) for value in size_values], list_sizes)]
    unknown_methods = set(selected_methods) - set(config.get("methods", {}))
    if unknown_methods:
        raise ValueError(f"Reproduction methods have no configuration: {sorted(unknown_methods)}")

    experiments = []
    for dataset_alias in selected_datasets:
        for model_alias in selected_models:
            for method in selected_methods:
                for list_size in selected_sizes:
                    for seed in selected_seeds:
                        method_config = config.get("methods", {}).get(method, {})
                        backend = resolve_output_backend(method, config.get("ranking", {}), method_config)
                        prompt, prompt_version = output_prompt_identity(method, backend, method_config)
                        identity = {
                            "experiment": config.get("experiment", "paper"),
                            "dataset": dataset_alias,
                            "dataset_config": dataset_entries[dataset_alias],
                            "model": model_alias,
                            "model_config": model_entries[model_alias],
                            "method": method,
                            "backend": backend,
                            "prompt": prompt,
                            "prompt_version": prompt_version,
                            "method_config": method_config,
                            "list_size": list_size,
                            "seed": seed,
                            "training": config.get("training", {}),
                            "ranking": config.get("ranking", {}),
                            "evaluation": config.get("evaluation", {}),
                        }
                        digest = hashlib.sha256(
                            json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
                        ).hexdigest()[:10]
                        run_id = "__".join(
                            [
                                _slug(dataset_alias),
                                _slug(model_alias),
                                _slug(method),
                                _slug(backend),
                                _slug(prompt_version),
                                f"n{list_size}",
                                f"s{seed}",
                                digest,
                            ]
                        )
                        experiments.append({**identity, "config_hash": digest, "run_id": run_id})
    return experiments


def resolve_experiment_config(config: Mapping[str, Any], experiment: Mapping[str, Any]) -> dict[str, Any]:
    reproduction = section(config, "reproduction")
    output_root = Path(reproduction.get("output_dir", ROOT / "runs" / "paper"))
    data_root = Path(reproduction.get("data_dir", ROOT / "data" / "processed" / "paper"))
    run_dir = output_root / str(experiment["run_id"])
    processed_dir = data_root / _slug(experiment["dataset"]) / f"n{experiment['list_size']}" / f"s{experiment['seed']}"
    model_options = dict(experiment["model_config"])
    model_name = model_options.pop("model_name", None)
    if not model_name:
        raise ValueError(f"Reproduction model '{experiment['model']}' requires model_name.")

    resolved = copy.deepcopy(dict(config))
    resolved.pop("reproduction", None)
    resolved["run"] = {
        key: experiment[key]
        for key in (
            "run_id",
            "config_hash",
            "dataset",
            "model",
            "method",
            "backend",
            "prompt",
            "prompt_version",
            "list_size",
            "seed",
        )
    }
    dataset_override = experiment["dataset_config"].get("data", experiment["dataset_config"])
    resolved["data"] = deep_merge(resolved.get("data", {}), dataset_override)
    resolved["data"].setdefault("dataset", {})["name"] = experiment["dataset_config"].get("name", experiment["dataset"])
    resolved["data"].setdefault("sampling", {})["list_sizes"] = [int(experiment["list_size"])]
    resolved["data"].setdefault("training", {})["seed"] = int(experiment["seed"])
    resolved["data"].setdefault("paths", {})["output_dir"] = str(processed_dir)

    resolved["training"] = deep_merge(resolved.get("training", {}), model_options.get("training", {}))
    resolved["training"].update(
        {
            "model_name": model_name,
            "train_path": str(processed_dir / "train.jsonl"),
            "val_path": str(processed_dir / "val.jsonl"),
            "run_dir": str(run_dir / "train"),
            "seed": int(experiment["seed"]),
        }
    )
    resolved["ranking"] = deep_merge(resolved.get("ranking", {}), model_options.get("ranking", {}))
    resolved["ranking"].update(
        {
            "model_name": model_name,
            "data_path": str(processed_dir / "test.jsonl"),
            "output_dir": str(run_dir / "eval"),
            "ranked_lists_path": str(run_dir / "eval" / "ranked_lists.json"),
            "seed": int(experiment["seed"]),
        }
    )
    resolved["evaluation"] = deep_merge(resolved.get("evaluation", {}), model_options.get("evaluation", {}))
    resolved["evaluation"].update(
        {
            "ranked_lists_path": str(run_dir / "eval" / "ranked_lists.json"),
            "metrics_path": str(run_dir / "eval" / "metrics.json"),
        }
    )
    method = str(experiment["method"])
    method_options = dict(resolved.get("methods", {}).get(method, {}))
    method_options["backend"] = str(experiment["backend"])
    method_options["dataset"] = str(experiment["dataset"])
    method_options["candidate_count"] = int(experiment["list_size"])
    method_options["output_dir"] = str(run_dir / "eval")
    method_options["seed"] = int(experiment["seed"])
    if method in TRAINABLE_METHODS:
        method_options["run_dir"] = str(run_dir / "train")
        method_options["adapter_path"] = str(run_dir / "train" / "checkpoints" / "final")
    if method == "stella":
        method_options["probe_path"] = str(processed_dir / "val.jsonl")
        method_options["transition_matrix_output"] = str(run_dir / "eval" / "transition_matrix.json")
    resolved.setdefault("methods", {})[method] = method_options
    return resolved


def _stage_artifacts(config: Mapping[str, Any], method: str, stage: str) -> list[Path]:
    if stage == "candidates":
        directory = Path(config["data"]["paths"]["output_dir"])
        return [directory / name for name in ("train.jsonl", "val.jsonl", "test.jsonl")]
    if stage == "train":
        return [Path(config["methods"][method]["adapter_path"])]
    if stage == "rank":
        return [Path(config["ranking"]["ranked_lists_path"])]
    if stage == "evaluate":
        return [Path(config["evaluation"]["metrics_path"])]
    raise ValueError(f"Unknown reproduction stage: {stage}")


def _completed(manifest: Mapping[str, Any], config: Mapping[str, Any], method: str, stage: str) -> bool:
    state = manifest.get("stages", {}).get(stage, {})
    return state.get("status") == "completed" and all(path.exists() for path in _stage_artifacts(config, method, stage))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def reproduce(
    config: Mapping[str, Any],
    *,
    datasets: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    list_sizes: Sequence[int] | None = None,
    stages: Sequence[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    requested_stages = list(stages or section(config, "reproduction").get("stages", REPRODUCTION_STAGES))
    unknown_stages = set(requested_stages) - set(REPRODUCTION_STAGES)
    if unknown_stages:
        raise ValueError(f"Unknown reproduction stages: {sorted(unknown_stages)}")
    experiments = expand_experiments(
        config,
        datasets=datasets,
        models=models,
        methods=methods,
        seeds=seeds,
        list_sizes=list_sizes,
    )
    summaries = []
    for experiment in experiments:
        resolved = resolve_experiment_config(config, experiment)
        method = str(experiment["method"])
        run_dir = Path(resolved["ranking"]["output_dir"]).parent
        manifest_path = run_dir / "status.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {"run": resolved["run"], "stages": {}}
        )
        if not dry_run:
            write_yaml(resolved, run_dir / "resolved_config.yaml")
        run_summary = {
            key: experiment[key]
            for key in (
                "run_id",
                "config_hash",
                "dataset",
                "model",
                "method",
                "backend",
                "prompt",
                "prompt_version",
                "list_size",
                "seed",
            )
        }
        run_summary.update(
            {
                "run_dir": str(run_dir),
                "resolved_config_path": str(run_dir / "resolved_config.yaml"),
                "ranked_lists_path": resolved["ranking"]["ranked_lists_path"],
                "metrics_path": resolved["evaluation"]["metrics_path"],
                "stages": {},
            }
        )
        for stage in requested_stages:
            if stage == "train" and method not in TRAINABLE_METHODS:
                run_summary["stages"][stage] = "not_applicable"
                continue
            if not force and _completed(manifest, resolved, method, stage):
                run_summary["stages"][stage] = "skipped"
                continue
            artifacts = _stage_artifacts(resolved, method, stage)
            if stage == "candidates" and not force and all(path.exists() for path in artifacts):
                run_summary["stages"][stage] = "reused"
                if not dry_run:
                    manifest.setdefault("stages", {})[stage] = {
                        "status": "completed",
                        "completed_at": _utc_now(),
                        "reused": True,
                        "artifacts": [str(path) for path in artifacts],
                    }
                    write_json(manifest, manifest_path)
                continue
            if dry_run:
                run_summary["stages"][stage] = "planned"
                continue
            started_at = _utc_now()
            manifest.setdefault("stages", {})[stage] = {"status": "running", "started_at": started_at}
            write_json(manifest, manifest_path)
            try:
                if stage == "candidates":
                    result = run_candidates_stage(resolved).to_dict()
                elif stage == "train":
                    result = run_train_stage(resolved, method=method).to_dict()
                elif stage == "rank":
                    result = run_rank_stage(resolved, method=method).to_dict()
                else:
                    result = run_evaluate_stage(resolved).to_dict()
                manifest["stages"][stage] = {
                    "status": "completed",
                    "started_at": started_at,
                    "completed_at": _utc_now(),
                    "artifacts": [str(path) for path in _stage_artifacts(resolved, method, stage)],
                    "result": result,
                }
                write_json(manifest, manifest_path)
                run_summary["stages"][stage] = "completed"
            except Exception as error:
                manifest["stages"][stage] = {
                    "status": "failed",
                    "started_at": started_at,
                    "failed_at": _utc_now(),
                    "error": f"{type(error).__name__}: {error}",
                }
                write_json(manifest, manifest_path)
                raise
        summaries.append(run_summary)
    output_root = Path(section(config, "reproduction").get("output_dir", ROOT / "runs" / "paper"))
    summary = {"num_runs": len(summaries), "dry_run": dry_run, "runs": summaries}
    if not dry_run:
        comparison = []
        for run_summary in summaries:
            metrics_path = Path(run_summary["metrics_path"])
            if not metrics_path.exists():
                continue
            comparison.append(
                {
                    key: run_summary[key]
                    for key in (
                        "run_id",
                        "config_hash",
                        "dataset",
                        "model",
                        "method",
                        "backend",
                        "prompt",
                        "prompt_version",
                        "list_size",
                        "seed",
                    )
                }
                | json.loads(metrics_path.read_text(encoding="utf-8"))
            )
        comparison_path = output_root / "comparison.json"
        write_json(comparison, comparison_path)
        summary["comparison_path"] = str(comparison_path)
        write_json(summary, output_root / "reproduction_summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run InvariRank paper data and experiments.")
    parser.add_argument("--config", help="Research YAML configuration; each command has a stage-specific default.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    candidates_parser = subparsers.add_parser("candidates", help="Generate candidate-list dataset splits.")
    candidates_parser.add_argument("--output-dir")

    train_parser = subparsers.add_parser("train", help="Train a framework reranker.")
    train_parser.add_argument("--method", default="invarirank")
    train_parser.add_argument("--model")
    train_parser.add_argument("--output-dir")

    rank_parser = subparsers.add_parser("rank", help="Rank a processed candidate set.")
    rank_parser.add_argument("--method", default="invarirank")
    rank_parser.add_argument("--model")
    rank_parser.add_argument("--num-samples", type=int)
    rank_parser.add_argument("--permutations", type=int)
    rank_parser.add_argument("--backend", choices=("generate", "span_logprob"))

    evaluation_parser = subparsers.add_parser("evaluate", help="Evaluate ranked-list records.")
    evaluation_parser.add_argument("--ranked-lists")
    evaluation_parser.add_argument("--output")
    evaluation_parser.add_argument("--top-k", nargs="+", type=int)

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run candidates, train, rank, and evaluate in sequence.",
    )
    pipeline_parser.add_argument("--method", default="invarirank")
    pipeline_parser.add_argument("--dry-run", action="store_true")

    reproduce_parser = subparsers.add_parser("reproduce", help="Execute or resume the paper experiment matrix.")
    reproduce_parser.add_argument("--datasets", nargs="+")
    reproduce_parser.add_argument("--models", nargs="+")
    reproduce_parser.add_argument("--methods", nargs="+")
    reproduce_parser.add_argument("--seeds", nargs="+", type=int)
    reproduce_parser.add_argument("--list-sizes", nargs="+", type=int)
    reproduce_parser.add_argument("--stages", nargs="+", choices=REPRODUCTION_STAGES)
    reproduce_parser.add_argument("--force", action="store_true")
    reproduce_parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = {
        "candidates": DEFAULT_CANDIDATES_CONFIG,
        "train": DEFAULT_TRAIN_CONFIG,
        "rank": DEFAULT_RANK_CONFIG,
        "evaluate": DEFAULT_EVALUATE_CONFIG,
        "reproduce": DEFAULT_CONFIG,
    }
    if args.command == "pipeline":
        if args.config:
            raise ValueError("The pipeline command uses the four default stage configs; --config is not supported.")
        result = run_pipeline(method=args.method, dry_run=args.dry_run)
        print(json.dumps(result, indent=2))
        return
    config = load_config(args.config or defaults[args.command])
    if args.command == "candidates":
        result = run_candidates_stage(config, output_dir=args.output_dir).to_dict()
    elif args.command == "train":
        result = run_train_stage(
            config,
            model_name=args.model,
            output_dir=args.output_dir,
            method=args.method,
        ).to_dict()
    elif args.command == "rank":
        result = run_rank_stage(
            config,
            model_name=args.model,
            method=args.method,
            num_samples=args.num_samples,
            permutations=args.permutations,
            backend=args.backend,
        ).to_dict()
    elif args.command == "evaluate":
        result = run_evaluate_stage(
            config,
            ranked_lists_path=args.ranked_lists,
            output_path=args.output,
            top_k=args.top_k,
        ).to_dict()
    else:
        result = reproduce(
            config,
            datasets=args.datasets,
            models=args.models,
            methods=args.methods,
            seeds=args.seeds,
            list_sizes=args.list_sizes,
            stages=args.stages,
            force=args.force,
            dry_run=args.dry_run,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

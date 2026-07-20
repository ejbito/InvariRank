"""Paper baselines and their shared registry."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from invarirank import (
    FINE_TUNED_METHODS,
    InvariRankReranker,
    RankedItem,
    RankingResult,
    RankingSample,
    Reranker,
    RerankerConfig,
)

from .generation import GeneratedRankingReranker, GeneratedRerankerConfig

DUAL_BACKEND_METHODS = {"zero_shot", "bootstrapping", "sgs", "stella"}
SPAN_ONLY_METHODS = set(FINE_TUNED_METHODS)
OUTPUT_BACKENDS = {"generate", "span_logprob"}
SUPPORTED_METHODS = frozenset(DUAL_BACKEND_METHODS | SPAN_ONLY_METHODS)


def _sample(sample: RankingSample | Mapping[str, Any]) -> RankingSample:
    return sample if isinstance(sample, RankingSample) else RankingSample.from_dict(sample)


def _permutation(permutation: Sequence[int] | None, count: int) -> list[int]:
    result = list(range(count)) if permutation is None else [int(value) for value in permutation]
    if len(result) != count or set(result) != set(range(count)):
        raise ValueError(f"permutation must contain every candidate index from 0 to {count - 1} exactly once.")
    return result


def _with_metadata(result: RankingResult, method: str, forward_passes: int) -> RankingResult:
    return RankingResult(
        user_id=result.user_id,
        items=result.items,
        permutation=result.permutation,
        split=result.split,
        metadata={**result.metadata, "method": method, "forward_passes": forward_passes},
    )


def _combined_backend_metadata(rankings: Sequence[RankingResult]) -> dict[str, Any]:
    if not rankings:
        return {}
    metadata = [dict(result.metadata) for result in rankings]
    combined: dict[str, Any] = {}
    for key in ("output_backend", "prompt_family", "prompt_version"):
        values = {str(value[key]) for value in metadata if value.get(key) is not None}
        if len(values) == 1:
            combined[key] = values.pop()
    generated = [value for value in metadata if value.get("output_backend") == "generate"]
    if generated:
        combined.update(
            {
                "generation_calls": sum(int(value.get("generation_calls", 1)) for value in generated),
                "generation_batches": sum(float(value.get("generation_batches", 1.0)) for value in generated),
                "input_tokens": sum(int(value.get("input_tokens", 0)) for value in generated),
                "generated_tokens": sum(int(value.get("generated_tokens", 0)) for value in generated),
                "latency_seconds": sum(float(value.get("latency_seconds", 0.0)) for value in generated),
                "parse_statuses": [str(value.get("parse_status", "unknown")) for value in generated],
                "repaired_outputs": sum(value.get("parse_status") == "repaired" for value in generated),
                "raw_outputs": [str(value.get("raw_output", "")) for value in generated],
                "unknown_label_count": sum(int(value.get("unknown_label_count", 0)) for value in generated),
                "duplicate_label_count": sum(int(value.get("duplicate_label_count", 0)) for value in generated),
                "missing_label_count": sum(int(value.get("missing_label_count", 0)) for value in generated),
                "generation_config": generated[0].get("generation_config", {}),
            }
        )
    return combined


def _rank_many(
    reranker: Reranker,
    requests: Sequence[tuple[RankingSample, Sequence[int]]],
    *,
    batch_size: int,
    progress_description: str | None = None,
) -> list[RankingResult]:
    batched = getattr(reranker, "rank_many", None)
    if progress_description is None:
        if callable(batched) and batch_size > 1:
            return list(batched(requests, batch_size=batch_size))
        return [reranker.rank(sample, permutation=permutation) for sample, permutation in requests]

    from tqdm.auto import tqdm

    results = []
    with tqdm(total=len(requests), desc=progress_description, unit="ranking", dynamic_ncols=True) as progress:
        if callable(batched) and batch_size > 1:
            for start in range(0, len(requests), batch_size):
                chunk = requests[start : start + batch_size]
                results.extend(batched(chunk, batch_size=batch_size))
                progress.update(len(chunk))
        else:
            for sample, permutation in requests:
                results.append(reranker.rank(sample, permutation=permutation))
                progress.update()
    return results


def borda_aggregate(
    sample: RankingSample,
    rankings: Sequence[RankingResult],
    input_permutation: Sequence[int],
    *,
    method: str,
    forward_passes: int,
) -> RankingResult:
    if not rankings:
        raise ValueError("Borda aggregation requires at least one ranking.")
    count = len(sample.candidates)
    points = {index: 0.0 for index in range(count)}
    raw_scores = {index: 0.0 for index in range(count)}
    for result in rankings:
        for rank, item in enumerate(result.items):
            points[item.candidate_index] += count - rank
            raw_scores[item.candidate_index] += item.score
    input_positions = {candidate: position for position, candidate in enumerate(input_permutation)}
    order = sorted(
        range(count),
        key=lambda candidate: (
            -points[candidate],
            -raw_scores[candidate],
            input_positions[candidate],
        ),
    )
    items = tuple(
        RankedItem(
            candidate_index=index,
            item_id=_candidate_id(sample.candidates[index], index),
            score=float(points[index]),
            input_position=input_positions[index],
            relevance=_relevance(sample.candidates[index]),
            candidate=dict(sample.candidates[index]),
        )
        for index in order
    )
    return RankingResult(
        user_id=sample.user_id,
        items=items,
        permutation=tuple(input_permutation),
        split=sample.split,
        metadata={
            "method": method,
            "forward_passes": forward_passes,
            "aggregation": "borda",
            "num_rankings": len(rankings),
            **_combined_backend_metadata(rankings),
        },
    )


class DirectMethod(Reranker):
    """One-pass method used for Zero-shot, LFT, InvariRank, and ablations."""

    def __init__(self, reranker: Reranker, *, name: str):
        self.reranker = reranker
        self.name = name

    def rank(
        self,
        sample: RankingSample | Mapping[str, Any],
        *,
        permutation: Sequence[int] | None = None,
    ) -> RankingResult:
        return _with_metadata(self.reranker.rank(sample, permutation=permutation), self.name, 1)


class Bootstrapping(Reranker):
    """Permutation ensembling with Borda-count aggregation."""

    def __init__(self, reranker: Reranker, *, num_samples: int = 3, seed: int = 42):
        if num_samples < 1:
            raise ValueError("num_samples must be at least one.")
        self.reranker = reranker
        self.num_samples = num_samples
        self.seed = seed

    def rank(
        self,
        sample: RankingSample | Mapping[str, Any],
        *,
        permutation: Sequence[int] | None = None,
    ) -> RankingResult:
        ranking_sample = _sample(sample)
        outer = _permutation(permutation, len(ranking_sample.candidates))
        permutations = [outer]
        for sample_index in range(1, self.num_samples):
            shuffled = list(outer)
            random.Random(self.seed + sample_index * 1009).shuffle(shuffled)
            permutations.append(shuffled)
        rankings = [self.reranker.rank(ranking_sample, permutation=value) for value in permutations]
        return borda_aggregate(
            ranking_sample,
            rankings,
            outer,
            method="bootstrapping",
            forward_passes=len(rankings),
        )


class SequentialGreedySelection(Reranker):
    """Greedily select the best remaining candidates over repeated model calls.

    Each pass chooses ``selection_size`` candidates before reranking the remainder.
    """

    def __init__(self, reranker: Reranker, *, selection_size: int = 1):
        if selection_size < 1:
            raise ValueError("selection_size must be at least one.")
        self.reranker = reranker
        self.selection_size = selection_size

    def rank(
        self,
        sample: RankingSample | Mapping[str, Any],
        *,
        permutation: Sequence[int] | None = None,
    ) -> RankingResult:
        ranking_sample = _sample(sample)
        outer = _permutation(permutation, len(ranking_sample.candidates))
        remaining = list(outer)
        selected: list[int] = []
        local_rankings: list[RankingResult] = []
        forward_passes = 0
        while remaining:
            local_sample = RankingSample(
                user_id=ranking_sample.user_id,
                history=ranking_sample.history,
                candidates=[ranking_sample.candidates[index] for index in remaining],
                split=ranking_sample.split,
            )
            local_result = self.reranker.rank(local_sample)
            local_rankings.append(local_result)
            forward_passes += 1
            chosen_local = [item.candidate_index for item in local_result.items[: self.selection_size]]
            chosen_global = [remaining[index] for index in chosen_local]
            selected.extend(chosen_global)
            chosen_set = set(chosen_global)
            remaining = [index for index in remaining if index not in chosen_set]

        input_positions = {candidate: position for position, candidate in enumerate(outer)}
        count = len(selected)
        items = tuple(
            RankedItem(
                candidate_index=index,
                item_id=_candidate_id(ranking_sample.candidates[index], index),
                score=float(count - rank),
                input_position=input_positions[index],
                relevance=_relevance(ranking_sample.candidates[index]),
                candidate=dict(ranking_sample.candidates[index]),
            )
            for rank, index in enumerate(selected)
        )
        return RankingResult(
            user_id=ranking_sample.user_id,
            items=items,
            permutation=tuple(outer),
            split=ranking_sample.split,
            metadata={
                "method": "sgs",
                "forward_passes": forward_passes,
                "selection_size": self.selection_size,
                **_combined_backend_metadata(local_rankings),
            },
        )


class StellaCalibrator:
    """Position transition likelihoods for STELLA Bayesian calibration."""

    def __init__(self, transition_matrix: Any, *, provenance: Mapping[str, Any] | None = None):
        matrix = np.asarray(transition_matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
            raise ValueError("STELLA transition matrix must be a non-empty square matrix.")
        if not np.isfinite(matrix).all() or (matrix < 0).any():
            raise ValueError("STELLA transition matrix must contain finite non-negative values.")
        row_sums = matrix.sum(axis=1, keepdims=True)
        if (row_sums <= 0).any():
            raise ValueError("Every STELLA transition-matrix row must have positive mass.")
        self.transition_matrix = matrix / row_sums
        self.provenance = dict(provenance or {})

    @property
    def size(self) -> int:
        return int(self.transition_matrix.shape[0])

    @classmethod
    def fit(
        cls,
        observations: Iterable[tuple[int, int]],
        *,
        size: int,
        smoothing: float = 1.0,
        provenance: Mapping[str, Any] | None = None,
    ) -> StellaCalibrator:
        if size < 1 or smoothing < 0:
            raise ValueError("size must be positive and smoothing must be non-negative.")
        counts = np.full((size, size), float(smoothing), dtype=np.float64)
        for true_position, predicted_position in observations:
            if not 0 <= true_position < size or not 0 <= predicted_position < size:
                raise ValueError("STELLA observation position is outside the matrix.")
            counts[true_position, predicted_position] += 1.0
        return cls(counts, provenance=provenance)

    @classmethod
    def load(cls, path: str | Path) -> StellaCalibrator:
        path = Path(path)
        if path.suffix.lower() == ".npy":
            return cls(np.load(path))
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, dict):
            return cls(value.get("transition_matrix", value), provenance=value.get("provenance"))
        return cls(value)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(
                {"transition_matrix": self.transition_matrix.tolist(), "provenance": self.provenance},
                handle,
                indent=2,
            )

    def validate_provenance(self, expected: Mapping[str, Any]) -> None:
        mismatches = {
            key: (
                self.provenance.get(key, False) if key == "top_one_generation" else self.provenance.get(key),
                value,
            )
            for key, value in expected.items()
            if value is not None
            and (self.provenance.get(key, False) if key == "top_one_generation" else self.provenance.get(key)) != value
        }
        if mismatches:
            details = ", ".join(
                f"{key}: matrix={observed!r}, requested={requested!r}"
                for key, (observed, requested) in sorted(mismatches.items())
            )
            raise ValueError(f"STELLA transition matrix provenance mismatch ({details}). Re-run probing.")

    def update(self, prior: Any, predicted_position: int) -> np.ndarray:
        prior_array = np.asarray(prior, dtype=np.float64)
        if prior_array.shape != (self.size,):
            raise ValueError(f"STELLA prior must have shape ({self.size},).")
        if not 0 <= predicted_position < self.size:
            raise ValueError("predicted_position is outside the transition matrix.")
        posterior = prior_array * self.transition_matrix[:, predicted_position]
        total = posterior.sum()
        return np.full(self.size, 1.0 / self.size) if total <= 0 else posterior / total


class Stella(Reranker):
    """Bayesian position calibration with low-entropy Borda aggregation."""

    def __init__(
        self,
        reranker: Reranker,
        calibrator: StellaCalibrator,
        *,
        max_updates: int = 10,
        aggregate_count: int = 3,
        seed: int = 42,
        convergence_tolerance: float = 1e-6,
        convergence_steps: int = 3,
        batch_size: int = 1,
    ):
        if max_updates < 1 or aggregate_count < 1 or convergence_steps < 1 or batch_size < 1:
            raise ValueError("STELLA update, aggregation, and convergence counts must be positive.")
        if aggregate_count > max_updates:
            raise ValueError("STELLA aggregate_count cannot exceed max_updates.")
        self.reranker = reranker
        self.calibrator = calibrator
        self.max_updates = max_updates
        self.aggregate_count = aggregate_count
        self.seed = seed
        self.convergence_tolerance = convergence_tolerance
        self.convergence_steps = convergence_steps
        self.batch_size = batch_size

    def rank(
        self,
        sample: RankingSample | Mapping[str, Any],
        *,
        permutation: Sequence[int] | None = None,
    ) -> RankingResult:
        ranking_sample = _sample(sample)
        count = len(ranking_sample.candidates)
        if count != self.calibrator.size:
            raise ValueError(f"STELLA matrix size {self.calibrator.size} does not match candidate count {count}.")
        outer = _permutation(permutation, count)
        candidate_prior = np.full(count, 1.0 / count)
        entropy_rankings: list[tuple[float, RankingResult]] = []
        raw_rankings: list[RankingResult] = []
        previous_entropy = None
        stable_steps = 0
        update_permutations = []
        for update_index in range(self.max_updates):
            current = list(outer)
            if update_index:
                random.Random(self.seed + update_index * 1009).shuffle(current)
            update_permutations.append(current)
        generated_rankings = _rank_many(
            self.reranker,
            [(ranking_sample, current) for current in update_permutations],
            batch_size=self.batch_size,
        )
        for current, raw_result in zip(update_permutations, generated_rankings):
            raw_rankings.append(raw_result)
            predicted_candidate = raw_result.items[0].candidate_index
            predicted_position = current.index(predicted_candidate)
            position_prior = np.asarray([candidate_prior[index] for index in current])
            position_posterior = self.calibrator.update(position_prior, predicted_position)
            for position, candidate_index in enumerate(current):
                candidate_prior[candidate_index] = position_posterior[position]
            entropy = float(-sum(value * math.log(value) for value in candidate_prior if value > 0))
            posterior_order = sorted(range(count), key=lambda index: (-candidate_prior[index], outer.index(index)))
            posterior_result = _probability_result(
                ranking_sample,
                outer,
                posterior_order,
                candidate_prior,
                entropy,
            )
            entropy_rankings.append((entropy, posterior_result))
            if previous_entropy is not None and abs(previous_entropy - entropy) <= self.convergence_tolerance:
                stable_steps += 1
            else:
                stable_steps = 0
            previous_entropy = entropy
            if stable_steps >= self.convergence_steps and len(entropy_rankings) >= self.aggregate_count:
                break

        selected = sorted(entropy_rankings, key=lambda value: value[0])[: self.aggregate_count]
        output = borda_aggregate(
            ranking_sample,
            [result for _, result in selected],
            outer,
            method="stella",
            forward_passes=len(entropy_rankings),
        )
        return RankingResult(
            user_id=output.user_id,
            items=output.items,
            permutation=output.permutation,
            split=output.split,
            metadata={
                **output.metadata,
                "bayesian_updates": len(entropy_rankings),
                "selected_entropies": [entropy for entropy, _ in selected],
                "aggregate_count": len(selected),
                **_combined_backend_metadata(raw_rankings),
            },
        )


def fit_stella_calibrator(
    reranker: Reranker,
    samples: Sequence[RankingSample | Mapping[str, Any]],
    *,
    ensemble_steps: int = 5,
    max_samples: int | None = None,
    smoothing: float = 1.0,
    seed: int = 42,
    provenance: Mapping[str, Any] | None = None,
    batch_size: int = 1,
) -> StellaCalibrator:
    if not samples:
        raise ValueError("STELLA calibration requires probing samples.")
    if ensemble_steps < 1:
        raise ValueError("ensemble_steps must be at least one.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least one.")
    selected_samples = list(samples[:max_samples] if max_samples is not None else samples)
    first = _sample(selected_samples[0])
    size = len(first.candidates)
    observations = []
    probe_requests: list[tuple[RankingSample, list[int]]] = []
    true_positions: list[int] = []
    for sample_index, raw_sample in enumerate(selected_samples):
        ranking_sample = _sample(raw_sample)
        if len(ranking_sample.candidates) != size:
            raise ValueError("STELLA probing samples must have a fixed candidate count.")
        relevance = [int(candidate.get("relevance", 0)) for candidate in ranking_sample.candidates]
        if max(relevance) <= 0:
            raise ValueError("Every STELLA probing sample must contain a relevant candidate.")
        ground_truth = max(range(size), key=lambda index: (relevance[index], -index))
        for true_position in range(size):
            for ensemble_index in range(ensemble_steps):
                others = [index for index in range(size) if index != ground_truth]
                random.Random(seed + sample_index * 100_003 + true_position * 1009 + ensemble_index).shuffle(others)
                permutation = others[:true_position] + [ground_truth] + others[true_position:]
                probe_requests.append((ranking_sample, permutation))
                true_positions.append(true_position)
    probe_results = _rank_many(
        reranker,
        probe_requests,
        batch_size=batch_size,
        progress_description="[STELLA] Calibration",
    )
    for true_position, (_, permutation), result in zip(true_positions, probe_requests, probe_results):
        predicted_position = permutation.index(result.items[0].candidate_index)
        observations.append((true_position, predicted_position))
    resolved_provenance = dict(provenance or {})
    resolved_provenance.setdefault("candidate_count", size)
    return StellaCalibrator.fit(
        observations,
        size=size,
        smoothing=smoothing,
        provenance=resolved_provenance,
    )


def load_backbone_method(
    name: str,
    model_name: str,
    values: Mapping[str, Any],
    options: Mapping[str, Any] | None = None,
) -> Reranker:
    if name not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method: {name}. Expected one of {sorted(SUPPORTED_METHODS)}")
    options = dict(options or {})
    backend = resolve_output_backend(name, values, options)
    calibrator = None
    if name == "stella":
        matrix_path = options.get("transition_matrix_path")
        if not matrix_path:
            raise ValueError("STELLA requires transition_matrix_path or explicit probing calibration.")
        calibrator = StellaCalibrator.load(matrix_path)
        calibrator.validate_provenance(stella_provenance(name, model_name, values, options, backend=backend))
    base = load_ranking_backend(name, model_name, values, options, backend=backend)
    if name in {"zero_shot", "lft", "invarirank"}:
        return DirectMethod(base, name=name)
    if name == "bootstrapping":
        return Bootstrapping(base, num_samples=int(options.get("num_samples", 3)), seed=int(options.get("seed", 42)))
    if name == "sgs":
        return SequentialGreedySelection(base, selection_size=int(options.get("selection_size", 1)))
    if name == "stella":
        assert calibrator is not None
        return Stella(
            base,
            calibrator,
            max_updates=int(options.get("max_updates", 10)),
            aggregate_count=int(options.get("aggregate_count", 3)),
            seed=int(options.get("seed", 42)),
            batch_size=int(options.get("batch_size", 1)),
        )
    raise AssertionError("unreachable")


def resolve_output_backend(name: str, values: Mapping[str, Any], options: Mapping[str, Any]) -> str:
    backend = str(options.get("backend", values.get("backend", "span_logprob")))
    if backend not in OUTPUT_BACKENDS:
        raise ValueError(f"Unsupported output backend: {backend}. Expected one of {sorted(OUTPUT_BACKENDS)}")
    if name in SPAN_ONLY_METHODS and backend != "span_logprob":
        raise ValueError(f"Method '{name}' only supports backend='span_logprob'.")
    if name not in DUAL_BACKEND_METHODS | SPAN_ONLY_METHODS:
        raise ValueError(f"Unsupported method: {name}. Expected one of {sorted(SUPPORTED_METHODS)}")
    return backend


def load_ranking_backend(
    name: str,
    model_name: str,
    values: Mapping[str, Any],
    options: Mapping[str, Any] | None = None,
    *,
    backend: str | None = None,
) -> Reranker:
    options = dict(options or {})
    resolved_backend = backend or resolve_output_backend(name, values, options)
    if resolved_backend == "generate":
        if options.get("prompt", "rankgpt") != "rankgpt":
            raise ValueError("Generated research methods only support the RankGPT prompt.")
        generation_values = dict(values.get("generation", {}))
        generation_values.update(options.get("generation", {}))
        generation_values.update(
            {
                "output_count": int(options.get("selection_size", 1)) if name == "sgs" else None,
                "max_length": int(values.get("max_length", values.get("max_seq_length", 4096))),
                "seed": int(options.get("seed", values.get("seed", 42))),
                "batch_size": int(options.get("batch_size", 1)),
                "top_one_generation": name == "stella" and bool(options.get("top_one_generation", False)),
            }
        )
        return GeneratedRankingReranker.from_pretrained(
            model_name,
            config=GeneratedRerankerConfig.from_mapping(generation_values),
            adapter_path=options.get("adapter_path"),
            device=str(values.get("device", "cuda")),
            dtype=str(values.get("dtype", "bfloat16")),
            trust_remote_code=bool(values.get("trust_remote_code", False)),
        )

    config_values = dict(values)
    architecture = {
        "zero_shot": ("causal", "standard"),
        "bootstrapping": ("causal", "standard"),
        "sgs": ("causal", "standard"),
        "stella": ("causal", "standard"),
    }
    if name in SPAN_ONLY_METHODS:
        reranker_config = RerankerConfig.for_method(name, config_values)
    else:
        config_values["attention_mask"], config_values["position_ids"] = architecture[name]
        config_values["prompt_template"] = "invarirank"
        reranker_config = RerankerConfig.from_mapping(config_values)
    adapter_path = options.get("adapter_path")
    if name.startswith("invarirank") and adapter_path is None:
        adapter_path = values.get("adapter_path")
    return InvariRankReranker.from_pretrained(
        model_name,
        config=reranker_config,
        adapter_path=adapter_path,
    )


def stella_provenance(
    name: str,
    model_name: str,
    values: Mapping[str, Any],
    options: Mapping[str, Any],
    *,
    backend: str,
    candidate_count: int | None = None,
) -> dict[str, Any]:
    prompt, prompt_version = output_prompt_identity(name, backend, options)
    dataset = options.get("dataset", values.get("dataset"))
    provenance = {
        "model_name": model_name,
        "output_backend": backend,
        "prompt_family": prompt,
        "prompt_version": prompt_version,
        "dataset": dataset,
        "candidate_count": candidate_count or options.get("candidate_count"),
        "top_one_generation": bool(options.get("top_one_generation", False)) if backend == "generate" else None,
    }
    return {key: value for key, value in provenance.items() if value is not None}


def output_prompt_identity(name: str, backend: str, options: Mapping[str, Any]) -> tuple[str, str]:
    if backend == "span_logprob":
        return "invarirank_marker", "invarirank-marker-v1"
    if name not in DUAL_BACKEND_METHODS:
        raise ValueError(f"Method '{name}' does not support generated output.")
    prompt = "rankgpt"
    version = "rankgpt-json-v1"
    if (name == "stella" and bool(options.get("top_one_generation", False))) or (
        name == "sgs" and int(options.get("selection_size", 1)) == 1
    ):
        version = "rankgpt-top1-json-v1"
    elif name == "sgs":
        version = "rankgpt-topk-json-v1"
    return prompt, version


def _probability_result(
    sample: RankingSample,
    input_permutation: Sequence[int],
    order: Sequence[int],
    probabilities: Any,
    entropy: float,
) -> RankingResult:
    input_positions = {candidate: position for position, candidate in enumerate(input_permutation)}
    return RankingResult(
        user_id=sample.user_id,
        items=tuple(
            RankedItem(
                candidate_index=index,
                item_id=_candidate_id(sample.candidates[index], index),
                score=float(probabilities[index]),
                input_position=input_positions[index],
                relevance=_relevance(sample.candidates[index]),
                candidate=dict(sample.candidates[index]),
            )
            for index in order
        ),
        permutation=tuple(input_permutation),
        split=sample.split,
        metadata={"entropy": entropy},
    )


def _candidate_id(candidate: Mapping[str, Any], fallback: int) -> str:
    for key in ("item_id", "id", "asin", "movie_id"):
        if key in candidate:
            return str(candidate[key])
    return str(fallback)


def _relevance(candidate: Mapping[str, Any]) -> int | None:
    value = candidate.get("relevance")
    return None if value is None else int(value)


__all__ = [
    "DUAL_BACKEND_METHODS",
    "SUPPORTED_METHODS",
    "OUTPUT_BACKENDS",
    "SPAN_ONLY_METHODS",
    "Bootstrapping",
    "DirectMethod",
    "SequentialGreedySelection",
    "Stella",
    "StellaCalibrator",
    "borda_aggregate",
    "fit_stella_calibrator",
    "load_backbone_method",
    "load_ranking_backend",
    "output_prompt_identity",
    "resolve_output_backend",
    "stella_provenance",
]

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from invarirank.contracts import RankingResult, RankingSample, Reranker

from experiments.reranking.methods.common import (
    combined_metadata,
    normalize_method_requests,
    permutation,
    rank_many,
    ranking_from_order,
    rebase_result,
    replace_metadata,
    request_seed,
    sample,
)


class StellaCalibrator:
    """Position transition likelihoods used by STELLA."""

    REQUIRED_PROVENANCE_FIELDS = (
        "model_name_or_path",
        "scoring",
        "prompt",
        "architecture",
        "adapter_path",
        "candidate_count",
    )

    def __init__(
        self,
        transition_matrix: Any,
        *,
        diagnostics: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ):
        matrix = np.asarray(transition_matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
            raise ValueError("STELLA transition matrix must be a non-empty square matrix.")
        if not np.isfinite(matrix).all() or (matrix < 0).any():
            raise ValueError("STELLA transition matrix must contain finite non-negative values.")
        row_sums = matrix.sum(axis=1, keepdims=True)
        if (row_sums <= 0).any():
            raise ValueError("Every STELLA transition-matrix row must have positive mass.")
        self.transition_matrix = matrix / row_sums
        self.diagnostics = {**_transition_matrix_diagnostics(self.transition_matrix), **dict(diagnostics or {})}
        self.provenance = dict(provenance or {})
        provenance_count = self.provenance.get("candidate_count")
        if provenance_count is not None and int(provenance_count) != self.size:
            raise ValueError(
                "STELLA calibration provenance candidate_count "
                f"{provenance_count} does not match matrix size {self.size}."
            )

    @property
    def size(self) -> int:
        return int(self.transition_matrix.shape[0])

    @classmethod
    def load(cls, path: str | Path) -> StellaCalibrator:
        source = Path(path)
        if source.suffix.lower() == ".npy":
            return cls(np.load(source))
        value = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(value, Mapping):
            return cls(
                value.get("transition_matrix", value.get("matrix")),
                diagnostics=value.get("diagnostics"),
                provenance=value.get("provenance"),
            )
        return cls(value)

    def validate_compatibility(self, expected: Mapping[str, Any]) -> None:
        missing = [field for field in self.REQUIRED_PROVENANCE_FIELDS if field not in self.provenance]
        if missing:
            raise ValueError(
                "STELLA transition matrix is missing calibration provenance "
                f"({', '.join(missing)}). Recalibrate it with calibrate_stella before inference."
            )

        mismatches = []
        for field in self.REQUIRED_PROVENANCE_FIELDS:
            expected_value = _normalized_provenance_value(field, expected.get(field))
            actual_value = _normalized_provenance_value(field, self.provenance.get(field))
            if actual_value != expected_value:
                mismatches.append(f"{field}: calibrated={actual_value!r}, inference={expected_value!r}")
        if mismatches:
            raise ValueError(
                "STELLA transition matrix is incompatible with this inference run: "
                + "; ".join(mismatches)
                + ". Recalibrate with the same model, adapter, scorer, prompt, architecture, and candidate count."
            )

    def update(self, prior: Sequence[float], predicted_position: int) -> np.ndarray:
        prior_array = np.asarray(prior, dtype=np.float64)
        if prior_array.shape != (self.size,):
            raise ValueError(f"Expected prior with shape ({self.size},), got {prior_array.shape}.")
        if not 0 <= predicted_position < self.size:
            raise ValueError("predicted_position is outside the transition matrix.")
        likelihood = self.transition_matrix[:, predicted_position]
        posterior = prior_array * likelihood
        total = posterior.sum()
        if total <= 0 or not np.isfinite(total):
            return np.full(self.size, 1.0 / self.size)
        return posterior / total


class Stella(Reranker):
    """Bayesian position calibration over repeated scorer calls."""

    def __init__(
        self,
        scorer: Reranker,
        calibrator: StellaCalibrator,
        *,
        max_updates: int = 10,
        seed: int = 42,
        convergence_tolerance: float = 1e-6,
        convergence_steps: int = 3,
        minimum_information_gain: float = 1e-6,
    ):
        if max_updates < 1:
            raise ValueError("max_updates must be at least one.")
        if convergence_steps < 1:
            raise ValueError("convergence_steps must be at least one.")
        self.scorer = scorer
        self.calibrator = calibrator
        self.max_updates = int(max_updates)
        self.seed = int(seed)
        self.convergence_tolerance = float(convergence_tolerance)
        self.convergence_steps = int(convergence_steps)
        self.minimum_information_gain = float(minimum_information_gain)

    def rank(
        self,
        sample_value: RankingSample | Mapping[str, Any],
        *,
        permutation: Sequence[int] | None = None,
    ) -> RankingResult:
        return self.rank_many([(sample_value, permutation)], batch_size=1)[0]

    def rank_many(
        self,
        samples: Sequence[
            RankingSample | Mapping[str, Any] | tuple[RankingSample | Mapping[str, Any], Sequence[int] | None]
        ],
        *,
        permutations: Sequence[Sequence[int] | None] | None = None,
        batch_size: int = 8,
    ) -> list[RankingResult]:
        states = []
        for sample_value, input_permutation in normalize_method_requests(samples, permutations):
            ranking_sample = sample(sample_value)
            count = len(ranking_sample.candidates)
            if count != self.calibrator.size:
                raise ValueError(f"STELLA matrix size {self.calibrator.size} does not match candidate count {count}.")
            outer = permutation(input_permutation, count)
            states.append(
                {
                    "sample": ranking_sample,
                    "outer": outer,
                    "request_seed": request_seed(self.seed, ranking_sample, outer),
                    "prior": np.full(count, 1.0 / count),
                    "raw_rankings": [],
                    "records": [],
                    "previous_entropy": None,
                    "stable_steps": 0,
                    "active": True,
                }
            )

        for update_index in range(self.max_updates):
            active = [index for index, state in enumerate(states) if state["active"]]
            if not active:
                break
            batch_requests = []
            currents = []
            for state_index in active:
                state = states[state_index]
                current = list(state["outer"])
                if update_index:
                    random.Random(state["request_seed"] + update_index * 1009).shuffle(current)
                currents.append(current)
                batch_requests.append((state["sample"], current))
            raw_results = rank_many(self.scorer, batch_requests, batch_size=batch_size)
            for state_index, current, raw_result in zip(active, currents, raw_results, strict=True):
                self._update_state(states[state_index], current, raw_result, update_index)

        return [self._finalize_state(state) for state in states]

    def _update_state(
        self,
        state: dict[str, Any],
        current: Sequence[int],
        raw_result: RankingResult,
        update_index: int,
    ) -> None:
        prior = state["prior"]
        state["raw_rankings"].append(raw_result)
        predicted_candidate = raw_result.items[0].candidate_index
        predicted_position = list(current).index(predicted_candidate)
        position_prior = np.asarray([prior[index] for index in current])
        position_posterior = self.calibrator.update(position_prior, predicted_position)
        for position, candidate_index in enumerate(current):
            prior[candidate_index] = position_posterior[position]

        entropy = _entropy(prior)
        information_gain = math.log(len(prior)) - entropy
        raw_positions = {item.candidate_index: rank for rank, item in enumerate(raw_result.items)}
        order = sorted(
            range(len(prior)),
            key=lambda index: (-prior[index], raw_positions[index], state["outer"].index(index)),
        )
        posterior = ranking_from_order(
            state["sample"],
            state["outer"],
            order,
            scores={index: float(prior[index]) for index in range(len(prior))},
            metadata={"entropy": entropy},
        )
        state["records"].append((entropy, update_index, posterior, raw_result, information_gain))

        previous = state["previous_entropy"]
        state["stable_steps"] = (
            state["stable_steps"] + 1
            if previous is not None and abs(previous - entropy) <= self.convergence_tolerance
            else 0
        )
        state["previous_entropy"] = entropy
        if state["stable_steps"] >= self.convergence_steps:
            state["active"] = False

    def _finalize_state(self, state: dict[str, Any]) -> RankingResult:
        entropy, update_index, posterior, raw_result, information_gain = min(
            state["records"],
            key=lambda value: (value[0], value[1]),
        )
        raw_rankings = state["raw_rankings"]
        use_fallback = information_gain <= self.minimum_information_gain
        output = rebase_result(state["sample"], raw_result, state["outer"]) if use_fallback else posterior
        return replace_metadata(
            output,
            {
                **output.metadata,
                "method": "stella",
                "forward_passes": len(raw_rankings),
                "bayesian_updates": len(raw_rankings),
                "aggregation": "minimum_entropy_posterior",
                "selected_entropy": entropy,
                "selected_update": update_index,
                "posterior_information_gain": information_gain,
                "posterior_fallback": use_fallback,
                "fallback_reason": "uninformative_posterior" if use_fallback else None,
                "seed": self.seed,
                "request_seed": state["request_seed"],
                "transition_diagnostics": dict(self.calibrator.diagnostics),
                **combined_metadata(raw_rankings),
            },
        )


def _transition_matrix_diagnostics(matrix: np.ndarray) -> dict[str, Any]:
    size = int(matrix.shape[0])
    row_entropies = [float(-sum(value * math.log(value) for value in row if value > 0)) for row in matrix]
    maximum_entropy = math.log(size) if size > 1 else 0.0
    return {
        "row_entropies": row_entropies,
        "mean_row_entropy": float(np.mean(row_entropies)),
        "mean_normalized_row_entropy": float(np.mean(row_entropies) / maximum_entropy) if maximum_entropy else 0.0,
        "minimum_probability": float(matrix.min()),
        "maximum_probability": float(matrix.max()),
    }


def _entropy(values: Sequence[float]) -> float:
    return float(-sum(value * math.log(value) for value in values if value > 0))


def _normalized_provenance_value(field: str, value: Any) -> Any:
    if field == "candidate_count":
        return None if value is None else int(value)
    if field == "adapter_path":
        return None if value is None or not str(value).strip() else str(Path(value).resolve())
    if field == "model_name_or_path" and value is not None and Path(str(value)).exists():
        return str(Path(str(value)).resolve())
    return None if value is None else str(value)

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from invarirank.contracts import Reranker

from experiments.reranking.base import BaseReranker
from experiments.reranking.methods.bootstrapping import Bootstrapping
from experiments.reranking.methods.common import candidate_export_result, ranking_sample
from experiments.reranking.methods.sgs import StochasticGreedySelection
from experiments.reranking.methods.stella import Stella, StellaCalibrator
from experiments.reranking.methods.zero_shot import ZeroShot
from experiments.reranking.scoring import ScoringConfig, load_scorer

METHODS = ("zero_shot", "bootstrapping", "stella", "sgs")
METHOD_SET = frozenset(METHODS)


class LLMReranker(BaseReranker):
    """Run the configured LLM reranking method on exported candidate records."""

    def __init__(
        self,
        method: str = "zero_shot",
        scoring: str = "marker_logprob",
        prompt: str = "marker",
        model_name_or_path: str = "meta-llama/Llama-3.2-3B-Instruct",
        adapter_path: str | None = None,
        transition_matrix_path: str | None = None,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        max_length: int = 4096,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        num_beams: int = 1,
        max_history_items: int = 20,
        parser_repair: str = "append_input_order",
        num_samples: int = 3,
        selection_size: int = 1,
        seed: int = 42,
        max_updates: int = 10,
        convergence_tolerance: float = 1e-6,
        convergence_steps: int = 3,
        minimum_information_gain: float = 1e-6,
        architecture: str = "lft",
    ) -> None:
        if method not in METHOD_SET:
            raise ValueError(f"Unknown reranking method '{method}'. Valid methods: {list(METHODS)}")
        if method == "stella" and not transition_matrix_path:
            raise ValueError("STELLA requires transition_matrix_path.")

        self.method = method
        self.scoring = scoring
        self.prompt = prompt
        self.model_name_or_path = model_name_or_path
        self.adapter_path = adapter_path
        self.max_history_items = max_history_items
        self.seed = seed

        scorer = load_scorer(
            ScoringConfig(
                scoring=scoring,
                prompt=prompt,
                model_name_or_path=model_name_or_path,
                adapter_path=adapter_path,
                device=device,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                parser_repair=parser_repair,
                seed=seed,
                output_count=selection_size if method == "sgs" else None,
                architecture=architecture,
            )
        )
        scorer_config = getattr(scorer, "config", None)
        if scorer_config is not None:
            architecture = (
                "invarirank"
                if (scorer_config.attention_mask, scorer_config.position_ids) == ("block", "shared")
                else "lft"
            )
        self.architecture = architecture
        self.runtime_info = {
            "scoring": scoring,
            "prompt": prompt,
            "model_name_or_path": model_name_or_path,
            "adapter_path": adapter_path,
            "architecture": architecture,
        }
        if method == "zero_shot":
            self.reranker: Reranker = ZeroShot(scorer)
        elif method == "bootstrapping":
            self.reranker = Bootstrapping(scorer, num_samples=num_samples, seed=seed)
        elif method == "sgs":
            self.reranker = StochasticGreedySelection(scorer, selection_size=selection_size, seed=seed)
        else:
            assert transition_matrix_path is not None
            calibrator = StellaCalibrator.load(transition_matrix_path)
            calibrator.validate_compatibility(
                {
                    "model_name_or_path": model_name_or_path,
                    "adapter_path": adapter_path,
                    "scoring": scoring,
                    "prompt": prompt,
                    "architecture": architecture,
                    "candidate_count": calibrator.size,
                }
            )
            self.reranker = Stella(
                scorer,
                calibrator,
                max_updates=max_updates,
                seed=seed,
                convergence_tolerance=convergence_tolerance,
                convergence_steps=convergence_steps,
                minimum_information_gain=minimum_information_gain,
            )

    def rerank_user(
        self,
        user_record: Mapping[str, Any],
        user_history: list[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        sample = ranking_sample(user_record, user_history=user_history)
        started = time.perf_counter()
        result = self.reranker.rank(sample)
        elapsed = time.perf_counter() - started
        return candidate_export_result(
            result,
            reranker=self.method,
            model=self.model_name_or_path,
            scoring=self.scoring,
            prompt=self.prompt,
            elapsed_seconds=elapsed,
        )

    def rerank_many_users(
        self,
        requests: Sequence[tuple[Mapping[str, Any], list[Mapping[str, Any]] | None]],
        *,
        batch_size: int = 8,
    ) -> list[dict[str, Any]]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least one.")
        if not requests:
            return []

        samples = [
            (ranking_sample(user_record, user_history=user_history), None)
            for user_record, user_history in requests
        ]
        started = time.perf_counter()
        results = self.reranker.rank_many(samples, batch_size=batch_size)
        elapsed_per_result = (time.perf_counter() - started) / len(results) if results else 0.0
        return [
            candidate_export_result(
                result,
                reranker=self.method,
                model=self.model_name_or_path,
                scoring=self.scoring,
                prompt=self.prompt,
                elapsed_seconds=elapsed_per_result,
            )
            for result in results
        ]

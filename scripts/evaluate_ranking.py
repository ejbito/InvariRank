from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config
from training.metrics import (
    hr_at_k,
    kendall_tau_from_rank_maps,
    ndcg_at_k,
    spearman_rho_from_rank_maps,
    topk_overlap_at_k,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ranked-list records.")
    parser.add_argument("--ranked-lists", help="Path to the ranked_lists.json file.")
    parser.add_argument("--config", help="Optional config file to resolve ranked_lists_path or output_dir.")
    parser.add_argument("--output", help="Optional JSON file to write evaluation results.")
    parser.add_argument(
        "--topks",
        nargs="+",
        type=int,
        default=[5, 10],
        help="Top-k values for HR, nDCG, and top-k agreement.",
    )
    return parser.parse_args()


def resolve_ranked_lists_path(cfg: Any, ranked_lists_path: str | None) -> Path:
    if ranked_lists_path:
        return Path(ranked_lists_path).resolve()
    if getattr(cfg, "ranked_lists_path", None):
        return Path(cfg.ranked_lists_path)
    if getattr(cfg, "output_dir", None):
        return Path(cfg.output_dir) / "ranked_lists.json"
    raise ValueError("Config must define ranked_lists_path or output_dir when --ranked-lists is not provided.")


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        rows: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Ranked lists file must contain a JSON list or JSONL records: {path}")


def build_score_vector(
    candidate_indices: list[int], output_candidates: list[int], output_scores: list[float]
) -> list[float]:
    pos_by_candidate = {cand: idx for idx, cand in enumerate(candidate_indices)}
    scores = [float("-inf")] * len(candidate_indices)
    for score, cand in zip(output_scores, output_candidates):
        if cand in pos_by_candidate:
            scores[pos_by_candidate[cand]] = float(score)
    return scores


def rank_map(order: list[Any]) -> dict[Any, int]:
    return {item: rank for rank, item in enumerate(order)}


def mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def evaluate_permutation_effectiveness(perm: dict[str, Any], topks: list[int]) -> dict[str, float]:
    input_data = perm.get("input", {})
    output_data = perm.get("output_ranking", {})

    candidate_indices = [int(x) for x in input_data.get("candidate_indices", [])]
    relevance = [int(x) for x in input_data.get("relevance", [])]
    output_candidates = [int(x) for x in output_data.get("candidate_indices", [])]
    output_scores = [float(x) for x in output_data.get("scores", [])]

    if not candidate_indices or not relevance or not output_candidates:
        return {}

    scores = build_score_vector(candidate_indices, output_candidates, output_scores)
    results: dict[str, float] = {}
    for k in topks:
        results[f"hr@{k}"] = hr_at_k(scores, relevance, k)
        results[f"ndcg@{k}"] = ndcg_at_k(scores, relevance, k)

    return results


def evaluate_record_robustness(record: dict[str, Any], topks: list[int]) -> dict[str, float]:
    output_maps: list[dict[int, int]] = []
    output_orders: list[list[int]] = []

    for perm in record.get("permutations", []):
        output_candidates = [int(x) for x in perm.get("output_ranking", {}).get("candidate_indices", [])]
        if output_candidates:
            output_orders.append(output_candidates)
            output_maps.append(rank_map(output_candidates))

    if len(output_maps) < 2:
        return {}

    kendalls: list[float] = []
    spearmans: list[float] = []
    topk_values: dict[int, list[float]] = {k: [] for k in topks}

    for (map_a, order_a), (map_b, order_b) in combinations(zip(output_maps, output_orders), 2):
        spearman = spearman_rho_from_rank_maps(map_a, map_b)
        if spearman is not None:
            spearmans.append(spearman)

        kendall = kendall_tau_from_rank_maps(map_a, map_b)
        if kendall is not None:
            kendalls.append(kendall)

        for k in topks:
            topk_values[k].append(topk_overlap_at_k(order_a, order_b, k))

    results: dict[str, float] = {}
    robustness_spearman = mean(spearmans)
    if robustness_spearman is not None:
        results["permutation_spearman"] = robustness_spearman

    robustness_kendall = mean(kendalls)
    if robustness_kendall is not None:
        results["permutation_kendall"] = robustness_kendall

    for k, values in topk_values.items():
        value = mean(values)
        if value is not None:
            results[f"permutation_topk_overlap@{k}"] = value

    return results


def aggregate_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    aggregated: dict[str, list[float]] = {}
    for metrics in metrics_list:
        for key, value in metrics.items():
            aggregated.setdefault(key, []).append(value)

    return {key: sum(values) / len(values) for key, values in aggregated.items() if values}


def load_ranked_lists(path: Path) -> list[dict[str, Any]]:
    return load_json_or_jsonl(path)


def main() -> None:
    args = parse_args()
    if not args.config and not args.ranked_lists:
        raise ValueError("Provide --config or --ranked-lists.")

    cfg = load_config(args.config) if args.config else None
    ranked_lists_path = resolve_ranked_lists_path(cfg, args.ranked_lists) if cfg else Path(args.ranked_lists).resolve()
    ranked_lists = load_ranked_lists(ranked_lists_path)

    effectiveness_metrics: list[dict[str, float]] = []
    robustness_metrics: list[dict[str, float]] = []
    total_permutations = 0

    for record in ranked_lists:
        for perm in record.get("permutations", []):
            results = evaluate_permutation_effectiveness(perm, args.topks)
            if results:
                effectiveness_metrics.append(results)
                total_permutations += 1

        robustness = evaluate_record_robustness(record, args.topks)
        if robustness:
            robustness_metrics.append(robustness)

    output = {
        "ranked_lists_path": str(ranked_lists_path),
        "num_records": len(ranked_lists),
        "num_permutations": total_permutations,
        "effectiveness": aggregate_metrics(effectiveness_metrics),
        "robustness": aggregate_metrics(robustness_metrics),
    }

    print(json.dumps(output, indent=2))
    if args.output:
        with Path(args.output).open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Wrote evaluation results to {args.output}")


if __name__ == "__main__":
    main()

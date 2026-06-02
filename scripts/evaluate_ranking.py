from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config
from training.metrics import (
    hr_at_k,
    ndcg_at_k,
    spearman_rho_from_rank_maps,
    kendall_tau_from_rank_maps,
    topk_overlap_at_k,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ranked lists using ranking metrics.")
    parser.add_argument("--ranked-lists", help="Path to the ranked lists JSON file.")
    parser.add_argument("--config", help="Optional config file to resolve ranked_lists_path or output_dir.")
    parser.add_argument("--output", help="Optional JSON file to write evaluation results.")
    parser.add_argument("--topks", nargs="+", type=int, default=[5, 10], help="Top-k values for HR, nDCG, and top-k overlap.")
    return parser.parse_args()


def resolve_ranked_lists_path(cfg: Any, ranked_lists_path: str | None) -> Path:
    if ranked_lists_path:
        return Path(ranked_lists_path).resolve()
    if getattr(cfg, "ranked_lists_path", None):
        return Path(cfg.ranked_lists_path)
    if getattr(cfg, "output_dir", None):
        return Path(cfg.output_dir) / "ranked_lists.json"
    raise ValueError("Config must define ranked_lists_path or output_dir when --ranked-lists is not provided.")


def to_float_list(values: Any) -> list[float]:
    if isinstance(values, list):
        return [float(v) for v in values]
    return [float(v) for v in values.detach().cpu().tolist()]


def build_relevance_order(candidate_indices: list[int], relevance: list[int]) -> list[int]:
    relevance_by_candidate = {cand: rel for cand, rel in zip(candidate_indices, relevance)}
    return sorted(candidate_indices, key=lambda cand: (-relevance_by_candidate[cand], cand))


def build_score_vector(candidate_indices: list[int], output_candidates: list[int], output_scores: list[float]) -> list[float]:
    pos_by_candidate = {cand: idx for idx, cand in enumerate(candidate_indices)}
    scores = [float("-inf")] * len(candidate_indices)
    for score, cand in zip(output_scores, output_candidates):
        if cand in pos_by_candidate:
            scores[pos_by_candidate[cand]] = float(score)
    return scores


def item_ids_for_candidates(candidate_indices: list[int], item_ids: list[Any]) -> dict[int, Any]:
    return {cand: iid for cand, iid in zip(candidate_indices, item_ids)}


def evaluate_permutation(perm: dict[str, Any], topks: list[int]) -> dict[str, float]:
    candidate_indices = [int(x) for x in perm["input"]["candidate_indices"]]
    relevance = [int(x) for x in perm["input"]["relevance"]]
    output_candidates = [int(x) for x in perm["output_ranking"]["candidate_indices"]]
    output_scores = [float(x) for x in perm["output_ranking"]["scores"]]
    output_item_ids = perm["output_ranking"].get("item_ids", [])

    if not candidate_indices or not relevance or not output_candidates:
        return {}

    scores = build_score_vector(candidate_indices, output_candidates, output_scores)
    relev_order = build_relevance_order(candidate_indices, relevance)
    score_order = output_candidates
    relevance_rank_map = {cand: rank for rank, cand in enumerate(relev_order)}
    score_rank_map = {cand: rank for rank, cand in enumerate(score_order)}

    item_id_map = item_ids_for_candidates(candidate_indices, perm["input"].get("item_ids", []))
    output_topk_ids = output_item_ids
    truth_topk_ids = [item_id_map[cand] for cand in relev_order if cand in item_id_map]

    results: dict[str, float] = {}
    for k in topks:
        results[f"hr@{k}"] = hr_at_k(scores, relevance, k)
        results[f"ndcg@{k}"] = ndcg_at_k(scores, relevance, k)
        results[f"topk_overlap@{k}"] = topk_overlap_at_k(output_topk_ids, truth_topk_ids, k)

    spearman = spearman_rho_from_rank_maps(relevance_rank_map, score_rank_map)
    if spearman is not None:
        results["spearman"] = spearman

    kendall = kendall_tau_from_rank_maps(relevance_rank_map, score_rank_map)
    if kendall is not None:
        results["kendall"] = kendall

    return results


def aggregate_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    aggregated: dict[str, list[float]] = {}
    for metrics in metrics_list:
        for key, value in metrics.items():
            aggregated.setdefault(key, []).append(value)

    return {key: sum(values) / len(values) for key, values in aggregated.items() if values}


def load_ranked_lists(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Ranked lists file must contain a JSON list: {path}")
    return data


def main() -> None:
    args = parse_args()
    if not args.config and not args.ranked_lists:
        raise ValueError("Provide --config or --ranked-lists.")

    cfg = load_config(args.config) if args.config else None
    ranked_lists_path = resolve_ranked_lists_path(cfg, args.ranked_lists) if cfg else Path(args.ranked_lists).resolve()
    ranked_lists = load_ranked_lists(ranked_lists_path)

    metrics_list: list[dict[str, float]] = []
    total_permutations = 0
    for record in ranked_lists:
        for perm in record.get("permutations", []):
            results = evaluate_permutation(perm, args.topks)
            if results:
                metrics_list.append(results)
                total_permutations += 1

    aggregated = aggregate_metrics(metrics_list)
    output = {
        "ranked_lists_path": str(ranked_lists_path),
        "num_records": len(ranked_lists),
        "num_permutations": total_permutations,
        **aggregated,
    }

    print(json.dumps(output, indent=2))
    if args.output:
        with Path(args.output).open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Wrote evaluation results to {args.output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invarirank.evaluation import evaluate_ranked_list_records
from invarirank.utils import read_json, write_json


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_metrics_summary(metrics: dict) -> None:
    metadata = metrics.get("metadata", {})
    print("\nEvaluation Summary")
    print("=" * 64)
    print(
        "Records: "
        f"{metadata.get('num_records', '-')}, "
        f"rankings/record: {fmt(metadata.get('avg_rankings_per_record'))}"
    )

    sections = [
        ("Effectiveness", metrics.get("effectiveness", {}), ["HR@5", "HR@10", "nDCG@5", "nDCG@10"]),
        (
            "Robustness",
            metrics.get("robustness", {}),
            ["kendall_tau", "spearman_rho", "top5_agreement", "top10_agreement"],
        ),
    ]

    for title, values, keys in sections:
        if not values:
            continue
        print(f"\n{title}")
        print("-" * 64)
        width = max(len(k) for k in keys)
        for key in keys:
            print(f"{key:<{width}}  {fmt(values.get(key))}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ranked-list records.")
    parser.add_argument("--input", required=True, help="Path to ranked_lists.json.")
    parser.add_argument("--output", required=True, help="Path to write metrics JSON.")
    parser.add_argument("--max-permutations", type=int, default=None)
    args = parser.parse_args()
    records = read_json(args.input)
    metrics = evaluate_ranked_list_records(
        records,
        max_permutations=args.max_permutations,
    )
    write_json(metrics, args.output)
    print_metrics_summary(metrics)
    print(f"Wrote metrics JSON to {args.output}")


if __name__ == "__main__":
    main()

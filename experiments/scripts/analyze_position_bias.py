from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.evaluation.position_bias import marginal_position_exposure, position_bias_metrics_path
from experiments.scripts.common import print_metrics
from experiments.utils.io import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze marginal top-k position exposure for reranking outputs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", default=None)
    parser.add_argument("--include-values", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = analyze_position_bias_file(
        args.input,
        k=args.k,
        output=args.output,
        include_values=args.include_values,
    )
    print_metrics(metrics)


def analyze_position_bias_file(
    input_path: str | Path,
    *,
    k: int = 5,
    output: str | Path | None = None,
    include_values: bool = False,
) -> dict:
    payload = read_json(input_path)
    metrics = marginal_position_exposure(payload["users"], k=k, include_values=include_values)
    metrics.update(
        {
            "stage": "position_bias",
            "dataset": payload.get("dataset"),
            "source_retriever": payload.get("source_retriever"),
            "reranker": payload.get("reranker"),
            "reranking_mode": payload.get("reranking_mode"),
            "model": payload.get("model"),
            "architecture": payload.get("architecture"),
            "num_users": len(payload.get("users", {})),
        }
    )

    output_path = Path(output) if output else position_bias_metrics_path(
        str(payload.get("dataset", "unknown_dataset")),
        str(payload.get("source_retriever", "unknown_retriever")),
        Path(str(payload.get("source_candidates", "candidates.json"))).stem,
        Path(input_path).stem,
        k,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(metrics, output_path)
    print(f"Saved position-bias metrics to {output_path}")
    return metrics


if __name__ == "__main__":
    main()

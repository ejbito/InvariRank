"""Small framework inference demo kept with the research-facing utilities."""

from __future__ import annotations

import argparse

from invarirank import InvariRankReranker, RerankerConfig

SAMPLE = {
    "user_id": "demo-user",
    "history": [
        {"item_id": "h1", "title": "The Matrix", "year": 1999, "genres": ["Action", "Sci-Fi"], "rating": 5},
        {"item_id": "h2", "title": "Arrival", "year": 2016, "genres": ["Drama", "Sci-Fi"], "rating": 4},
    ],
    "candidates": [
        {"item_id": "m1", "title": "Interstellar", "year": 2014, "genres": ["Adventure", "Sci-Fi"]},
        {"item_id": "m2", "title": "The Notebook", "year": 2004, "genres": ["Drama", "Romance"]},
        {"item_id": "m3", "title": "Blade Runner 2049", "year": 2017, "genres": ["Drama", "Sci-Fi"]},
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank a small candidate set with the InvariRank framework API.")
    parser.add_argument("--model", required=True, help="Hugging Face causal language model name or local path.")
    parser.add_argument("--adapter", help="Optional trained PEFT adapter directory.")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RerankerConfig(device=args.device)
    reranker = InvariRankReranker.from_pretrained(args.model, config=config, adapter_path=args.adapter)
    result = reranker.rank(SAMPLE)

    for rank, item in enumerate(result.items, start=1):
        print(f"{rank}. {item.item_id}: {item.candidate.get('title', '')} ({item.score:.4f})")


if __name__ == "__main__":
    main()

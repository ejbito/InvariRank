from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from invarirank import RerankerConfig, Trainer, TrainingConfig

from experiments.reranking.history import load_user_histories
from experiments.scripts.common import load_dataset_settings, processed_dir, reranker_training_dir
from experiments.utils.io import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LFT or InvariRank marker reranker adapter.")
    parser.add_argument("--method", choices=["lft", "invarirank"], default="invarirank")
    parser.add_argument("--train-candidates", required=True)
    parser.add_argument("--val-candidates", required=True)
    parser.add_argument("--dataset", default="movielens")
    parser.add_argument("--dataset-config", default="experiments/configs/datasets.yaml")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--model-name-or-path", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--total-optimizer-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--train-num-permutations", type=int, default=1)
    parser.add_argument("--eval-num-permutations", type=int, default=10)
    parser.add_argument("--max-history-items", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dataset_settings(args.dataset_config, args.dataset)
    data_dir = processed_dir(config, args.processed_dir)
    train_payload = read_json(args.train_candidates)
    val_payload = read_json(args.val_candidates)
    train_histories = load_user_histories(
        data_dir,
        max_history_items=args.max_history_items,
        split=str(train_payload.get("split") or "train"),
    )
    val_histories = load_user_histories(
        data_dir,
        max_history_items=args.max_history_items,
        split=str(val_payload.get("split") or "val"),
    )
    train_samples = _samples_from_candidate_payload(train_payload, train_histories)
    val_samples = _samples_from_candidate_payload(val_payload, val_histories)
    output_dir = Path(args.output_dir) if args.output_dir else reranker_training_dir(args.dataset, args.method)
    trainer = Trainer.from_pretrained(
        args.model_name_or_path,
        train_samples,
        val_samples,
        reranker_config=RerankerConfig.for_method(
            args.method,
            {
                "device": args.device,
                "dtype": args.torch_dtype,
                "max_length": args.max_length,
                "prompt_template": "invarirank",
            },
        ),
        training_config=TrainingConfig(
            total_optimizer_steps=args.total_optimizer_steps,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            train_num_permutations=args.train_num_permutations,
            eval_num_permutations=args.eval_num_permutations,
        ),
    )
    result = trainer.train(output_dir=output_dir)
    print(f"Saved {args.method} adapter to {Path(output_dir) / 'checkpoints' / 'final'}")
    print(result)


def _samples_from_candidate_artifact(
    path: str | Path,
    histories: Mapping[int, list[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    payload = read_json(path)
    return _samples_from_candidate_payload(payload, histories, source=path)


def _samples_from_candidate_payload(
    payload: Mapping[str, Any],
    histories: Mapping[int, list[Mapping[str, Any]]],
    *,
    source: str | Path = "candidate payload",
) -> list[dict[str, Any]]:
    samples = []
    for user_record in payload["users"].values():
        user_id = int(user_record["user_id"])
        relevant = {str(item_id) for item_id in user_record.get("ground_truth_item_ids", [])}
        candidates = []
        for _rank_key, candidate in sorted(user_record["candidates"].items(), key=lambda item: int(item[0])):
            value = dict(candidate)
            value.setdefault("title", _title(value))
            value["relevance"] = 1 if str(value.get("item_id")) in relevant else 0
            candidates.append(value)
        if any(candidate["relevance"] > 0 for candidate in candidates):
            samples.append(
                {
                    "user_id": str(user_id),
                    "history": [dict(item) for item in histories.get(user_id, [])],
                    "candidates": candidates,
                    "split": payload.get("split"),
                }
            )
    if not samples:
        raise ValueError(f"No trainable samples with relevant candidates found in {source}.")
    return samples


def _title(candidate: Mapping[str, Any]) -> str:
    for key in ("title", "name", "main_category", "item_id"):
        value = candidate.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return "unknown item"


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...utils import save_jsonl, set_seed
from .amazon import AmazonDataset
from .config_utils import cfg_get
from .movielens import MovieLensDataset


DATASET_REGISTRY = {
    "movielens": MovieLensDataset,
    "amazon": AmazonDataset,
}


def validate_sample(sample: dict[str, Any], num_candidates: int) -> None:
    required = {"user_id", "history", "candidates", "target_ranking", "list_length", "split"}
    missing = sorted(required - set(sample))
    if missing:
        raise ValueError(f"Dataset sample is missing field(s): {missing}")

    candidates = sample["candidates"]
    if len(candidates) != num_candidates:
        raise ValueError(f"Expected {num_candidates} candidates, found {len(candidates)}")

    item_ids = [c["item_id"] for c in candidates]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Duplicate items in candidate list")

    rels = [c["relevance"] for c in candidates]
    if not all(isinstance(r, int) for r in rels):
        raise ValueError("All candidate relevance labels must be integers")


def validate_split(samples: list[dict[str, Any]]) -> None:
    for sample in samples:
        validate_sample(sample, int(sample["list_length"]))


def build_dataset_splits(cfg: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    set_seed(int(cfg_get(cfg, "training.seed", cfg_get(cfg, "seed", 42))))
    dataset_name = str(cfg_get(cfg, "dataset.name", "")).lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Expected one of {sorted(DATASET_REGISTRY)}")

    dataset = DATASET_REGISTRY[dataset_name](cfg)
    print("[Dataset] Build start")
    dataset.load_raw()
    dataset.build_item_metadata()
    dataset.build_user_histories()

    train, val, test = dataset.generate_samples()
    print("[Dataset] Validating samples")
    validate_split(train)
    validate_split(val)
    validate_split(test)
    return train, val, test


def write_dataset_splits(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_jsonl(train, output / "train.jsonl")
    save_jsonl(val, output / "val.jsonl")
    save_jsonl(test, output / "test.jsonl")
    print(f"[Dataset] Wrote train={len(train)}, val={len(val)}, test={len(test)} to {output}")

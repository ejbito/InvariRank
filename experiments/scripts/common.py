from __future__ import annotations

from pathlib import Path
from typing import Any


def load_dataset_settings(config_path: str | Path, dataset: str) -> dict[str, Any]:
    from experiments.data.config import load_dataset_config

    return load_dataset_config(config_path, dataset)


def processed_dir(config: dict[str, Any], override: str | Path | None = None) -> Path:
    return Path(override or config["processed_dir"])


def ground_truth_from_interactions(interactions: Any) -> dict[int, list[int]]:
    return (
        interactions.groupby("user_id")["item_id"]
        .apply(lambda values: [int(value) for value in values])
        .to_dict()
    )


def load_trained_retriever(
    retriever_name: str,
    *,
    dataset: str,
    artifact_dir: str | Path = "artifacts/retrievers",
    show_progress: bool = True,
):
    from experiments.retrieval.registry import get_retriever_class

    retriever_cls = get_retriever_class(retriever_name)
    retriever = retriever_cls.load(Path(artifact_dir) / dataset / retriever_name)
    retriever.show_progress = show_progress
    return retriever


def extend_retriever_seen_items(retriever: Any, data_dir: str | Path, split: str) -> None:
    """Exclude every interaction chronologically available before an evaluation split."""
    from experiments.data.interactions import load_interactions

    root = Path(data_dir)
    if split == "train":
        paths = [root / "retriever_train.csv"]
    elif split == "val":
        paths = [root / "train.csv"]
    elif split == "test":
        paths = [root / "train.csv", root / "val.csv"]
    else:
        raise ValueError("split must be one of: train, val, test")
    for path in paths:
        if not path.exists():
            continue
        interactions = load_interactions(path)
        for user_id, item_ids in interactions.groupby("user_id")["item_id"]:
            retriever.seen_items_.setdefault(int(user_id), set()).update(int(item_id) for item_id in item_ids)


def retrieval_metrics_path(dataset: str, retriever: str, split: str, k: int) -> Path:
    return Path("artifacts/metrics/retrieval") / dataset / retriever / f"{split}_k{k}.json"


def candidate_output_path(
    dataset: str,
    retriever: str,
    split: str,
    k: int,
    *,
    max_users: int | None = None,
    selected_users: int | None = None,
    require_ground_truth: bool = False,
) -> Path:
    suffix = "_gt_only" if require_ground_truth else ""
    user_suffix = f"_users{selected_users}" if max_users is not None and selected_users is not None else ""
    return Path("artifacts/candidates") / dataset / retriever / f"{split}_k{k}{user_suffix}{suffix}.json"


def reranker_training_dir(dataset: str, method: str, run_name: str = "invarirank_marker") -> Path:
    return Path("artifacts/rerankers") / dataset / method / run_name


def reranking_metrics_path(
    dataset: str,
    retriever: str,
    candidate_file_stem: str,
    reranker_file_stem: str,
    k: int,
) -> Path:
    return (
        Path("artifacts/metrics/reranking")
        / dataset
        / retriever
        / candidate_file_stem
        / f"{reranker_file_stem}_k{k}.json"
    )


def print_metrics(metrics: dict[str, Any]) -> None:
    for name, value in metrics.items():
        if name in {"per_user_robustness", "position_observations"}:
            continue
        if isinstance(value, float):
            print(f"{name}: {value:.6f}")
        else:
            print(f"{name}: {value}")

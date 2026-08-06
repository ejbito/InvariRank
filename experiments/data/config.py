from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.utils.io import read_yaml


def load_dataset_config(path: str | Path, dataset: str) -> dict[str, Any]:
    config = read_yaml(path)
    if "datasets" in config:
        try:
            return config["datasets"][dataset]
        except KeyError as exc:
            valid = ", ".join(sorted(config["datasets"]))
            raise ValueError(f"Dataset '{dataset}' not found in {path}. Valid datasets: {valid}") from exc

    # Backward compatibility for older single-dataset config files.
    if "dataset" in config:
        dataset_config = config["dataset"]
        if dataset_config.get("name", dataset) != dataset:
            raise ValueError(
                f"Dataset config at {path} is for '{dataset_config.get('name')}', not '{dataset}'."
            )
        return dataset_config

    raise ValueError(f"Dataset config at {path} must contain either 'datasets' or 'dataset'.")

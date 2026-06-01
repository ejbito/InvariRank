from pathlib import Path

import yaml

from invarirank.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def iter_yaml_configs():
    return sorted((ROOT / "configs").rglob("*.yaml"))


def test_all_yaml_configs_load():
    for path in iter_yaml_configs():
        cfg = load_config(path)
        assert cfg.seed is not None


def test_eval_config_uses_current_ranking_controls():
    forbidden = {
        "test_path",
        "ranking_num_users",
        "num_users",
        "test_max_users",
        "test_max_samples",
        "sample_shuffle",
        "sample_seed",
    }
    path = ROOT / "configs" / "eval" / "rank.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    assert "data_path" in raw
    assert "output_dir" in raw
    assert "eval_num_permutations" in raw
    assert not (forbidden & set(raw))


def test_expected_barebones_configs_exist():
    expected = [
        "configs/data/movielens.yaml",
        "configs/data/amazon_books.yaml",
        "configs/dev/smoke.yaml",
        "configs/eval/rank.yaml",
        "configs/train/train.yaml",
    ]
    for relative_path in expected:
        assert (ROOT / relative_path).exists(), relative_path


def test_only_one_non_data_config_per_stage():
    expected_counts = {"dev": 1, "eval": 1, "train": 1}
    for stage, count in expected_counts.items():
        assert len(list((ROOT / "configs" / stage).glob("*.yaml"))) == count

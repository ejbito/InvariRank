from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .framework import (
    InvariRankReranker,
    RankingSample,
    RerankerConfig,
    _load_json_mapping,
    _save_json_mapping,
)
from .modeling import (
    align_scores_to_shared_candidates,
    build_lora_model,
    load_tokenizer,
    select_device,
)
from .prompts import build_prompt, candidate_id, extract_relevance_labels


def _metric_values(values: Any) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    return [float(value) for value in values]


def _hr_at_k(scores: Any, relevance: Any, k: int) -> float:
    score_values = _metric_values(scores)
    relevance_values = _metric_values(relevance)
    if not score_values or not relevance_values:
        return 0.0
    count = min(len(score_values), len(relevance_values))
    order = sorted(range(count), key=lambda index: score_values[index], reverse=True)
    return 1.0 if any(relevance_values[index] > 0 for index in order[: min(k, count)]) else 0.0


def _ndcg_at_k(scores: Any, relevance: Any, k: int) -> float:
    score_values = _metric_values(scores)
    relevance_values = _metric_values(relevance)
    count = min(len(score_values), len(relevance_values), k)
    if count <= 0:
        return 0.0
    order = sorted(
        range(min(len(score_values), len(relevance_values))),
        key=lambda index: score_values[index],
        reverse=True,
    )[:count]
    ideal = sorted(relevance_values, reverse=True)[:count]

    def dcg(labels: list[float]) -> float:
        return sum((2.0**label - 1.0) / math.log2(rank + 2) for rank, label in enumerate(labels))

    ideal_dcg = dcg(ideal)
    return 0.0 if ideal_dcg == 0 else float(dcg([relevance_values[index] for index in order]) / ideal_dcg)


def _spearman_rho_from_rank_maps(first: dict[Any, int], second: dict[Any, int]) -> float | None:
    keys = sorted(set(first) & set(second))
    count = len(keys)
    if count < 2:
        return None
    difference_squared = sum((first[key] - second[key]) ** 2 for key in keys)
    return float(1.0 - (6.0 * difference_squared) / (count * (count * count - 1)))


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    train_num_permutations: int = 1
    eval_num_permutations: int = 10
    val_perms_deterministic: bool = True
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lambda_rank: float = 1.0
    lambda_perm: float = 0.0
    permutation_loss: str = "kl"
    num_epochs: int | None = None
    total_optimizer_steps: int | None = 500
    save_every_steps: int | None = None
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    extras: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extras", dict(self.extras or {}))
        if self.train_num_permutations < 1 or self.eval_num_permutations < 1:
            raise ValueError("Permutation counts must be at least one.")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least one.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")
        if self.num_epochs is None and self.total_optimizer_steps is None:
            raise ValueError("Set num_epochs or total_optimizer_steps.")
        if self.permutation_loss not in {"kl", "symkl", "jeffreys"}:
            raise ValueError(f"Unsupported permutation_loss: {self.permutation_loss}")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> TrainingConfig:
        data = dict(values)
        known = {item.name for item in fields(cls)} - {"extras"}
        kwargs = {key: data.pop(key) for key in list(data) if key in known}
        if "lora_target_modules" in kwargs:
            kwargs["lora_target_modules"] = tuple(kwargs["lora_target_modules"])
        return cls(**kwargs, extras=data)

    def to_dict(self) -> dict[str, Any]:
        """Return a round-trippable, JSON-compatible configuration mapping."""
        values = asdict(self)
        extras = values.pop("extras")
        values["lora_target_modules"] = list(values["lora_target_modules"])
        return {**extras, **values}

    def save_json(self, path: str | Path) -> None:
        """Save this configuration as human-readable JSON."""
        _save_json_mapping(self.to_dict(), path)

    @classmethod
    def from_json(cls, path: str | Path) -> TrainingConfig:
        """Load and validate a configuration from JSON."""
        return cls.from_mapping(_load_json_mapping(path))

    def to_namespace(self, **overrides: Any) -> SimpleNamespace:
        values = dict(self.extras)
        values.update(asdict(self))
        values.pop("extras", None)
        values["lora_target_modules"] = list(values["lora_target_modules"])
        values.update(overrides)
        return SimpleNamespace(**values)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def sample_permutation(count: int, *, deterministic: bool = False, seed: int | None = None) -> list[int]:
    permutation = list(range(count))
    generator = random.Random(seed) if deterministic else random
    generator.shuffle(permutation)
    return permutation


class ListwiseRankingDataset:
    def __init__(self, samples: list[dict[str, Any]], cfg: Any, tokenizer: Any, *, mode: str = "train"):
        self.samples = samples
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.mode = mode

    def __len__(self) -> int:
        return len(self.samples)

    def _num_permutations(self) -> int:
        if self.mode == "train":
            return int(getattr(self.cfg, "train_num_permutations", 1))
        return int(getattr(self.cfg, "eval_num_permutations", 1))

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        candidate_count = len(sample["candidates"])
        tokenized = []
        relevance = []
        permutations = []
        deterministic = self.mode != "train" and bool(getattr(self.cfg, "val_perms_deterministic", True))
        for permutation_index in range(self._num_permutations()):
            permutation = sample_permutation(
                candidate_count,
                deterministic=deterministic,
                seed=index * 1009 + permutation_index,
            )
            prompt = build_prompt(sample, permutation, self.cfg)
            encoding = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=int(self.cfg.max_seq_length),
            )
            tokenized.append(encoding)
            relevance.append(extract_relevance_labels(sample, permutation))
            permutations.append(permutation)
        return {
            "sample_index": index,
            "user_id": sample.get("user_id", str(index)),
            "split": sample.get("split", self.mode),
            "history": sample.get("history", []),
            "candidates": sample["candidates"],
            "num_items": candidate_count,
            "list_length": candidate_count,
            "candidate_ids": [
                candidate_id(candidate, candidate_index)
                for candidate_index, candidate in enumerate(sample["candidates"])
            ],
            "tokenized": tokenized,
            "relevance": relevance,
            "permutations": permutations,
            "sample": sample,
        }


def listwise_collator(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("ListwiseRankingDataset currently expects batch_size=1.")
    return batch[0]


def filter_and_subsample(
    samples: list[dict[str, Any]],
    num_samples: int | None = None,
) -> list[dict[str, Any]]:
    valid = [sample for sample in samples if sample.get("candidates")]
    return valid if num_samples is None else valid[: int(num_samples)]


def lambda_rank_loss(scores: Any, relevance: Any, sigma: float = 1.0, eps: float = 1e-8):
    import torch
    import torch.nn.functional as functional

    device = scores.device
    scores = scores.float()
    relevance = relevance.float()
    if torch.all(relevance == relevance[0]):
        return torch.tensor(0.0, device=device)

    count = scores.numel()
    sorted_indices = torch.argsort(scores, descending=True)
    rank_positions = torch.empty(count, dtype=torch.long, device=device)
    rank_positions[sorted_indices] = torch.arange(count, device=device)
    ideal_relevance = torch.sort(relevance, descending=True).values
    discounts = 1.0 / torch.log2(torch.arange(count, device=device).float() + 2.0)
    ideal_dcg = torch.sum((torch.pow(2.0, ideal_relevance) - 1.0) * discounts).clamp(min=eps)

    score_differences = scores.unsqueeze(1) - scores.unsqueeze(0)
    relevance_differences = relevance.unsqueeze(1) - relevance.unsqueeze(0)
    preference_mask = relevance_differences > 0
    if preference_mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    gains = torch.pow(2.0, relevance) - 1.0
    first_discounts = discounts[rank_positions].unsqueeze(1)
    second_discounts = discounts[rank_positions].unsqueeze(0)
    delta_ndcg = torch.abs((gains.unsqueeze(1) - gains.unsqueeze(0)) * (first_discounts - second_discounts)) / ideal_dcg
    pair_loss = functional.softplus(-sigma * score_differences)
    return (delta_ndcg * pair_loss * preference_mask.float()).sum() / (preference_mask.sum().float() + eps)


def permutation_invariance_loss(
    scores_list: list[Any],
    permutations: list[list[int]],
    mode: str = "kl",
    temperature: float = 1.0,
):
    import torch
    import torch.nn.functional as functional

    aligned, _ = align_scores_to_shared_candidates(scores_list, permutations)
    if aligned is None or len(aligned) < 2:
        return torch.tensor(0.0, device=scores_list[0].device)
    stacked = torch.stack([scores.float() for scores in aligned], dim=0)
    log_probabilities = functional.log_softmax(stacked / max(temperature, 1e-6), dim=-1)
    probabilities = log_probabilities.exp()
    base_log = log_probabilities[0]
    base_probability = probabilities[0]
    losses = []
    for index in range(1, log_probabilities.shape[0]):
        current_log = log_probabilities[index]
        current_probability = probabilities[index]
        forward_kl = torch.sum(base_probability * (base_log - current_log), dim=-1)
        if mode == "kl":
            losses.append(forward_kl)
        elif mode in {"symkl", "jeffreys"}:
            reverse_kl = torch.sum(current_probability * (current_log - base_log), dim=-1)
            losses.append(forward_kl + reverse_kl)
        else:
            raise ValueError(f"Unsupported permutation loss mode: {mode}")
    return torch.stack(losses).mean()


def train_step(
    batch: dict[str, Any],
    scorer: Any,
    optimizer: Any,
    scaler: Any,
    cfg: Any,
    micro_step: int,
) -> dict[str, Any]:
    import torch

    scorer.train()
    device = torch.device(cfg.device)
    use_autocast = cfg.dtype == "float16" and device.type == "cuda"
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    scores_list = []
    relevance_list = []
    permutations = batch["permutations"]

    with torch.amp.autocast(autocast_device, enabled=use_autocast):
        for encoding, relevance in zip(batch["tokenized"], batch["relevance"]):
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            scores = scorer(input_ids, attention_mask, expected_candidates=len(relevance))
            relevance_tensor = torch.tensor(relevance, device=device, dtype=torch.float32)[: scores.numel()]
            scores_list.append(scores)
            relevance_list.append(relevance_tensor)

        rank_loss = torch.stack(
            [lambda_rank_loss(scores, relevance) for scores, relevance in zip(scores_list, relevance_list)]
        ).mean()
        if cfg.lambda_perm > 0 and len(scores_list) >= 2:
            permutation_loss = permutation_invariance_loss(
                scores_list,
                permutations,
                mode=cfg.permutation_loss,
            )
        else:
            permutation_loss = torch.tensor(0.0, device=device)
        loss = (cfg.lambda_rank * rank_loss + cfg.lambda_perm * permutation_loss) / cfg.gradient_accumulation_steps

    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    did_step = False
    if (micro_step + 1) % cfg.gradient_accumulation_steps == 0:
        parameters = [parameter for parameter in scorer.parameters() if parameter.requires_grad]
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, cfg.max_grad_norm)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        did_step = True

    return {
        "loss_total": float(loss.item() * cfg.gradient_accumulation_steps),
        "loss_rank": float(rank_loss.item()),
        "loss_perm": float(permutation_loss.item()),
        "did_step": did_step,
    }


def evaluate_validation(loader: Any, scorer: Any, cfg: Any) -> dict[str, float]:
    import torch
    from tqdm.auto import tqdm

    scorer.eval()
    device = torch.device(cfg.device)
    hit_rate_5, hit_rate_10, ndcg_5, ndcg_10, spearmans = [], [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            scores_list = []
            for encoding in batch["tokenized"]:
                scores_list.append(
                    scorer(
                        encoding["input_ids"].to(device),
                        encoding["attention_mask"].to(device),
                        expected_candidates=len(batch["relevance"][len(scores_list)]),
                    )
                )
            first_scores = scores_list[0]
            first_relevance = torch.tensor(
                batch["relevance"][0],
                device=device,
                dtype=torch.float32,
            )[: first_scores.numel()]
            hit_rate_5.append(_hr_at_k(first_scores, first_relevance, 5))
            hit_rate_10.append(_hr_at_k(first_scores, first_relevance, 10))
            ndcg_5.append(_ndcg_at_k(first_scores, first_relevance, 5))
            ndcg_10.append(_ndcg_at_k(first_scores, first_relevance, 10))

            aligned, _ = align_scores_to_shared_candidates(scores_list, batch["permutations"])
            if aligned is not None and len(aligned) > 1:
                base_order = {
                    index: rank
                    for rank, index in enumerate(aligned[0].detach().cpu().argsort(descending=True).tolist())
                }
                for other in aligned[1:]:
                    other_order = {
                        index: rank for rank, index in enumerate(other.detach().cpu().argsort(descending=True).tolist())
                    }
                    value = _spearman_rho_from_rank_maps(base_order, other_order)
                    if value is not None:
                        spearmans.append(value)

    def average(values: list[float]) -> float:
        return float(sum(values) / max(len(values), 1))

    return {
        "hr@5": average(hit_rate_5),
        "hr@10": average(hit_rate_10),
        "ndcg@5": average(ndcg_5),
        "ndcg@10": average(ndcg_10),
        "perm_spearman": average(spearmans),
    }


def save_checkpoint(
    scorer: Any,
    optimizer: Any,
    checkpoint_dir: str | Path,
    tag: str,
    epoch: int,
    global_step: int,
    metrics: dict[str, float],
) -> None:
    import torch

    path = ensure_dir(Path(checkpoint_dir) / tag)
    scorer.backbone.save_pretrained(path)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
        },
        path / "trainer.pt",
    )


class Trainer:
    """Train a framework reranker from in-memory ranking samples."""

    def __init__(
        self,
        reranker: InvariRankReranker,
        train_samples: list[RankingSample | Mapping[str, Any]],
        validation_samples: list[RankingSample | Mapping[str, Any]],
        config: TrainingConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self.reranker = reranker
        self.train_samples = [_sample_to_dict(sample) for sample in train_samples]
        self.validation_samples = [_sample_to_dict(sample) for sample in validation_samples]
        self.config = _coerce_training_config(config)
        if not self.train_samples:
            raise ValueError("Trainer requires at least one training sample.")

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        train_samples: list[RankingSample | Mapping[str, Any]],
        validation_samples: list[RankingSample | Mapping[str, Any]],
        *,
        reranker_config: RerankerConfig | Mapping[str, Any] | None = None,
        training_config: TrainingConfig | Mapping[str, Any] | None = None,
    ) -> Trainer:
        framework_config = (
            reranker_config
            if isinstance(reranker_config, RerankerConfig)
            else RerankerConfig.from_mapping(reranker_config or {})
        )
        framework_config = replace(framework_config, model_name=model_name)
        train_config = _coerce_training_config(training_config)
        device = select_device(framework_config.device)
        combined = vars(framework_config.to_namespace()).copy()
        combined.update(vars(train_config.to_namespace()))
        combined.update({"model_name": model_name, "device": str(device)})
        cfg = SimpleNamespace(**combined)
        tokenizer = load_tokenizer(cfg)
        backbone = build_lora_model(cfg, tokenizer, device)
        reranker = InvariRankReranker(backbone, tokenizer, framework_config, device=device)
        return cls(reranker, train_samples, validation_samples, train_config)

    def train(self, *, output_dir: str | Path) -> dict[str, Any]:
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from tqdm.auto import tqdm

        set_seed(self.config.seed)
        output = ensure_dir(output_dir)
        checkpoint_dir = ensure_dir(output / "checkpoints")
        combined = vars(self.reranker._legacy_config).copy()
        combined.update(vars(self.config.to_namespace()))
        combined.update(
            {
                "device": str(self.reranker.device),
                "dtype": self.reranker.config.dtype,
                "max_seq_length": self.reranker.config.max_length,
            }
        )
        cfg = SimpleNamespace(**combined)
        train_dataset = ListwiseRankingDataset(self.train_samples, cfg, self.reranker.tokenizer, mode="train")
        validation_dataset = ListwiseRankingDataset(
            self.validation_samples,
            cfg,
            self.reranker.tokenizer,
            mode="val",
        )
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=listwise_collator)
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=listwise_collator,
        )
        parameters = [parameter for parameter in self.reranker.scorer.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("The reranker has no trainable parameters.")
        optimizer = AdamW(parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        device = torch.device(cfg.device)
        scaler = torch.amp.GradScaler("cuda") if cfg.dtype == "float16" and device.type == "cuda" else None

        with (output / "config.json").open("w", encoding="utf-8") as handle:
            json.dump({**combined, "output_dir": str(output)}, handle, indent=2)
        log_path = output / "training_log.jsonl"
        global_step = 0
        micro_step = 0
        maximum_epochs = cfg.num_epochs or 10**9
        for epoch in range(1, int(maximum_epochs) + 1):
            progress = tqdm(train_loader, desc=f"Training epoch {epoch}")
            for batch in progress:
                log = train_step(batch, self.reranker.scorer, optimizer, scaler, cfg, micro_step)
                micro_step += 1
                if log["did_step"]:
                    global_step += 1
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"epoch": epoch, "global_step": global_step, **log}) + "\n")
                    if cfg.save_every_steps and global_step % int(cfg.save_every_steps) == 0:
                        metrics = evaluate_validation(validation_loader, self.reranker.scorer, cfg)
                        save_checkpoint(
                            self.reranker.scorer,
                            optimizer,
                            checkpoint_dir,
                            f"step_{global_step}",
                            epoch,
                            global_step,
                            metrics,
                        )
                    if cfg.total_optimizer_steps is not None and global_step >= int(cfg.total_optimizer_steps):
                        return self._finish(
                            validation_loader,
                            optimizer,
                            checkpoint_dir,
                            cfg,
                            epoch,
                            global_step,
                        )
                progress.set_postfix({"step": global_step, "loss": f"{log['loss_total']:.4f}"})
        return self._finish(validation_loader, optimizer, checkpoint_dir, cfg, epoch, global_step)

    def _finish(
        self,
        validation_loader: Any,
        optimizer: Any,
        checkpoint_dir: Path,
        cfg: Any,
        epoch: int,
        global_step: int,
    ) -> dict[str, Any]:
        metrics = evaluate_validation(validation_loader, self.reranker.scorer, cfg)
        save_checkpoint(
            self.reranker.scorer,
            optimizer,
            checkpoint_dir,
            "final",
            epoch,
            global_step,
            metrics,
        )
        return {"global_step": global_step, "metrics": metrics}


def run_training_pipeline(cfg: Any) -> dict[str, Any]:
    required = ["model_name", "train_path", "val_path", "run_dir"]
    missing = [name for name in required if not getattr(cfg, name, None)]
    if missing:
        raise ValueError(f"Missing required config field(s): {', '.join(missing)}")
    train_samples = filter_and_subsample(
        load_jsonl(cfg.train_path),
        getattr(cfg, "train_max_samples", None),
    )
    validation_samples = filter_and_subsample(
        load_jsonl(cfg.val_path),
        getattr(cfg, "val_max_samples", None),
    )
    return Trainer.from_pretrained(
        cfg.model_name,
        train_samples,
        validation_samples,
        reranker_config=RerankerConfig.from_mapping(vars(cfg)),
        training_config=TrainingConfig.from_mapping(vars(cfg)),
    ).train(output_dir=cfg.run_dir)


def _coerce_training_config(config: TrainingConfig | Mapping[str, Any] | None) -> TrainingConfig:
    if config is None:
        return TrainingConfig()
    if isinstance(config, TrainingConfig):
        return config
    if isinstance(config, Mapping):
        return TrainingConfig.from_mapping(config)
    raise TypeError("config must be a TrainingConfig, mapping, or None.")


def _sample_to_dict(sample: RankingSample | Mapping[str, Any]) -> dict[str, Any]:
    return sample.to_dict() if isinstance(sample, RankingSample) else dict(sample)


__all__ = [
    "ListwiseRankingDataset",
    "Trainer",
    "TrainingConfig",
    "evaluate_validation",
    "filter_and_subsample",
    "lambda_rank_loss",
    "listwise_collator",
    "permutation_invariance_loss",
    "run_training_pipeline",
    "sample_permutation",
    "save_checkpoint",
    "train_step",
]

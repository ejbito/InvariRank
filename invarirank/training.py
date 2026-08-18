from __future__ import annotations

import json
import math
import random
import shutil
from collections.abc import Mapping
from dataclasses import replace
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .config import RerankerConfig, TrainingConfig
from .contracts import RankingSample
from .modeling import (
    align_scores_to_shared_candidates,
    build_lora_model,
    load_tokenizer,
    select_device,
)
from .prompts import build_prompt, candidate_id, extract_relevance_labels
from .reranker import InvariRankReranker

LATEST_CHECKPOINT_NAME = "latest"
FINAL_CHECKPOINT_NAME = "final"
TRAINER_STATE_NAME = "trainer.pt"


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
        _validate_relevance_labels(samples, f"{mode} dataset")
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
        elif mode == "jeffreys":
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
    *,
    reranker: InvariRankReranker | None = None,
    training_config: Any | None = None,
    micro_step: int = 0,
    scaler: Any | None = None,
) -> None:
    import torch

    path = ensure_dir(Path(checkpoint_dir) / tag)
    if reranker is not None:
        reranker.save_pretrained(path)
    else:
        scorer.backbone.save_pretrained(path)
        interaction_model = getattr(scorer, "interaction_model", None)
        if interaction_model is not None:
            from safetensors.torch import save_file

            from .config import INTERACTION_WEIGHTS_NAME

            save_file(
                {
                    name: tensor.detach().cpu().contiguous()
                    for name, tensor in interaction_model.state_dict().items()
                },
                str(path / INTERACTION_WEIGHTS_NAME),
            )
    state = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "micro_step": int(micro_step),
        "optimizer": optimizer.state_dict(),
        "gradients": _capture_gradients(scorer),
        "metrics": metrics,
        "rng_state": _capture_rng_state(),
    }
    if training_config is not None:
        state["training_signature"] = _training_resume_signature(training_config)
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, path / TRAINER_STATE_NAME)


def _capture_gradients(scorer: Any) -> dict[str, Any]:
    return {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in scorer.module.named_parameters()
        if parameter.grad is not None
    }


def _restore_gradients(scorer: Any, gradients: Mapping[str, Any]) -> None:
    parameters = dict(scorer.module.named_parameters())
    unexpected = sorted(set(gradients) - set(parameters))
    if unexpected:
        raise ValueError(f"Checkpoint contains gradients for unknown parameters: {unexpected}")
    for parameter in parameters.values():
        parameter.grad = None
    for name, gradient in gradients.items():
        parameter = parameters[name]
        parameter.grad = gradient.to(device=parameter.device, dtype=parameter.dtype)


def _capture_rng_state() -> dict[str, Any]:
    import torch

    state: dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    import torch

    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "numpy" in state:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ImportError:
            pass


def _training_resume_signature(cfg: Any) -> dict[str, Any]:
    keys = (
        "seed",
        "train_num_permutations",
        "gradient_accumulation_steps",
        "learning_rate",
        "weight_decay",
        "max_grad_norm",
        "lambda_rank",
        "lambda_perm",
        "permutation_loss",
    )
    return {key: getattr(cfg, key) for key in keys}


def _load_training_state(
    checkpoint: Path,
    scorer: Any,
    optimizer: Any,
    cfg: Any,
    scaler: Any | None = None,
) -> tuple[int, int, int]:
    import torch

    state_path = checkpoint / TRAINER_STATE_NAME
    if not state_path.is_file():
        raise ValueError(f"Resume checkpoint is missing {TRAINER_STATE_NAME}: {checkpoint}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    required = {"epoch", "global_step", "optimizer"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"Resume checkpoint {checkpoint} is missing trainer fields: {missing}")
    saved_signature = state.get("training_signature")
    current_signature = _training_resume_signature(cfg)
    if saved_signature is not None and saved_signature != current_signature:
        differences = {
            key: {"saved": saved_signature.get(key), "requested": current_signature.get(key)}
            for key in current_signature
            if saved_signature.get(key) != current_signature.get(key)
        }
        raise ValueError(f"Training configuration does not match the latest checkpoint: {differences}")
    optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    _restore_gradients(scorer, state.get("gradients", {}))
    _restore_rng_state(state.get("rng_state", {}))
    return int(state["epoch"]), int(state["global_step"]), int(state.get("micro_step", 0))


def _save_checkpoint_atomically(
    scorer: Any,
    optimizer: Any,
    checkpoint_dir: Path,
    tag: str,
    epoch: int,
    global_step: int,
    micro_step: int,
    metrics: dict[str, float],
    reranker: InvariRankReranker,
    cfg: Any,
    scaler: Any | None = None,
) -> Path:
    staging_tag = f".{tag}_staging"
    backup = checkpoint_dir / f".{tag}_previous"
    staging = checkpoint_dir / staging_tag
    destination = checkpoint_dir / tag
    _remove_checkpoint_directory(staging, checkpoint_dir)
    save_checkpoint(
        scorer,
        optimizer,
        checkpoint_dir,
        staging_tag,
        epoch,
        global_step,
        metrics,
        reranker=reranker,
        training_config=cfg,
        micro_step=micro_step,
        scaler=scaler,
    )
    _remove_checkpoint_directory(backup, checkpoint_dir)
    if destination.exists():
        destination.rename(backup)
    try:
        staging.rename(destination)
    except Exception:
        if backup.exists() and not destination.exists():
            backup.rename(destination)
        raise
    _remove_checkpoint_directory(backup, checkpoint_dir)
    return destination


def _remove_checkpoint_directory(path: Path, checkpoint_dir: Path) -> None:
    if not path.exists():
        return
    resolved_parent = path.resolve().parent
    if resolved_parent != checkpoint_dir.resolve():
        raise ValueError(f"Refusing to remove checkpoint path outside {checkpoint_dir}: {path}")
    if not path.is_dir():
        raise ValueError(f"Expected checkpoint path to be a directory: {path}")
    shutil.rmtree(path)


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
        _validate_relevance_labels(self.train_samples, "training")
        _validate_relevance_labels(self.validation_samples, "validation")

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
        resolved_train_samples = [_sample_to_dict(sample) for sample in train_samples]
        resolved_validation_samples = [_sample_to_dict(sample) for sample in validation_samples]
        _validate_relevance_labels(resolved_train_samples, "training")
        _validate_relevance_labels(resolved_validation_samples, "validation")
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
        return cls(reranker, resolved_train_samples, resolved_validation_samples, train_config)

    def train(
        self,
        *,
        output_dir: str | Path,
        resume_from_checkpoint: str | Path | None = None,
    ) -> dict[str, Any]:
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

        start_epoch = 1
        global_step = 0
        micro_step = 0
        if resume_from_checkpoint is not None:
            resume_path = Path(resume_from_checkpoint)
            completed_epoch, global_step, micro_step = _load_training_state(
                resume_path,
                self.reranker.scorer,
                optimizer,
                cfg,
                scaler,
            )
            start_epoch = completed_epoch + 1
            print(
                f"Resuming from {resume_path} at epoch {start_epoch}, "
                f"global step {global_step}, micro step {micro_step}."
            )
        elif (output / "training_log.jsonl").exists():
            # A directory without a usable checkpoint is a fresh run. Avoid
            # mixing logs from a previously failed pre-checkpoint attempt.
            (output / "training_log.jsonl").write_text("", encoding="utf-8")

        with (output / "config.json").open("w", encoding="utf-8") as handle:
            json.dump({**combined, "output_dir": str(output)}, handle, indent=2)
        log_path = output / "training_log.jsonl"
        maximum_epochs = cfg.num_epochs or 10**9
        if cfg.total_optimizer_steps is not None and global_step >= int(cfg.total_optimizer_steps):
            return self._finish(
                validation_loader,
                optimizer,
                checkpoint_dir,
                cfg,
                max(0, start_epoch - 1),
                global_step,
                micro_step,
                scaler,
            )
        last_log: dict[str, Any] = {}
        last_completed_epoch = start_epoch - 1
        for epoch in range(start_epoch, int(maximum_epochs) + 1):
            progress = tqdm(train_loader, desc=f"Training epoch {epoch}")
            for batch in progress:
                log = train_step(batch, self.reranker.scorer, optimizer, scaler, cfg, micro_step)
                last_log = log
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
                            reranker=self.reranker,
                            training_config=cfg,
                            micro_step=micro_step,
                            scaler=scaler,
                        )
                    if cfg.total_optimizer_steps is not None and global_step >= int(cfg.total_optimizer_steps):
                        return self._finish(
                            validation_loader,
                            optimizer,
                            checkpoint_dir,
                            cfg,
                            epoch,
                            global_step,
                            micro_step,
                            scaler,
                        )
                progress.set_postfix({"step": global_step, "loss": f"{log['loss_total']:.4f}"})
            latest_metrics = {
                key: float(value)
                for key, value in last_log.items()
                if key.startswith("loss_") and isinstance(value, (int, float))
            }
            latest = _save_checkpoint_atomically(
                self.reranker.scorer,
                optimizer,
                checkpoint_dir,
                LATEST_CHECKPOINT_NAME,
                epoch,
                global_step,
                micro_step,
                latest_metrics,
                self.reranker,
                cfg,
                scaler,
            )
            last_completed_epoch = epoch
            print(f"Saved completed epoch {epoch} checkpoint to {latest}")
        return self._finish(
            validation_loader,
            optimizer,
            checkpoint_dir,
            cfg,
            last_completed_epoch,
            global_step,
            micro_step,
            scaler,
        )

    def _finish(
        self,
        validation_loader: Any,
        optimizer: Any,
        checkpoint_dir: Path,
        cfg: Any,
        epoch: int,
        global_step: int,
        micro_step: int,
        scaler: Any | None,
    ) -> dict[str, Any]:
        metrics = evaluate_validation(validation_loader, self.reranker.scorer, cfg)
        final_path = _save_checkpoint_atomically(
            self.reranker.scorer,
            optimizer,
            checkpoint_dir,
            FINAL_CHECKPOINT_NAME,
            epoch,
            global_step,
            micro_step,
            metrics,
            self.reranker,
            cfg,
            scaler,
        )
        _remove_checkpoint_directory(checkpoint_dir / LATEST_CHECKPOINT_NAME, checkpoint_dir)
        _remove_checkpoint_directory(checkpoint_dir / f".{LATEST_CHECKPOINT_NAME}_previous", checkpoint_dir)
        _remove_checkpoint_directory(checkpoint_dir / f".{LATEST_CHECKPOINT_NAME}_staging", checkpoint_dir)
        print(f"Saved final checkpoint to {final_path}; removed the latest epoch checkpoint.")
        return {"global_step": global_step, "metrics": metrics}


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


def _validate_relevance_labels(samples: list[dict[str, Any]], split: str) -> None:
    for sample_index, sample in enumerate(samples):
        candidates = sample.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"{split.capitalize()} sample {sample_index} must contain at least one candidate.")
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise TypeError(
                    f"{split.capitalize()} sample {sample_index} candidate {candidate_index} must be a mapping."
                )
            if "relevance" not in candidate:
                raise ValueError(
                    f"{split.capitalize()} sample {sample_index} candidate {candidate_index} is missing the required "
                    "relevance label."
                )
            relevance = candidate["relevance"]
            if isinstance(relevance, bool) or not isinstance(relevance, Integral):
                raise TypeError(
                    f"{split.capitalize()} sample {sample_index} candidate {candidate_index} relevance must be a "
                    f"non-negative integer, got {type(relevance).__name__}."
                )
            if relevance < 0:
                raise ValueError(
                    f"{split.capitalize()} sample {sample_index} candidate {candidate_index} relevance must be "
                    f"non-negative, got {relevance}."
                )


__all__ = [
    "ListwiseRankingDataset",
    "Trainer",
    "TrainingConfig",
    "evaluate_validation",
    "lambda_rank_loss",
    "listwise_collator",
    "permutation_invariance_loss",
    "sample_permutation",
    "save_checkpoint",
    "train_step",
]

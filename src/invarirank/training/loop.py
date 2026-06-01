from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import dump_config, require_fields
from ..data import ListwiseRankingDataset, filter_and_subsample, listwise_collator
from ..evaluation import hr_at_k, ndcg_at_k
from ..modeling import build_lora_model, load_tokenizer, select_device
from ..ranking import MeanLogProbListwiseScorer, align_scores_to_shared_candidates
from ..utils import ensure_dir, load_jsonl, set_seed
from .losses import lambda_rank_loss, permutation_invariance_loss


def train_step(batch: dict[str, Any], scorer: Any, optimizer: Any, scaler: Any, cfg: Any, micro_step: int) -> dict[str, Any]:
    import torch

    scorer.train()
    device = torch.device(cfg.device)
    use_autocast = cfg.dtype == "float16" and device.type == "cuda"
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    scores_list = []
    relevance_list = []
    perms = batch["permutations"]

    with torch.amp.autocast(autocast_device, enabled=use_autocast):
        for enc, rel in zip(batch["tokenized"], batch["relevance"]):
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            scores = scorer(input_ids, attention_mask)
            rel_tensor = torch.tensor(rel, device=device, dtype=torch.float32)[: scores.numel()]
            scores_list.append(scores)
            relevance_list.append(rel_tensor)

        loss_rank = torch.stack([lambda_rank_loss(s, r) for s, r in zip(scores_list, relevance_list)]).mean()
        if cfg.lambda_perm > 0 and len(scores_list) >= 2:
            loss_perm = permutation_invariance_loss(scores_list, perms, mode=cfg.permutation_loss)
        else:
            loss_perm = torch.tensor(0.0, device=device)
        loss = (cfg.lambda_rank * loss_rank + cfg.lambda_perm * loss_perm) / cfg.gradient_accumulation_steps

    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    did_step = False
    if (micro_step + 1) % cfg.gradient_accumulation_steps == 0:
        params = [p for p in scorer.parameters() if p.requires_grad]
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        did_step = True

    return {
        "loss_total": float(loss.item() * cfg.gradient_accumulation_steps),
        "loss_rank": float(loss_rank.item()),
        "loss_perm": float(loss_perm.item()),
        "did_step": did_step,
    }


def evaluate(loader: Any, scorer: Any, cfg: Any) -> dict[str, float]:
    import torch
    from tqdm.auto import tqdm

    scorer.eval()
    device = torch.device(cfg.device)
    hr5, hr10, ndcg5, ndcg10, spears = [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            scores_list = []
            for enc in batch["tokenized"]:
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                scores_list.append(scorer(input_ids, attention_mask))

            scores0 = scores_list[0]
            rel0 = torch.tensor(batch["relevance"][0], device=device, dtype=torch.float32)[: scores0.numel()]
            hr5.append(hr_at_k(scores0, rel0, 5))
            hr10.append(hr_at_k(scores0, rel0, 10))
            ndcg5.append(ndcg_at_k(scores0, rel0, 5))
            ndcg10.append(ndcg_at_k(scores0, rel0, 10))

            aligned, _ = align_scores_to_shared_candidates(scores_list, batch["permutations"])
            if aligned is not None and len(aligned) > 1:
                from ..evaluation import spearman_rho_from_rank_maps

                base_order = {i: rank for rank, i in enumerate(aligned[0].detach().cpu().argsort(descending=True).tolist())}
                for other in aligned[1:]:
                    other_order = {i: rank for rank, i in enumerate(other.detach().cpu().argsort(descending=True).tolist())}
                    spears.append(spearman_rho_from_rank_maps(base_order, other_order))

    mean = lambda xs: float(sum(xs) / max(len(xs), 1))
    return {
        "hr@5": mean(hr5),
        "hr@10": mean(hr10),
        "ndcg@5": mean(ndcg5),
        "ndcg@10": mean(ndcg10),
        "perm_spearman": mean(spears),
    }


def save_checkpoint(scorer: Any, optimizer: Any, cfg: Any, tag: str, epoch: int, global_step: int, metrics: dict[str, float]) -> None:
    import torch

    path = Path(cfg.checkpoint_dir) / tag
    ensure_dir(path)
    scorer.backbone.save_pretrained(path)
    torch.save(
        {"epoch": epoch, "global_step": global_step, "optimizer": optimizer.state_dict(), "metrics": metrics},
        path / "trainer.pt",
    )


def run_training_pipeline(cfg: Any) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    require_fields(cfg, ["model_name", "train_path", "val_path", "run_dir"])
    set_seed(int(cfg.seed))
    device = select_device(cfg.device)
    cfg.device = str(device)

    tokenizer = load_tokenizer(cfg)
    model = build_lora_model(cfg, tokenizer, device)
    scorer = MeanLogProbListwiseScorer(model, tokenizer, cfg).to(device)

    train_data = filter_and_subsample(load_jsonl(cfg.train_path), getattr(cfg, "train_max_samples", None))
    val_data = filter_and_subsample(load_jsonl(cfg.val_path), getattr(cfg, "val_max_samples", None))
    train_dataset = ListwiseRankingDataset(train_data, cfg, tokenizer, mode="train")
    val_dataset = ListwiseRankingDataset(val_data, cfg, tokenizer, mode="val")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=listwise_collator)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=listwise_collator)

    optimizer = AdamW([p for p in scorer.parameters() if p.requires_grad], lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if cfg.dtype == "float16" and device.type == "cuda" else None

    dump_config(cfg, Path(cfg.run_dir) / "config.json")
    log_path = Path(cfg.run_dir) / "training_log.jsonl"
    global_step = 0
    micro_step = 0
    target_steps = getattr(cfg, "total_optimizer_steps", None)
    max_epochs = getattr(cfg, "num_epochs", None) or 10**9

    for epoch in range(1, int(max_epochs) + 1):
        pbar = tqdm(train_loader, desc=f"Training epoch {epoch}")
        for batch in pbar:
            log = train_step(batch, scorer, optimizer, scaler, cfg, micro_step)
            micro_step += 1
            if log["did_step"]:
                global_step += 1
                if cfg.save_every_steps and global_step % int(cfg.save_every_steps) == 0:
                    metrics = evaluate(val_loader, scorer, cfg)
                    save_checkpoint(scorer, optimizer, cfg, f"step_{global_step}", epoch, global_step, metrics)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"epoch": epoch, "global_step": global_step, **log}) + "\n")
                if target_steps is not None and global_step >= int(target_steps):
                    metrics = evaluate(val_loader, scorer, cfg)
                    save_checkpoint(scorer, optimizer, cfg, "final", epoch, global_step, metrics)
                    return {"global_step": global_step, "metrics": metrics}
            pbar.set_postfix({"step": global_step, "loss": f"{log['loss_total']:.4f}"})

    metrics = evaluate(val_loader, scorer, cfg)
    save_checkpoint(scorer, optimizer, cfg, "final", epoch, global_step, metrics)
    return {"global_step": global_step, "metrics": metrics}

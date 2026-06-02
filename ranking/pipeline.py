from __future__ import annotations

from pathlib import Path
from typing import Any

from config import require_fields
from datasets.utils import load_jsonl, set_seed, write_json
from model import load_model_for_ranking, load_tokenizer, select_device
from training.dataset import ListwiseRankingDataset, filter_and_subsample, listwise_collator

from .scoring import MeanLogProbListwiseScorer, build_rank_record


def run_ranking_pipeline(cfg: Any) -> list[dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    require_fields(cfg, ["model_name", "data_path", "output_dir"])
    set_seed(int(cfg.seed))
    device = select_device(cfg.device)
    cfg.device = str(device)

    tokenizer = load_tokenizer(cfg)
    model = load_model_for_ranking(cfg, tokenizer, device)
    scorer = MeanLogProbListwiseScorer(model, tokenizer, cfg).to(device)

    samples = filter_and_subsample(
        load_jsonl(cfg.data_path),
        getattr(cfg, "ranking_num_samples", None),
    )
    print(f"[Ranking] samples={len(samples)}, permutations={int(getattr(cfg, 'eval_num_permutations', 1))}")
    dataset = ListwiseRankingDataset(samples, cfg, tokenizer, mode="eval")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=listwise_collator)

    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Ranking"):
            scores_list = []
            for enc, rel, perm in zip(batch["tokenized"], batch["relevance"], batch["permutations"]):
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                scores = scorer(input_ids, attention_mask)
                scores_list.append(scores)
            records.append(build_rank_record(batch, scores_list, batch["permutations"]))

    output = Path(getattr(cfg, "ranked_lists_path", Path(cfg.output_dir) / "ranked_lists.json"))
    write_json(records, output)
    return records

# Workflows

## Smoke Check

Use this before training larger models to confirm config loading, prompt construction, span extraction, attention masks, position IDs, and a tiny model ranking pass.

```bash
python scripts/smoke_check.py --config configs/dev/smoke.yaml --run-model
```

## Dataset Preparation

Prepare data:

```bash
python scripts/prepare_dataset.py --config configs/data/movielens.yaml
```

or:

```bash
python scripts/prepare_dataset.py --config configs/data/amazon_books.yaml
```

## Training

```bash
python scripts/train.py --config configs/train/train.yaml
```

## Ranking

Rank:

```bash
python scripts/rank.py --config configs/eval/rank.yaml
```

## Evaluation

Evaluate:

```bash
python scripts/evaluate.py --input runs/eval/invarirank_movielens/ranked_lists.json --output runs/eval/invarirank_movielens/metrics.json
```

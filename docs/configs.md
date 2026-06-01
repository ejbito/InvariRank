# Configs

The repository keeps dataset configs plus one sample config for dev, ranking, and training:

```text
configs/data/movielens.yaml  # build processed JSONL data
configs/data/amazon_books.yaml
configs/dev/smoke.yaml       # tiny smoke run
configs/eval/rank.yaml       # ranking run
configs/train/train.yaml     # fine-tuning run
```

Edit these files directly for your machine and experiment.

## Ranking Controls

In `configs/eval/rank.yaml`:

```yaml
data_path: ../../data/processed/movielens/25/test.jsonl
output_dir: ../../runs/eval/invarirank_movielens
adapter_path: ../../runs/train/invarirank_movielens/checkpoints/final
ranking_num_samples: 10
eval_num_permutations: 10
```

- `ranking_num_samples`: how many users/samples to rank.
- `eval_num_permutations`: how many candidate permutations per sample.

## InvariRank Setting

```yaml
attention_mask: block
position_ids: shared
```

For a standard causal baseline, change these to:

```yaml
attention_mask: causal
position_ids: standard
```

# InvariRank Research Guide

This directory contains the checkout-only research pipeline for the InvariRank paper. It owns dataset preparation,
candidate retrieval, paper baselines, generated-output parsing, evaluation metrics, and experiment orchestration. It
is intentionally separate from the reusable [`invarirank`](../invarirank/README.md) package.

Run every command below from the repository root.

## Installation and data

Install the research dependencies:

```bash
pip install -e ".[research]"
```

Add the `train` extra when training LFT/InvariRank adapters or loading PEFT artifacts:

```bash
pip install -e ".[train,research]"
```

Default raw-data locations are:

```text
data/raw/movielens/ratings.csv
data/raw/movielens/movies.csv

data/raw/amazon_books/reviews.jsonl
data/raw/amazon_books/meta.jsonl
```

Update the corresponding paths in a copied YAML configuration when your files live elsewhere. Gated Hugging Face
models require CLI authentication or an `HF_TOKEN`.

## Pipeline overview

```text
raw interactions
  -> chronological user splits
  -> LightGCN retrieval and candidate lists
  -> baseline inference or LFT/InvariRank training
  -> ranking under controlled input permutations
  -> effectiveness, robustness, validity, and efficiency metrics
```

The four stages can run independently:

```powershell
python -m research.run candidates
python -m research.run train --method invarirank
python -m research.run rank --method invarirank
python -m research.run evaluate
```

Place `--config` before the stage name when using a custom file:

```powershell
python -m research.run --config path/to/config.yaml candidates
```

## Stage 1: candidate-list generation

The default [candidate configuration](configs/candidates.yaml) creates chronological train, validation, and test
samples using MovieLens 32M and LightGCN retrieval:

```powershell
python -m research.run candidates
```

Outputs:

```text
data/processed/<dataset>/<list_size>/
|-- train.jsonl
|-- val.jsonl
`-- test.jsonl
```

Each record contains user history, retrieved candidates, graded relevance, target ranking, list length, and split.
Important controls include:

| Setting | Meaning |
| --- | --- |
| `data.dataset.name` | `movielens` or `amazon_books` |
| `data.dataset.min_user_interactions` | Minimum history required for an eligible user |
| `data.dataset.max_interactions_per_user` | Optional cap on interactions retained per user |
| `data.training.max_users` | Maximum eligible users processed |
| `data.split.history_length` | Maximum serialized recommendation history |
| `data.sampling.list_sizes` | Candidate-set sizes to produce |
| `data.retrieval.epochs` | LightGCN optimization epochs |
| `data.retrieval.edge_samples_per_epoch` | Sampled training edges per epoch |
| `data.retrieval.use_cuda` / `use_amp` | GPU and mixed-precision retrieval |

MovieLens generation selects up to three future positive items and fills each list with retrieved hard negatives and
catalogue candidates. `max_users` is an upper bound: chronological eligibility and positive-item requirements can
produce fewer output records.

Candidate loading, metadata processing, LightGCN fitting, and list sampling display progress bars. For a
resource-bounded run, reduce users, epochs, sampled edges, embedding dimension, and layers in a copied config; record
those retrieval settings with any reported experiment.

## Stage 2: framework training

LFT and InvariRank are the only methods that require training:

```powershell
python -m research.run train --method lft
python -m research.run train --method invarirank
```

Override the model or output directory without changing YAML:

```powershell
python -m research.run train `
  --method invarirank `
  --model meta-llama/Llama-3.2-3B-Instruct `
  --output-dir runs/train/invarirank_trial
```

The method preset selects the intended architecture:

| Method | Attention | Position IDs | Score |
| --- | --- | --- | --- |
| LFT | causal | standard | marker-span mean log probability |
| InvariRank | block | shared | marker-span mean log probability |

Every training and validation candidate requires a non-negative integer `relevance` label. Training supports
LambdaRank, optional permutation-consistency loss, LoRA, validation permutations, progress reporting, and
checkpointing. See [the default training config](configs/train.yaml) for optimization controls.

## Stage 3: ranking

The [ranking configuration](configs/rank.yaml) defines the model, input candidates, method settings, generation
behavior, sample limit, permutation count, and output directories.

Run generated-output baselines:

```powershell
python -m research.run rank --method zero_shot --num-samples 100 --permutations 5
python -m research.run rank --method bootstrapping --num-samples 100 --permutations 5
python -m research.run rank --method sgs --num-samples 100 --permutations 5
python -m research.run rank --method stella --num-samples 100 --permutations 5
```

Run a trained method:

```powershell
python -m research.run rank --method invarirank
python -m research.run rank --method lft
```

The runner uses the same deterministic outer permutations for every method when `ranking.seed`, sample order, and
permutation count match. Each output record preserves original candidate identity, exact input permutations, scores,
rankings, relevance, generation metadata, and efficiency data.

### Methods and computational cost

| Method | Default backend | Internal rankings per outer permutation |
| --- | --- | ---: |
| Zero-shot | `generate` | 1 |
| Bootstrapping | `generate` | `num_samples` (default 3) |
| SGS | `generate` | `ceil(candidate_count / selection_size)` |
| STELLA | `generate` | Up to `max_updates`, plus one calibration stage |
| LFT | `span_logprob` | 1 |
| InvariRank | `span_logprob` | 1 |

Zero-shot, Bootstrapping, SGS, and STELLA also support the marker-span `span_logprob` backend for controlled research
comparisons. LFT and InvariRank remain `span_logprob` only because their checkpoints are trained against that score.
SGS removes the candidates selected in each round and reproducibly shuffles the remaining candidates using its
configured `seed` before the next model call.

The CLI reports model-loading status followed by completed `user x permutation` rankings, elapsed time, throughput,
and ETA. STELLA reports calibration separately. An outer Bootstrapping, SGS, or STELLA ranking contains multiple model
calls, so its outer bar can pause while that unit is being completed.

Runtime grows multiplicatively. For example, 100 users and 10 outer permutations produce 1,000 Zero-shot calls,
3,000 default Bootstrapping calls, and 25,000 default SGS calls for 25-candidate lists. Time a small subset on the
target hardware before scheduling a large experiment.

### STELLA calibration

Without an existing matrix, STELLA probes a validation set before ranking:

```yaml
methods:
  stella:
    backend: generate
    top_one_generation: true
    batch_size: 8
    probe_path: data/processed/movielens/25/val.jsonl
    probe_samples: 100
    ensemble_steps: 5
    smoothing: 1.0
    max_updates: 10
    convergence_tolerance: 0.000001
    convergence_steps: 3
    minimum_information_gain: 0.000001
    transition_matrix_output: runs/eval/stella_movielens/transition_matrix.json
```

Set `transition_matrix_path` to the saved matrix on later runs to skip calibration. The matrix provenance is tied to
the model, backend, prompt version, dataset, and candidate count; incompatible reuse is rejected. Calibration treats
every relevant candidate in a probing sample as a target, so candidate lists with multiple positives remain valid.
The saved matrix also includes observation counts, row entropy, row similarity, and probability-range diagnostics.

At inference, STELLA updates its candidate posterior until entropy converges or `max_updates` is reached, then returns
the complete ranking induced by the single minimum-entropy posterior. Consensus across the raw internal rankings
breaks posterior ties. If the selected posterior has no meaningful information gain over a uniform distribution,
STELLA falls back to the corresponding raw model ranking instead of copying the outer input order. Result metadata
records the selected entropy and update, information gain, fallback status, diagnostics, and actual model-call count.
`batch_size` applies to the independent probing requests; inference updates remain sequential so convergence can stop
model calls early.

## Stage 4: evaluation

Evaluate one ranked-list artifact directly:

```powershell
python -m research.run evaluate `
  --ranked-lists runs/eval/zero_shot_movielens/ranked_lists.json `
  --output runs/eval/zero_shot_movielens/metrics.json `
  --top-k 5 10
```

Repeat with each method-specific directory. Evaluation displays record-level progress and produces:

| Group | Metrics | Direction |
| --- | --- | --- |
| Effectiveness | `hr@k`, `ndcg@k` | Higher is better |
| Robustness | permutation Spearman, Kendall, top-k overlap | Higher is better |
| Validity | PPI, GPI, PCR, LRI | Lower is better |
| Efficiency | passes, calls, batches, tokens, latency | Descriptive |
| Generation validity | valid/repaired/failed rates and label errors | More valid, fewer repairs/errors |

Effectiveness is averaged over user-permutation results. Robustness and validity compare permutations within each
user before aggregation. At least two permutations are required; five or more provides more useful pairwise evidence.
Repaired generated outputs remain in effectiveness results, so report output-validity statistics alongside them.

## Configurations

| Config | Purpose |
| --- | --- |
| [`configs/candidates.yaml`](configs/candidates.yaml) | Data paths, chronological splits, sampling, and LightGCN |
| [`configs/train.yaml`](configs/train.yaml) | LFT/InvariRank model, LoRA, loss, and validation |
| [`configs/rank.yaml`](configs/rank.yaml) | Baselines, adapters, generation, permutations, and outputs |
| [`configs/evaluate.yaml`](configs/evaluate.yaml) | Ranked-list input, metric output, and top-k values |
| [`configs/paper.yaml`](configs/paper.yaml) | Dataset/model/method/list-size/seed experiment matrix |

Each stage loads its matching default when `--config` is omitted. Paths resolve relative to `project_root`, and CLI
overrides take precedence. Copy defaults for new studies rather than overwriting the paper configurations.

## Prompting and output backends

LFT and InvariRank use framework-owned `[SPAN]` and `[ITEM]` markers and score each candidate from its marker-span mean
log probability. Generated baselines use the RankGPT-style contract in `research/prompts.py` and return candidate
labels in a JSON `rank_order` field.

Generated outputs are checked for missing, duplicate, and unknown labels. The configured repair policy can append
missing labels in input order, and every repair is retained in output metadata for later auditing. SGS requests only
its next selection group; STELLA can request only the top item it consumes.

## Pipeline and paper reproduction

Run all default stages together:

```powershell
python -m research.run pipeline --method invarirank
python -m research.run pipeline --method invarirank --dry-run
```

Inference-only methods skip training. The convenience pipeline uses the four default stage configs; custom configs
are supported by individual stage commands or the reproduction runner.

Run a filtered, resumable paper matrix:

```powershell
python -m research.run --config research/configs/paper.yaml reproduce `
  --datasets movielens `
  --models llama_3b `
  --methods zero_shot bootstrapping sgs stella `
  --list-sizes 25 `
  --seeds 42 `
  --stages candidates rank evaluate
```

Use `--dry-run` to inspect the matrix. Each run stores its resolved configuration and stage manifest; completed stages
resume automatically, while `--force` reruns selected work.

## Outputs

```text
data/processed/<dataset>/<list_size>/
|-- train.jsonl
|-- val.jsonl
`-- test.jsonl

runs/train/<run_name>/
|-- config.json
|-- training_log.jsonl
`-- checkpoints/final/

runs/eval/<method_name>/
|-- ranked_lists.json
|-- metrics.json
`-- transition_matrix.json       # STELLA probing

runs/paper/<experiment>/<run_id>/
|-- resolved_config.yaml
|-- status.json
|-- train/
`-- eval/
```

`ranked_lists.json` is sufficient to recompute effectiveness and permutation metrics without rerunning the model.
The reproduction root also writes `comparison.json`, joining results with dataset, model, method, list size, seed,
prompt, and configuration identity.

## Research module map

| Module | Responsibility |
| --- | --- |
| `data.py` | MovieLens/Amazon processing, chronological splits, LightGCN, candidates |
| `baselines.py` | Zero-shot, Bootstrapping, SGS, STELLA, and backend selection |
| `generation.py` | Batched generated-ranking model adapter and parsing metadata |
| `prompts.py` | RankGPT-style research prompts and label parsing |
| `evaluation.py` | Effectiveness, robustness, validity, efficiency, and generation validity |
| `run.py` | Stage CLI, progress reporting, manifests, and reproduction orchestration |

The reusable ranking contracts, InvariRank architecture, training API, and permutation suite remain in
[`invarirank/`](../invarirank/README.md).

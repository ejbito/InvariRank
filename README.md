# InvariRank: Position-Invariant Listwise Reranking for LLM-Based Recommendation

[![Paper](https://img.shields.io/badge/paper-2604.27599-red.svg)](https://arxiv.org/abs/2604.27599)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)

Code for the paper ["One Pass, Any Order: Position-Invariant Listwise Reranking for LLM-Based Recommendation"](https://arxiv.org/abs/2604.27599).

InvariRank is a listwise LLM reranking method for recommendation. It reduces candidate-order sensitivity by isolating candidate computation with a structured attention mask and giving every candidate a shared positional frame. Candidates are then scored in one model pass using their marker-span mean log probabilities.

The repository is organized as both a reusable framework and a research codebase:

- `invarirank/` contains the framework API, modeling, prompting, and training code.
- `research/` contains datasets, LightGCN candidate retrieval, paper baselines, generated-output prompting, evaluation, and experiment orchestration.

## News

- **[2026.04.30]** Paper available on [arXiv](https://arxiv.org/abs/2604.27599).
- **[2026.04.02]** Paper accepted to the SIGIR 2026 short paper track.

## Quick Example

A ranking sample contains a user history and a retrieved candidate set:

```python
sample = {
    "user_id": "u1",
    "history": [
        {
            "item_id": "h1",
            "title": "The Matrix",
            "year": 1999,
            "genres": ["Action", "Sci-Fi"],
            "rating": 5,
        },
        {
            "item_id": "h2",
            "title": "Inception",
            "year": 2010,
            "genres": ["Action", "Sci-Fi"],
            "rating": 5,
        },
        {
            "item_id": "h3",
            "title": "Arrival",
            "year": 2016,
            "genres": ["Drama", "Sci-Fi"],
            "rating": 4,
        },
    ],
    "candidates": [
        {
            "item_id": "m1",
            "title": "Interstellar",
            "year": 2014,
            "genres": ["Adventure", "Sci-Fi"],
        },
        {
            "item_id": "m2",
            "title": "The Notebook",
            "year": 2004,
            "genres": ["Drama", "Romance"],
        },
        {
            "item_id": "m3",
            "title": "Blade Runner 2049",
            "year": 2017,
            "genres": ["Drama", "Sci-Fi"],
        },
    ],
}
```

Each InvariRank candidate must have a non-empty `title` (or `name`) and a unique ID. IDs may use `item_id`, `id`,
`asin`, or `movie_id`; when omitted, the candidate's original index is used as a fallback.

A conventional listwise LLM reranker can change its output when the candidates are serialized in a different order:

```python
input_order_a = ["Interstellar", "The Notebook", "Blade Runner 2049"]
input_order_b = ["The Notebook", "Blade Runner 2049", "Interstellar"]

standard_output_a = ["Interstellar", "Blade Runner 2049", "The Notebook"]
standard_output_b = ["Blade Runner 2049", "Interstellar", "The Notebook"]
```

InvariRank is designed to produce a more stable ranking across those input permutations:

```python
invarirank_output_a = ["Interstellar", "Blade Runner 2049", "The Notebook"]
invarirank_output_b = ["Interstellar", "Blade Runner 2049", "The Notebook"]
```

Use the framework API to score a candidate set:

```python
from invarirank import InvariRankReranker, RerankerConfig

reranker = InvariRankReranker.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    adapter_path="runs/train/invarirank_movielens/checkpoints/final",
    config=RerankerConfig(device="cuda", max_length=4096),
)

result = reranker.rank(sample)

for item in result.items:
    print(item.item_id, item.score)
```

`RankingSample`, `RankedItem`, `RankingResult`, `RerankerConfig`, and `Reranker` form the typed public framework contract. Dictionary samples remain supported for straightforward integration.

## Repository Structure

```text
invarirank/
├── __init__.py          public framework API
├── framework.py         ranking contracts, configuration, and reranker facade
├── permutations.py      external adapters and controlled permutation experiments
├── modeling.py          model loading, span scoring, masks, and position IDs
├── prompts.py           marker-based InvariRank prompt construction
└── training.py          datasets, losses, validation, LoRA, and checkpoints

research/
├── baselines.py         Zero-shot, Bootstrapping, SGS, and STELLA
├── data.py              MovieLens, Amazon Books, and LightGCN retrieval
├── demo.py              small framework inference demo
├── evaluation.py        effectiveness, robustness, and validity metrics
├── generation.py        generated-ranking backend
├── prompts.py           RankGPT-style generation prompts and output parsing
├── run.py               stage runner and paper reproduction orchestration
└── configs/
    ├── candidates.yaml  candidate-list generation
    ├── train.yaml       LFT and InvariRank training
    ├── rank.yaml        baseline and fine-tuned ranking
    ├── evaluate.yaml    ranked-list evaluation
    └── paper.yaml       SIGIR 2026 experiment matrix

pyproject.toml            package metadata and dependencies
requirements.txt          complete development dependency list
```

The framework is deliberately small. Dataset preparation, baselines, generated-output parsing, paper metrics, and reproduction logic do not live inside `invarirank/`.

## Environment Setup

Use Python 3.10 or newer.

Clone the repository and enter it:

```bash
git clone https://github.com/ejbito/InvariRank.git
cd InvariRank
```

If the repository is already cloned, update it with:

```bash
git pull
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Upgrade `pip`, then install the framework in editable mode:

```bash
python -m pip install --upgrade pip
pip install -e .
```

Install optional training, research, and development dependencies when working with the complete repository:

```bash
pip install -e ".[train,research,dev]"
```

The published wheel contains only the reusable `invarirank` package. The `research/` source tree and its YAML configs
remain repository-only. To run paper experiments, work from a repository checkout, install the `research` extra, and
invoke stages with `python -m research.run ...` from the repository root. The core distribution intentionally does not
install an `invarirank-research` console command.

Alternatively, `requirements.txt` installs the complete development environment:

```bash
pip install -r requirements.txt
```

For gated Hugging Face models, authenticate with the Hugging Face CLI or provide an `HF_TOKEN` before loading the model.

## Code Quality

The repository uses Ruff for linting and formatting:

```bash
python -m ruff check .
python -m ruff format --check .
```

Apply Ruff formatting with:

```bash
python -m ruff format .
```

## Usage

The framework can be used directly from Python. The research pipeline is divided into four independently runnable stages: candidate generation, training, ranking, and evaluation.

### Framework Inference

The quickest executable example is:

```powershell
python -m research.demo `
  --model meta-llama/Llama-3.2-3B-Instruct `
  --device cuda
```

For application code, use `InvariRankReranker` as shown in the [Quick Example](#quick-example). An optional `permutation` can be supplied as candidate indices:

```python
result = reranker.rank(sample, permutation=[2, 0, 1])
```

The returned `RankingResult` records both the final ranking and the input permutation used to produce it.

Rank multiple candidate sets in padded model-forward batches:

```python
results = reranker.rank_many(samples, batch_size=8)
```

Use one optional permutation per sample when controlled input orders are already available:

```python
results = reranker.rank_many(samples, permutations=permutations, batch_size=8)
```

Save and reload the configured reranker as one InvariRank directory:

```python
reranker.save_pretrained("saved/invarirank")
reranker = InvariRankReranker.from_pretrained("saved/invarirank")
```

### Permutation Experiments

`PermutationSuite` runs domain-neutral controlled input-order experiments with InvariRank or another reranker:

```python
from invarirank import PermutationSuite

suite = PermutationSuite(reranker)
results = suite.random(sample, count=6, seed=42, batch_size=8)
```

The suite also provides `fixed(...)`, `sweep(...)`, and `templates(...)`. Results retain original candidate indices,
item IDs, input positions, scores, and exact permutations. See [`invarirank/README.md`](invarirank/README.md) for
external score and generated-order callback examples.

### Framework Training

Training uses the same sample schema as inference:

```python
from invarirank import RerankerConfig, Trainer, TrainingConfig

trainer = Trainer.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    train_samples,
    validation_samples,
    reranker_config=RerankerConfig.for_method("invarirank", {"device": "cuda"}),
    training_config=TrainingConfig(total_optimizer_steps=500),
)

trainer.train(output_dir="runs/train/my_invarirank_model")
```

The framework provides LambdaRank optimization, optional permutation-consistency loss, LoRA fine-tuning, validation over candidate permutations, and checkpointing.

### 1. Generate Candidate Lists

Edit the raw data paths and retrieval settings in `research/configs/candidates.yaml`, then run:

```powershell
python -m research.run candidates
```

Use a custom candidate config by placing `--config` before the command:

```powershell
python -m research.run --config path/to/candidates.yaml candidates
```

The stage creates chronological train, validation, and test candidate lists using MovieLens 32M or Amazon Books interactions. The default retrieval method is LightGCN.

```text
data/processed/<dataset>/<list_size>/
├── train.jsonl
├── val.jsonl
└── test.jsonl
```

Important candidate-generation controls include:

- `data.split.history_length`: maximum historical interactions included for a user.
- `data.sampling.list_sizes`: candidate-set sizes to generate.
- `data.training.max_users`: optional limit for faster experiments.
- `data.retrieval.epochs`: LightGCN training epochs.
- `data.retrieval.edge_samples_per_epoch`: sampled interaction edges per epoch.
- `data.retrieval.batch_size`: LightGCN optimization batch size.
- `data.retrieval.use_cuda` and `use_amp`: GPU and mixed-precision retrieval training.

### 2. Train LFT or InvariRank

Train InvariRank with the default training config:

```powershell
python -m research.run train --method invarirank
```

Train the listwise fine-tuning baseline:

```powershell
python -m research.run train --method lft
```

Override the model or destination without editing YAML:

```powershell
python -m research.run train `
  --method invarirank `
  --model meta-llama/Llama-3.2-3B-Instruct `
  --output-dir runs/train/invarirank_trial
```

The method preset selects the intended architecture:

| Method | Attention mask | Position IDs | Scoring |
| --- | --- | --- | --- |
| LFT | causal | standard | marker-span mean log probability |
| InvariRank | block | shared | marker-span mean log probability |

### 3. Rank Candidate Lists

Rank with a trained InvariRank adapter:

```powershell
python -m research.run rank --method invarirank
```

Run a generated-output baseline:

```powershell
python -m research.run rank `
  --method zero_shot `
  --num-samples 100 `
  --permutations 10
```

Run the same inference baseline with InvariRank-style marker-span scoring:

```powershell
python -m research.run rank `
  --method bootstrapping `
  --backend span_logprob `
  --num-samples 100 `
  --permutations 10
```

Supported research methods are:

- Zero-shot direct listwise ranking.
- Bootstrapping with permutation sampling and Borda aggregation.
- Sequential Greedy Selection (SGS).
- STELLA Bayesian position calibration and ensemble aggregation.
- Listwise fine-tuning (LFT).
- InvariRank.

### 4. Evaluate Rankings

Evaluate the ranked lists specified in `research/configs/evaluate.yaml`:

```powershell
python -m research.run evaluate
```

Evaluate a specific artifact directly:

```powershell
python -m research.run evaluate `
  --ranked-lists runs/eval/invarirank_movielens/ranked_lists.json `
  --output runs/eval/invarirank_movielens/metrics.json `
  --top-k 5 10
```

The report separates:

- Effectiveness: `hr@k` and `ndcg@k`.
- Permutation robustness: Spearman correlation, Kendall correlation, and top-k overlap.
- Ranking validity: `PPI`, `GPI`, `PCR`, and `LRI`, including standard deviations where applicable.
- Efficiency: forward passes, generation calls, generation batches, token counts, and generation latency when available.
- Generated-output validity: valid, repaired, and failed output rates plus label parsing statistics.

### Run the Four Stages Together

The pipeline command directly loads the four default stage configs and passes artifacts from one stage to the next:

```powershell
python -m research.run pipeline --method invarirank
```

Preview the pipeline without loading models or writing stage artifacts:

```powershell
python -m research.run pipeline --method invarirank --dry-run
```

Inference-only methods skip training automatically. The pipeline deliberately uses the four default configs; `--config` is intended for individual stages and is not accepted by `pipeline`.

### Reproduce Paper Experiments

`research/configs/paper.yaml` describes the full dataset, model, method, list-size, and seed matrix. A complete run is available, although most experiments can be run as filtered subsets:

```powershell
python -m research.run --config research/configs/paper.yaml reproduce `
  --datasets movielens `
  --models llama_3b `
  --methods zero_shot invarirank `
  --stages rank evaluate
```

Preview selected runs with `--dry-run`. Completed stages resume from their manifests; use `--force` to rerun selected stages.

## Configs

The repository uses one default YAML file per pipeline stage:

| Config | Purpose |
| --- | --- |
| `research/configs/candidates.yaml` | Raw paths, chronological splits, candidate sampling, and LightGCN retrieval |
| `research/configs/train.yaml` | LFT and InvariRank model, LoRA, loss, and validation settings |
| `research/configs/rank.yaml` | Method backends, adapters, generation, permutations, and ranking output paths |
| `research/configs/evaluate.yaml` | Ranked-list input, metrics output, and top-k values |
| `research/configs/paper.yaml` | Full SIGIR 2026 experiment matrix and shared stage settings |

Each stage command loads its matching config when `--config` is omitted. Paths are resolved relative to `project_root`, and command-line overrides take precedence over YAML values.

The most important ranking controls are:

```yaml
ranking:
  model_name: meta-llama/Llama-3.2-3B-Instruct
  device: cuda
  dtype: bfloat16
  data_path: data/processed/movielens/25/test.jsonl
  ranking_num_samples: 500
  eval_num_permutations: 10
  max_seq_length: 4096
```

Method-specific settings live under `methods`:

```yaml
methods:
  zero_shot:
    backend: generate
    output_dir: runs/eval/zero_shot_movielens

  invarirank:
    backend: span_logprob
    adapter_path: runs/train/invarirank_movielens/checkpoints/final
    output_dir: runs/eval/invarirank_movielens
```

Generation behavior is shared under `generation`, including deterministic decoding, beam count, chat-template use, maximum new tokens, and incomplete-output repair.

### STELLA Configuration

STELLA first estimates a position transition matrix from a probing set, then applies Bayesian calibration during ranking:

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
    aggregate_count: 3
    transition_matrix_output: runs/eval/stella_movielens/transition_matrix.json
```

`top_one_generation` requests only the item STELLA consumes rather than an unused complete ranking. `batch_size` groups independent probing and Bayesian-update requests. Reduce it if generation exceeds GPU memory.

After probing, set `transition_matrix_path` to the saved matrix to reuse it. A matrix is tied to its model, backend, prompt version, dataset, and candidate count; regenerate it when any of those change.

## Prompting and Scoring

InvariRank and LFT use structural markers around the shared context and each candidate:

```text
[SPAN]
... instruction and user history ...
[/SPAN]

[ITEM]
... candidate item ...
[/ITEM]
```

Marker insertion is enforced by `invarirank/prompts.py`. The model assigns each candidate a score from the mean log probability of the tokens within its item span.

The inference baselines use a single RankGPT-style generation contract from `research/prompts.py`. Generated responses contain candidate labels in a JSON `rank_order` field. SGS can request only its next selection group, while STELLA can request only the top item.

| Method | `generate` | `span_logprob` | Default paper backend |
| --- | ---: | ---: | --- |
| Zero-shot | Yes | Yes | `generate` |
| Bootstrapping | Yes | Yes | `generate` |
| SGS | Yes | Yes | `generate` |
| STELLA | Yes | Yes | `generate` |
| LFT | No | Yes | `span_logprob` |
| InvariRank | No | Yes | `span_logprob` |

For Zero-shot, Bootstrapping, SGS, and STELLA:

- `generate` follows common RankGPT-style usage and parses the model's generated order.
- `span_logprob` uses the same marker-span mean-log-probability scorer as InvariRank.

LFT and InvariRank remain `span_logprob` only because their LambdaRank checkpoints are trained against those candidate-span scores. Selecting `generate` for either method fails validation rather than silently changing the method.

Generated outputs are validated against the candidate labels. Configurable repair can append missing labels in input order, and output validity statistics are recorded for later auditing.

## Architecture

The reusable framework path is:

```text
RankingSample
  -> InvariRank marker prompt
  -> tokenizer and candidate-span extraction
  -> decoder-only language model
  -> block attention + shared candidate position IDs
  -> marker-span mean log probabilities
  -> RankingResult
```

The research pipeline builds on that framework without becoming part of its core logic:

```text
raw MovieLens or Amazon Books interactions
  -> chronological user splits
  -> LightGCN retrieval and candidate-list generation
  -> train LFT/InvariRank or load an inference baseline
  -> rank multiple candidate permutations
  -> effectiveness, robustness, validity, and efficiency evaluation
```

The separation of responsibilities is:

- `invarirank/framework.py`: public data contracts and reranker facade.
- `invarirank/permutations.py`: callable external adapters and controlled permutation experiments.
- `invarirank/prompts.py`: framework-owned marker prompt.
- `invarirank/modeling.py`: span extraction, structured attention, shared position IDs, and scoring.
- `invarirank/training.py`: framework training and checkpointing.
- `research/data.py`: paper datasets and candidate retrieval.
- `research/baselines.py`: baseline algorithms and backend selection.
- `research/generation.py` and `research/prompts.py`: generated-output ranking.
- `research/evaluation.py`: paper metrics.
- `research/run.py`: independent stages and reproduction orchestration.

## Outputs

Candidate generation writes:

```text
data/processed/<dataset>/<list_size>/
├── train.jsonl
├── val.jsonl
└── test.jsonl
```

Training writes:

```text
runs/train/<run_name>/
├── config.json
├── training_log.jsonl
└── checkpoints/
    └── final/
        ├── adapter files
        └── trainer.pt
```

Ranking and evaluation write:

```text
runs/eval/<method_name>/
├── ranked_lists.json
├── metrics.json
└── transition_matrix.json   # STELLA only, when probing is run
```

`ranked_lists.json` retains aligned candidate indices, item IDs, input permutations, scores, relevance labels, method metadata, and efficiency metadata. This allows effectiveness and permutation robustness to be evaluated without rerunning the model.

Paper reproduction additionally writes one self-contained directory per experiment:

```text
runs/paper/sigir_2026/<experiment_id>/
├── resolved_config.yaml
├── status.json
├── train/
└── eval/
    ├── ranked_lists.json
    └── metrics.json
```

The reproduction root contains `comparison.json`, which combines comparable metrics with dataset, model, method, list size, seed, and configuration identity.

## Cite

```bibtex
@misc{bito2026passorderpositioninvariantlistwise,
  title={One Pass, Any Order: Position-Invariant Listwise Reranking for LLM-Based Recommendation},
  author={Ethan Bito and Yongli Ren and Estrid He},
  year={2026},
  eprint={2604.27599},
  archivePrefix={arXiv},
  primaryClass={cs.IR},
  url={https://arxiv.org/abs/2604.27599}
}
```

## License

This project is released under the [MIT License](LICENSE).

# InvariRank: Position-Invariant Listwise Reranking for LLM-Based Recommendation

[![Paper](https://img.shields.io/badge/paper-2604.27599-red.svg)](https://arxiv.org/abs/2604.27599)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)

Code for the paper ["One Pass, Any Order: Position-Invariant Listwise Reranking for LLM-Based Recommendation"](https://arxiv.org/abs/2604.27599).

InvariRank is a listwise LLM reranking method for recommendation. It reduces candidate-order sensitivity by using a structured attention mask and shared candidate position IDs, so candidates can be scored more consistently across input permutations.

## News

- **[2026.04.30]** Paper available on arXiv: https://arxiv.org/abs/2604.27599
- **[2026.04.02]** Paper accepted to the SIGIR 2026 short paper track.

## Quick Example

A recommendation sample contains a user history and candidate items:

```python
sample = {
    "user_id": "u1",
    "history": [
        {"item_id": "h1", "title": "The Matrix", "year": 1999, "genres": ["Action", "Sci-Fi"], "rating": 5},
        {"item_id": "h2", "title": "Inception", "year": 2010, "genres": ["Action", "Sci-Fi"], "rating": 5},
        {"item_id": "h3", "title": "Arrival", "year": 2016, "genres": ["Drama", "Sci-Fi"], "rating": 4},
    ],
    "candidates": [
        {"item_id": "m1", "title": "Interstellar", "year": 2014, "genres": ["Adventure", "Sci-Fi"]},
        {"item_id": "m2", "title": "The Notebook", "year": 2004, "genres": ["Drama", "Romance"]},
        {"item_id": "m3", "title": "Blade Runner 2049", "year": 2017, "genres": ["Drama", "Sci-Fi"]},
    ],
}
```

A standard listwise LLM reranker can change its output when the input order changes:

```python
input_order_a = ["Interstellar", "The Notebook", "Blade Runner 2049"]
input_order_b = ["The Notebook", "Blade Runner 2049", "Interstellar"]

standard_output_a = ["Interstellar", "Blade Runner 2049", "The Notebook"]
standard_output_b = ["Blade Runner 2049", "Interstellar", "The Notebook"]
```

InvariRank is designed to make the ranked output stable across candidate orders:

```python
invarirank_output_a = ["Interstellar", "Blade Runner 2049", "The Notebook"]
invarirank_output_b = ["Interstellar", "Blade Runner 2049", "The Notebook"]
```

## Repository Structure

```text
configs/               minimal example configs for data, training, and ranking
datasets/              raw dataset builders for MovieLens and Amazon Books
model/                 model loading plus InvariRank mask/position logic
prompts/               prompt builders and JSON wording templates
ranking/               ranking pipeline and mean-log-prob scoring
retriever/             LightGCN retrieval for candidate generation
scripts/               command-line entry points
training/              tokenized listwise dataset, losses, metrics, training loop
config.py              YAML/JSON config loading
requirements.txt
pyproject.toml
```

Important files:

```text
datasets/base.py       shared raw-to-JSONL dataset build flow
datasets/ml_32m.py     MovieLens 32M support
datasets/books.py      Amazon Books support
training/dataset.py    processed JSONL -> prompt permutations -> tokenized batches
model/base.py          tokenizer/model/LoRA loading
model/invarirank.py    span extraction, block mask, shared position IDs
prompts/invarirank.py  InvariRank prompt builder with required markers
prompts/zeroshot.py    generic RankGPT-style prompt builder without markers
```

## Environment Setup

Use Python 3.10 or newer. Python 3.12 is recommended.

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

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For gated Hugging Face models, authenticate with the Hugging Face CLI or set `HF_TOKEN`.


## Code Quality

This repository uses Ruff for lightweight linting and formatting:

```bash
python -m ruff check .
python -m ruff format .
```

## Usage

The repository includes three minimal example configs in `configs/`. Edit paths, model names, sample counts, and training settings before running larger experiments.

### 1. Build Dataset

Build processed train/validation/test JSONL files from MovieLens or Amazon Books:

```powershell
python scripts/build_dataset.py --config configs/dataset.yaml
```

The dataset config should define raw file paths, output paths, split settings, sampling settings, and retrieval settings. The script writes:

```text
data/processed/<dataset>/<list_size>/
|-- train.jsonl
|-- val.jsonl
`-- test.jsonl
```

### 2. Train Model

Train the reranker from processed JSONL files:

```powershell
python scripts/train_model.py --config configs/train.yaml
```

For InvariRank, use:

```yaml
attention_mask: block
position_ids: shared
prompt_style: invarirank
```

The training pipeline uses `training/dataset.py` to create candidate permutations, build prompts, tokenize them, and pass them to the model.

### 3. Run Ranking

Run ranking on processed JSONL files:

```powershell
python scripts/run_ranking.py --config configs/rank.yaml
```

Ranking writes:

```text
runs/eval/<run_name>/
`-- ranked_lists.json
```

### 4. Evaluate Ranking

Evaluate a ranked lists file and compute ranking metrics:

```powershell
python scripts/evaluate_ranking.py --config configs/rank.yaml
```

Or point directly at a ranked list JSON file:

```powershell
python scripts/evaluate_ranking.py --ranked-lists runs/eval/invarirank_movielens/ranked_lists.json
```

The output separates effectiveness and robustness:

- `effectiveness.hr@k`, `effectiveness.ndcg@k`
- `robustness.permutation_spearman`, `robustness.permutation_kendall`
- `robustness.permutation_topk_overlap@k`

## Configs

```text
configs/dataset.yaml    build MovieLens processed JSONL data
configs/train.yaml      fine-tune an InvariRank reranker
configs/rank.yaml       run ranking with a trained adapter
```

The configs are intentionally small. They are meant to show the expected fields and should be edited for your local dataset paths, model choice, output directories, and experiment size.

## Prompt Styles

Prompt construction is handled by the `prompts/` package.

InvariRank prompts enforce the structural markers required by the method:

```text
[SPAN]
... user history and instruction ...
[/SPAN]

[ITEM]
... candidate item ...
[/ITEM]
```

The wording is controlled by JSON templates in `prompts/templates/`, but marker insertion is enforced in Python so InvariRank cannot accidentally be run without the required span and item markers.

Available prompt builders:

- `prompt_style: invarirank`: InvariRank prompt with `[SPAN]` and `[ITEM]` markers.
- `prompt_style: rankgpt` or `prompt_style: zeroshot`: generic listwise prompt without InvariRank markers.

Template files:

```text
prompts/templates/invarirank.json
prompts/templates/rankgpt.json
prompts/templates/simple.json
```

## Architecture

The current workflow is:

```text
raw interactions
  -> datasets/ + retriever/
  -> train/val/test JSONL
  -> training/dataset.py creates prompt permutations
  -> prompts/ builds method-specific prompts
  -> model/ applies InvariRank mask and position logic
  -> training/ fine-tunes the model
  -> ranking/ scores candidate lists with mean log probability
```

The mean-log-probability scorer lives in `ranking/scoring.py` and is intentionally method-agnostic. Future baselines can reuse the same scorer with different prompt styles or model settings.

## Outputs

Training runs typically write:

```text
runs/train/<run_name>/
|-- config.json
|-- training_log.jsonl
`-- checkpoints/
```

Ranking runs write:

```text
runs/eval/<run_name>/
`-- ranked_lists.json
```

## Cite

```latex
@misc{bito2026passorderpositioninvariantlistwise,
      title={One Pass, Any Order: Position-Invariant Listwise Reranking for LLM-Based Recommendation},
      author={Ethan Bito and Yongli Ren and Estrid He},
      year={2026},
      eprint={2604.27599},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2604.27599},
}
```

## License

This project is released under the [MIT License](LICENSE).

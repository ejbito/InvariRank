# InvariRank: Position-Invariant Listwise Reranking for LLM-Based Recommendation

[![Paper](https://img.shields.io/badge/paper-2604.27599-red.svg)](https://arxiv.org/abs/2604.27599)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)

Code for paper "[One Pass, Any Order: Position-Invariant Listwise Reranking for LLM-Based Recommendation](https://arxiv.org/abs/2604.27599)"

InvariRank is a listwise LLM reranking method for recommendation. It is designed to reduce candidate-order sensitivity by using a structured attention mask and shared candidate position IDs, so the model scores candidates more consistently across input permutations.

## News

- **[2026.04.30]** Our paper is now available at https://arxiv.org/abs/2604.27599
- **[2026.04.02]** Our paper has been accepted to SIGIR 2026 short paper track!

## Quick example

Below is a recommendation reranking example. A user has watched several science-fiction movies, and the system needs to rerank the candidate movies:

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

A standard listwise LLM reranker can produce different rankings when the candidate order changes:

```python
input_order_a = ["Interstellar", "The Notebook", "Blade Runner 2049"]
input_order_b = ["The Notebook", "Blade Runner 2049", "Interstellar"]

standard_output_a = ["Interstellar", "Blade Runner 2049", "The Notebook"]
standard_output_b = ["Blade Runner 2049", "Interstellar", "The Notebook"]
```

InvariRank aims to make the output stable across these candidate orders:

```python
invarirank_output_a = ["Interstellar", "Blade Runner 2049", "The Notebook"]
invarirank_output_b = ["Interstellar", "Blade Runner 2049", "The Notebook"]
```

## Python Version

Use **Python 3.10 or newer**. Python 3.12 is recommended.

## Project Structure

```text
configs/
|-- data/
|   |-- movielens.yaml
|   `-- amazon_books.yaml
|-- dev/
|   `-- smoke.yaml
|-- eval/
|   `-- rank.yaml
`-- train/
    `-- train.yaml

scripts/
|-- prepare_dataset.py
|-- smoke_check.py
|-- train.py
|-- rank.py
`-- evaluate.py

src/invarirank/
|-- data/
|-- modeling/
|-- training/
|-- ranking/
`-- evaluation/

docs/
tests/
requirements.txt
README.md
```

## Environment Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

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

## Step By Step

### 1. Dataset Preparation

The MovieLens config expects raw files at:

```text
data/raw/movielens/ratings.csv
data/raw/movielens/movies.csv
```

If your files are somewhere else, edit `configs/data/movielens.yaml`.

MovieLens:

```powershell
python scripts/prepare_dataset.py --config configs/data/movielens.yaml
```

Amazon Books:

```powershell
python scripts/prepare_dataset.py --config configs/data/amazon_books.yaml
```

The dataset scripts write processed JSONL files under the configured `paths.output_dir`, usually:

```text
data/processed/<dataset>/<list_size>/
  train.jsonl
  val.jsonl
  test.jsonl
```

If you want to test the code path before training larger models, run the [smoke check](docs/workflows.md#smoke-check).

### 2. Training

```powershell
python scripts/train.py --config configs/train/train.yaml
```

The default training config uses:

```yaml
attention_mask: block
position_ids: shared
```

This is the InvariRank setting.

### 3. Ranking

```powershell
python scripts/rank.py --config configs/eval/rank.yaml
```

`configs/eval/rank.yaml` follows `configs/train/train.yaml`: same model, same InvariRank settings, and `adapter_path` pointing to the trained checkpoint.

Set the number of samples and permutations directly in `configs/eval/rank.yaml`:

```yaml
ranking_num_samples: 10
eval_num_permutations: 10
```

### 4. Evaluation

```powershell
python scripts/evaluate.py --input runs/eval/invarirank_movielens/ranked_lists.json --output runs/eval/invarirank_movielens/metrics.json
```

The evaluator reports:

- effectiveness: `HR@5`, `HR@10`, `nDCG@5`, `nDCG@10`
- robustness: Kendall, Spearman, top-k agreement

## Main Outputs

Processed datasets:

```text
data/processed/<dataset>/<list_size>/
|-- train.jsonl
|-- val.jsonl
`-- test.jsonl
```

Training:

```text
runs/train/invarirank_movielens/
|-- config.json
|-- training_log.jsonl
`-- checkpoints/
```

Ranking and evaluation:

```text
runs/eval/invarirank_movielens/
|-- ranked_lists.json
`-- metrics.json
```

## Configs

Current configs:

```text
configs/data/movielens.yaml
configs/data/amazon_books.yaml
configs/dev/smoke.yaml
configs/eval/rank.yaml
configs/train/train.yaml
```

The main ranking controls are:

```yaml
ranking_num_samples: 10
eval_num_permutations: 10
```

To compare against a standard causal/LFT-style setting, change:

```yaml
attention_mask: causal
position_ids: standard
```

## Architecture

- `data`: builds train/validation/test JSONL files with retrieval candidates.
- `modeling`: loads models and constructs candidate spans, attention masks, and position IDs.
- `training`: fine-tunes the reranker with LambdaRank-style supervision.
- `ranking`: reranks each sample under configured candidate permutations.
- `evaluation`: reports effectiveness and robustness metrics.

The workflow is:

```text
raw interactions
  -> dataset preparation + retrieval candidates
  -> train/val/test JSONL
  -> InvariRank training
  -> multi-permutation ranking
  -> effectiveness and robustness evaluation
```

## Documentation

- [Config Reference](docs/configs.md)
- [Data Format](docs/data_format.md)
- [Ranked-List Schema](docs/ranked_lists_schema.md)
- [Metrics](docs/metrics.md)
- [Workflows](docs/workflows.md)

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

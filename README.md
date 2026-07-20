# InvariRank: Position-Invariant Listwise Reranking

[![Paper](https://img.shields.io/badge/paper-2604.27599-red.svg)](https://arxiv.org/abs/2604.27599)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)

Code for the paper ["One Pass, Any Order: Position-Invariant Listwise Reranking for LLM-Based Recommendation"](https://arxiv.org/abs/2604.27599).

InvariRank reranks a retrieved recommendation candidate set in one language-model pass while reducing sensitivity to
the order in which candidates appear. It isolates candidate computation with structured attention, gives candidates a
shared positional frame, and scores their marker spans directly.

## News

- **[2026.04.30]** Paper available on [arXiv](https://arxiv.org/abs/2604.27599).
- **[2026.04.02]** Paper accepted to the SIGIR 2026 short paper track.

## What is in this repository?

The repository has two active, deliberately separate capabilities:

| Area | Purpose | Documentation |
| --- | --- | --- |
| `invarirank/` | Reusable recommendation inference, training, serialization, and permutation experiments | [Framework guide](invarirank/README.md) |
| `research/` | Candidate generation, paper baselines, evaluation, and experiment reproduction | [Research guide](research/README.md) |

The currently implemented InvariRank workflow assumes a recommendation domain. We plan to extend the framework to
additional domains and add a dedicated position-bias analysis suite in future work.

## Installation

Use Python 3.10 or newer. From a repository checkout:

```bash
python -m venv .venv
python -m pip install --upgrade pip
pip install -e .
```

Install optional dependencies only when needed:

```bash
pip install -e ".[train]"           # LoRA training
pip install -e ".[research]"        # repository research tools
pip install -e ".[train,research]"  # complete experiment workflow
pip install -e ".[dev]"             # tests and Ruff
```

The installable wheel contains only `invarirank`. Research commands and YAML configurations require a repository
checkout. Gated Hugging Face models also require CLI authentication or an `HF_TOKEN`.

## Minimal framework example

```python
from invarirank import InvariRankReranker, RerankerConfig

sample = {
    "user_id": "u1",
    "history": [{"item_id": "h1", "title": "The Matrix", "rating": 5}],
    "candidates": [
        {"item_id": "m1", "title": "Interstellar"},
        {"item_id": "m2", "title": "The Notebook"},
        {"item_id": "m3", "title": "Blade Runner 2049"},
    ],
}

reranker = InvariRankReranker.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    adapter_path="path/to/invarirank-adapter",
    config=RerankerConfig(device="cuda", max_length=4096),
)

result = reranker.rank(sample)
for item in result.items:
    print(item.item_id, item.score)
```

This adapter example requires the `train` extra. Omit `adapter_path` when loading an unadapted causal model.

The framework also supports padded batched inference, controlled permutations, external score/order callbacks,
LambdaRank training, and complete save/reload operations. See the [framework guide](invarirank/README.md) for the
input contract and full API.

## Research workflow

Research stages are independently runnable from the repository root:

```powershell
python -m research.run candidates
python -m research.run rank --method zero_shot --num-samples 100 --permutations 5
python -m research.run evaluate `
  --ranked-lists runs/eval/zero_shot_movielens/ranked_lists.json `
  --output runs/eval/zero_shot_movielens/metrics.json
```

The research implementation includes Zero-shot, Bootstrapping, SGS, STELLA, LFT, and InvariRank; MovieLens 32M and
Amazon Books processing; LightGCN retrieval; deterministic permutation runs; paper metrics; progress reporting; and a
resumable reproduction matrix. Dataset setup, method costs, configurations, outputs, and complete commands are in the
[research guide](research/README.md).

## Repository layout

```text
invarirank/             reusable Python framework
research/               checkout-only research pipeline
research/configs/       stage and reproduction configurations
pyproject.toml          package metadata and dependencies
```

## Development

```bash
python -m ruff check .
python -m ruff format --check invarirank
```

## Citation

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

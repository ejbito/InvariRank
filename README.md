<div align="center">

<h1>InvariRank</h1>

<!-- <h3>Position-Invariant and Position-Bias-Aware LLM Reranking for Recommendation</h3> -->

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#-license)
<br>
[![SIGIR Paper](https://img.shields.io/badge/SIGIR%20'26-ACM-2f4f4f.svg)](https://dl.acm.org/doi/abs/10.1145/3805712.3809952)
[![SIGIR arXiv](https://img.shields.io/badge/SIGIR%20'26-arXiv%3A2604.27599-b31b1b.svg)](https://arxiv.org/abs/2604.27599)
[![RecSys arXiv](https://img.shields.io/badge/RecSys%20'26-arXiv%3A2608.03091-b31b1b.svg)](https://arxiv.org/abs/2608.03091)

</div>

This repository contains reusable package code and experiment pipelines for studying position-invariant and
position-bias-aware listwise LLM reranking for recommendation. The reusable `invarirank` package is kept separate from
the `experiments` workflow used for paper-style two-stage recommendation experiments.

## 📄 Papers

> **One Pass, Any Order: Position-Invariant Listwise Reranking for LLM-Based Recommendation**  
> Ethan Bito, Yongli Ren, and Estrid He.<br>
> *Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval
> (SIGIR '26).*  
> [ACM](https://dl.acm.org/doi/abs/10.1145/3805712.3809952) | [arXiv](https://arxiv.org/abs/2604.27599)

> **Position Bias Undermines Preference Consistency in Listwise LLM-Based Reranking**  
> Ethan Bito, Yongli Ren, and Estrid He.<br>
> *Proceedings of the 20th ACM Conference on Recommender Systems (RecSys '26).*  
> [arXiv](https://arxiv.org/abs/2608.03091)

## ✨ Highlights

- **Reusable package API.** `invarirank/` provides reusable reranking contracts, configuration, inference, training,
  serialization, and controlled input-order experiments.
- **Experiment-first workflow.** `experiments/` contains the repo-only pipeline for dataset preparation, retrieval,
  candidate export, reranker training/inference, and evaluation.
- **Multiple reranking methods.** The experiment layer supports zero-shot reranking, bootstrapping, STELLA, SGS, and
  InvariRank-style marker scoring.
- **Position-bias analysis.** Reranker evaluation includes preference consistency metrics, with separate analysis for
  marginal top-k input-position exposure.
- **Two-stage recommendation setup.** The scripts follow the retrieval-to-reranking workflow used across the papers.

## 🧭 Repository Guide

The repository has two deliberately separate parts:

| Area | Purpose | Documentation |
| --- | --- | --- |
| `invarirank/` | Importable Python package for reusable reranking contracts, InvariRank inference/training, serialization, and controlled input-order experiments. | [Framework guide](invarirank/README.md) |
| `experiments/` | Repo-only experiment workflow around the package: dataset preparation, retrieval, candidate export, paper rerankers, and metrics. | [Experiments guide](experiments/README.md) |

Use `invarirank/` when you want a package API. Use `experiments/` when you want to reproduce or extend the paper-style
two-stage recommendation pipeline.

## 🛠️ Installation

Use Python 3.10 or newer. From a repository checkout:

```bash
python -m venv .venv
python -m pip install --upgrade pip
pip install -e .
```

Optional extras:

```bash
pip install -e ".[train]"
pip install -e ".[experiments]"
pip install -e ".[train,experiments]"
```

The `train` extra is for LoRA fine-tuning and PEFT adapter loading. The `experiments` extra is for dataset processing,
retrieval, candidate export, paper rerankers, and evaluation. Gated Hugging Face models require CLI authentication or
an `HF_TOKEN`.

## 🚀 Experiment Flow

The experiment workflow is:

```text
prepare dataset
-> train retriever
-> evaluate retriever
-> export candidates
-> train reranker
-> run reranker
-> evaluate reranker
-> analyze position bias
```

`train_reranker` is optional. Use it only when creating a fine-tuned InvariRank marker adapter. Zero-shot,
bootstrapping, STELLA, and SGS runs can go directly from candidate export to reranker inference.
Position-bias analysis is a separate optional stage for marginal top-k input-position exposure.

The [experiments guide](experiments/README.md) is the command reference for this workflow. It shows each script stage
with every available option, including boolean flags and path overrides.

## 🔁 Reranking Methods

The experiment reranking layer exposes four methods:

```text
zero_shot
bootstrapping
stella
sgs
```

Supported prompt/scoring pairs:

| Prompt | Scoring | Use |
| --- | --- | --- |
| `rankgpt` | `generation` | JSON generated-output ranking. |
| `marker` | `marker_logprob` | InvariRank marker prompt with marker-span log-probability scoring. |

Fine-tuned LFT and InvariRank adapters are loaded with `--adapter-path` and marker scoring. Saved framework adapters
restore their tokenizer assets and architecture configuration automatically.

## 📦 Package Example

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

See the [framework guide](invarirank/README.md) for the package API and the [experiments guide](experiments/README.md)
for the full pipeline.

## 📁 Repository Layout

```text
invarirank/             reusable Python package
experiments/            repo-only experiment pipeline
experiments/configs/    dataset, retriever, and reranker defaults
requirements.txt        direct dependency list
pyproject.toml          package metadata and optional extras
```

## 📝 Citation

Please cite the relevant paper when using this repository.

```bibtex
@inproceedings{10.1145/3805712.3809952,
author = {Bito, Ethan and Ren, Yongli and He, Estrid},
title = {One Pass, Any Order: Position-Invariant Listwise Reranking for LLM-Based Recommendation},
year = {2026},
isbn = {9798400725999},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3805712.3809952},
doi = {10.1145/3805712.3809952},
booktitle = {Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {3625-3629},
numpages = {5},
keywords = {recommender systems, large language models, listwise ranking, permutation invariance, position bias},
location = {Australia},
series = {SIGIR '26}
}
```

The RecSys citation will be updated to the official ACM citation when it becomes available.

```bibtex
@misc{bito2026positionbiasunderminespreference,
title={Position Bias Undermines Preference Consistency in Listwise LLM-Based Reranking}, 
author={Ethan Bito and Yongli Ren and Estrid He},
year={2026},
eprint={2608.03091},
archivePrefix={arXiv},
primaryClass={cs.IR},
url={https://arxiv.org/abs/2608.03091},
}
```

## 📜 License

This project is released under the [MIT License](LICENSE).

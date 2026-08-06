# InvariRank Experiments Pipeline

[Repository overview](../README.md) | [Framework guide](../invarirank/README.md)

The `experiments/` directory owns the end-to-end recommendation experiment workflow around the reusable
`invarirank/` package. It prepares datasets, trains and evaluates retrievers, exports reranker candidates, trains an
optional InvariRank adapter, runs paper reranking methods, and evaluates reranked outputs.

This directory is repo-only experiment code. It is not included in the installable package surface declared in
`pyproject.toml`.

## Install

From the repository root:

```bash
pip install -e ".[train,experiments]"
```

The `experiments` extra includes dataset and retrieval dependencies. The `train` extra adds PEFT/LoRA dependencies
for fine-tuning adapters. Gated Hugging Face models require CLI authentication or an `HF_TOKEN`.

## Layout

| Area | Purpose |
| --- | --- |
| `data/` | Dataset preparation, interaction validation/loading, mappings, sparse matrices, item metadata, and candidate JSON. |
| `retrieval/` | Retrieval models: implicit ALS and LightGCN. |
| `reranking/` | Paper reranking methods, prompt builders, scoring backends, parsers, and permutation utilities. |
| `evaluation/` | Retrieval metrics, reranking metrics, PPI/GPI preference consistency, parser quality, and position-bias analysis. |
| `scripts/` | Command-line stages for the experiment workflow. |
| `configs/` | Dataset, retriever, and reranker defaults. |
| `utils/` | Small IO, progress, and seed helpers. |

## Pipeline

Run commands from the repository root. The intended stage order is:

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

`train_reranker` is optional. Use it when creating a fine-tuned InvariRank marker adapter. Plain zero-shot,
bootstrapping, STELLA, and SGS runs can go directly from `export_candidates` to `run_reranker`.

`run_retriever` is intentionally not part of the workflow. `evaluate_retriever` and `export_candidates` load the
trained retriever artifact and compute recommendations in memory.

## Stage 1: Prepare Dataset

Configured dataset names live in `experiments/configs/datasets.yaml`:

```text
demo_movielens
movielens
amazon_movies
amazon_books
```

Complete option template:

```powershell
python -m experiments.scripts.prepare_datasets `
  --config experiments/configs/datasets.yaml `
  --dataset movielens `
  --raw-dir data/raw/movielens `
  --processed-dir data/processed/movielens `
  --reviews-file ratings.csv `
  --metadata-file movies.csv `
  --min-rating 4.0 `
  --min-user-interactions 5 `
  --split-strategy temporal_ratio `
  --train-ratio 0.7 `
  --val-ratio 0.1 `
  --no-progress
```

Options:

| Option | Default | Notes |
| --- | --- | --- |
| `--config` | `experiments/configs/datasets.yaml` | Dataset config file. |
| `--dataset` | `movielens` | Config key to prepare. |
| `--raw-dir` | config value | Override raw input directory. |
| `--processed-dir` | config value | Override processed output directory. |
| `--reviews-file` | config value | Amazon review filename/path override. Ignored by MovieLens. |
| `--metadata-file` | config value | Amazon metadata filename/path override. Ignored by MovieLens. |
| `--min-rating` | config value | Minimum rating treated as positive for retrieval and relevance labels; lower-rated interactions remain available to histories. |
| `--min-user-interactions` | config value | Minimum interactions required per user. |
| `--split-strategy` | config value | `temporal_ratio` or `leave_one_out`. |
| `--train-ratio` | config value | Per-user chronological training fraction. |
| `--val-ratio` | config value | Per-user chronological validation fraction. |
| `--no-progress` | off | Disable progress bars. |

Outputs are written under the configured or overridden processed directory:

```text
train.csv
retriever_train.csv
train_queries.csv
val.csv
test.csv
items.csv
user_mapping.json
item_mapping.json
stats.json
```

`retriever_train.csv` is the chronological context used to fit the retriever. `train_queries.csv` reserves the final
eligible training interaction per user for reranker supervision. Validation histories use all of `train.csv`, and test
histories use `train.csv` plus `val.csv`; user-specific ratings remain in those histories.

## Stage 2: Train Retriever

Configured retrievers live in `experiments/configs/retrievers.yaml`:

```text
implicit_als
lightgcn
```

Complete option template:

```powershell
python -m experiments.scripts.train_retriever `
  --dataset movielens `
  --dataset-config experiments/configs/datasets.yaml `
  --retriever-config experiments/configs/retrievers.yaml `
  --retriever implicit_als `
  --processed-dir data/processed/movielens `
  --artifact-dir artifacts/retrievers `
  --no-progress
```

Options:

| Option | Default | Notes |
| --- | --- | --- |
| `--dataset` | `movielens` | Dataset config key. |
| `--dataset-config` | `experiments/configs/datasets.yaml` | Dataset config file. |
| `--retriever-config` | `experiments/configs/retrievers.yaml` | Retriever config file. |
| `--retriever` | required | `implicit_als` or `lightgcn`. |
| `--processed-dir` | config value | Processed dataset directory override. |
| `--artifact-dir` | `artifacts/retrievers` | Root directory for retriever artifacts. |
| `--no-progress` | off | Disable progress bars. |

Default outputs:

```text
artifacts/retrievers/{dataset}/{retriever}/model.pkl
artifacts/retrievers/{dataset}/{retriever}/training_stats.json
```

## Stage 3: Evaluate Retriever

Complete option template:

```powershell
python -m experiments.scripts.evaluate_retriever `
  --dataset movielens `
  --dataset-config experiments/configs/datasets.yaml `
  --retriever implicit_als `
  --processed-dir data/processed/movielens `
  --artifact-dir artifacts/retrievers `
  --split test `
  --k 100 `
  --max-users 100 `
  --sample-users `
  --user-sample-seed 42 `
  --output artifacts/metrics/retrieval/movielens/implicit_als/test_k100_users100.json `
  --no-progress
```

Options:

| Option | Default | Notes |
| --- | --- | --- |
| `--dataset` | `movielens` | Dataset config key. |
| `--dataset-config` | `experiments/configs/datasets.yaml` | Dataset config file. |
| `--retriever` | required | Trained retriever name. |
| `--processed-dir` | config value | Processed dataset directory override. |
| `--artifact-dir` | `artifacts/retrievers` | Root directory containing retriever artifacts. |
| `--split` | `test` | `train`, `val`, or `test`. |
| `--k` | `100` | Evaluation cutoff. |
| `--max-users` | all users | Limit evaluated users. |
| `--sample-users` | off | Randomly sample users instead of taking the first users. |
| `--user-sample-seed` | `42` | User sampling seed. |
| `--output` | derived path | Metrics JSON path override. |
| `--no-progress` | off | Disable progress bars. |

Default metrics path:

```text
artifacts/metrics/retrieval/{dataset}/{retriever}/{split}_k{k}.json
```

## Stage 4: Export Candidates

Complete option template:

```powershell
python -m experiments.scripts.export_candidates `
  --dataset movielens `
  --dataset-config experiments/configs/datasets.yaml `
  --retriever implicit_als `
  --processed-dir data/processed/movielens `
  --artifact-dir artifacts/retrievers `
  --split test `
  --k 25 `
  --max-users 100 `
  --sample-users `
  --user-sample-seed 42 `
  --candidate-batch-size 1000 `
  --output artifacts/candidates/movielens/implicit_als/test_k10_users100.json `
  --require-ground-truth-in-candidates `
  --no-progress
```

Options:

| Option | Default | Notes |
| --- | --- | --- |
| `--dataset` | `movielens` | Dataset config key. |
| `--dataset-config` | `experiments/configs/datasets.yaml` | Dataset config file. |
| `--retriever` | required | Trained retriever name. |
| `--processed-dir` | config value | Processed dataset directory override. |
| `--artifact-dir` | `artifacts/retrievers` | Root directory containing retriever artifacts. |
| `--split` | `test` | `train`, `val`, or `test`. Training uses `train_queries.csv`. |
| `--k` | `25` | Number of candidates per user. |
| `--max-users` | all users | Limit exported users. |
| `--sample-users` | off | Randomly sample users instead of taking the first users. |
| `--user-sample-seed` | `42` | User sampling seed. |
| `--candidate-batch-size` | `1000` | Batch size when scanning for users with ground-truth coverage. |
| `--output` | derived path | Candidate JSON path override. |
| `--require-ground-truth-in-candidates` | on | Keep only users whose candidate list contains held-out ground truth; positives are never injected. |
| `--allow-missing-ground-truth` | off | Include users whose retrieved list does not contain ground truth. |
| `--no-progress` | off | Disable progress bars. |

Default candidate paths:

```text
artifacts/candidates/{dataset}/{retriever}/{split}_k{k}.json
artifacts/candidates/{dataset}/{retriever}/{split}_k{k}_users{n}.json
artifacts/candidates/{dataset}/{retriever}/{split}_k{k}_users{n}_gt_only.json
```

Candidate JSON is keyed by user ID as a string. Each user's `candidates` object is keyed by retrieval rank as a
string, and candidate records always include `item_id` plus any available metadata from `items.csv`.

## Stage 5: Train Reranker

This stage trains either an LFT (`causal`/`standard`) or InvariRank (`block`/`shared`) marker adapter from candidate JSON files with relevance labels.
Use the validation and training candidate artifacts exported from Stage 4.

Complete option template:

```powershell
python -m experiments.scripts.train_reranker `
  --method invarirank `
  --train-candidates artifacts/candidates/movielens/implicit_als/train_k25_users100_gt_only.json `
  --val-candidates artifacts/candidates/movielens/implicit_als/val_k25_users100_gt_only.json `
  --dataset movielens `
  --dataset-config experiments/configs/datasets.yaml `
  --processed-dir data/processed/movielens `
  --model-name-or-path meta-llama/Llama-3.2-3B-Instruct `
  --output-dir artifacts/rerankers/movielens/invarirank/invarirank_marker `
  --device cuda `
  --torch-dtype bfloat16 `
  --max-length 4096 `
  --total-optimizer-steps 500 `
  --learning-rate 5e-5 `
  --gradient-accumulation-steps 16 `
  --train-num-permutations 1 `
  --eval-num-permutations 10 `
  --max-history-items 20
```

Options:

| Option | Default | Notes |
| --- | --- | --- |
| `--method` | `invarirank` | `lft` or `invarirank`; the architecture is saved with the adapter. |
| `--train-candidates` | required | Candidate JSON for training. |
| `--val-candidates` | required | Candidate JSON for validation. |
| `--dataset` | `movielens` | Dataset config key. |
| `--dataset-config` | `experiments/configs/datasets.yaml` | Dataset config file. |
| `--processed-dir` | config value | Processed dataset directory override. |
| `--model-name-or-path` | `meta-llama/Llama-3.2-3B-Instruct` | Base model or local model path. |
| `--output-dir` | derived path | Training output directory override. |
| `--device` | `cuda` | Runtime device; falls back inside framework where supported. |
| `--torch-dtype` | `bfloat16` | Model dtype. |
| `--max-length` | `4096` | Prompt token limit. |
| `--total-optimizer-steps` | `500` | Stop after this many optimizer steps. |
| `--learning-rate` | `5e-5` | Optimizer learning rate. |
| `--gradient-accumulation-steps` | `16` | Micro-batches per optimizer step. |
| `--train-num-permutations` | `1` | Number of train permutations per sample. |
| `--eval-num-permutations` | `10` | Number of validation permutations per sample. |
| `--max-history-items` | `20` | Max history items loaded per user. |

Default training directory:

```text
artifacts/rerankers/{dataset}/{method}/invarirank_marker/
```

The final adapter is saved under:

```text
artifacts/rerankers/{dataset}/{method}/invarirank_marker/checkpoints/final/
```

## Stage 6: Run Reranker

Configured reranker defaults live in `experiments/configs/rerankers.yaml`. Available methods:

```text
zero_shot
bootstrapping
stella
sgs
```

Supported prompt/scoring pairs:

| Prompt | Scoring | Description |
| --- | --- | --- |
| `rankgpt` | `generation` | RankGPT-style prompt that asks the model to return JSON with `rank_order`. |
| `marker` | `marker_logprob` | InvariRank marker prompt scored with marker-span log probabilities. |

`run_reranker` calibrates STELLA automatically. It looks for a compatible cached transition matrix and, when none is
available, looks for matching validation candidates, exports them from the trained retriever when necessary, calibrates,
caches the matrix, and continues directly to test reranking:

```powershell
python -m experiments.scripts.run_reranker `
  --input artifacts/candidates/movielens/lightgcn/test_k25_gt_only.json `
  --reranker stella `
  --dataset movielens `
  --scoring marker_logprob `
  --num-permutations 10 `
  --batch-size 8
```

By default, the runner first checks the filename obtained by replacing the leading `test` in `--input` with `val`, then
searches the same directory for another validation artifact with the same retriever and candidate count. If neither
exists, it exports a ground-truth-filtered validation artifact automatically. Use `--calibration-input` to select an
explicit validation artifact. Use
`--transition-matrix-path` to choose a particular cache path; otherwise the matrix is cached under
`artifacts/calibration/{dataset}/{retriever}/`. The standalone `calibrate_stella` command remains available for
explicit calibration workflows but is not required.

Calibration places each relevant candidate at every input position, shuffles the remaining candidates, and writes a
smoothed empirical position-transition matrix. The input must have `split: val`, and
every probe user must have at least one ground-truth item among the candidates. Probe users are sampled reproducibly
with `--calibration-seed` rather than selected as an artifact-order prefix.
The calibration stage prints its total scoring-request and batch counts and displays a `Calibrating STELLA` progress
bar with an ETA unless `--no-progress` is supplied.

The calibration artifact records the model, adapter, scoring method, prompt, architecture, and candidate count used
to estimate it. `run_reranker` automatically rebuilds legacy matrices without this provenance and calibrations whose
settings do not match the inference run. Direct framework use still rejects incompatible matrices. In particular, use
the same `--scoring`, `--prompt`, and `--architecture` options for calibration and inference.

Complete option template:

```powershell
python -m experiments.scripts.run_reranker `
  --input artifacts/candidates/movielens/implicit_als/test_k10_users100.json `
  --reranker zero_shot `
  --reranker-config experiments/configs/rerankers.yaml `
  --dataset movielens `
  --dataset-config experiments/configs/datasets.yaml `
  --processed-dir data/processed/movielens `
  --output artifacts/reranking/movielens/implicit_als/test_k10_users100/zero_shot_generation.json `
  --max-users 100 `
  --max-history-items 20 `
  --model-name-or-path meta-llama/Llama-3.2-3B-Instruct `
  --adapter-path artifacts/rerankers/movielens/invarirank/invarirank_marker/checkpoints/final `
  --transition-matrix-path artifacts/reranking/movielens/implicit_als/test_k10_users100/transition_matrix.json `
  --scoring marker_logprob `
  --prompt marker `
  --architecture invarirank `
  --device cuda `
  --torch-dtype bfloat16 `
  --max-new-tokens 512 `
  --max-length 4096 `
  --temperature 0.0 `
  --top-p 1.0 `
  --parser-repair append_input_order `
  --shuffle-candidates `
  --num-permutations 5 `
  --permutation-seed 42 `
  --batch-size 8 `
  --save-prompts `
  --save-raw-responses `
  --no-progress
```

Options:

| Option | Default | Notes |
| --- | --- | --- |
| `--input` | required | Candidate JSON from Stage 4. |
| `--reranker` | required | `zero_shot`, `bootstrapping`, `stella`, or `sgs`. |
| `--reranker-config` | `experiments/configs/rerankers.yaml` | Reranker defaults file. |
| `--dataset` | `movielens` | Dataset config key. |
| `--dataset-config` | `experiments/configs/datasets.yaml` | Dataset config file. |
| `--processed-dir` | config value | Processed dataset directory override. |
| `--output` | derived path | Reranker output JSON path override. |
| `--max-users` | all users | Limit reranked users. |
| `--max-history-items` | config value | Override max history items. |
| `--model-name-or-path` | config value | Base model or local model path. |
| `--adapter-path` | none | Optional trained adapter path. |
| `--transition-matrix-path` | derived cache path | Optional explicit STELLA matrix cache path. |
| `--calibration-input` | inferred from `--input` | Validation candidates used when STELLA must build/rebuild its matrix. |
| `--calibration-max-users` | `150` | Maximum seeded validation users used for STELLA calibration. |
| `--calibration-repeats` | `5` | Shuffled negative arrangements per relevant-item position. |
| `--calibration-smoothing` | `1.0` | Additive smoothing for transition counts. |
| `--calibration-seed` | `42` | Seed for probe-user selection and calibration permutations. |
| `--calibration-candidate-batch-size` | `1000` | Retriever scan batch size when validation candidates must be exported. |
| `--retriever-artifact-dir` | `artifacts/retrievers` | Trained retriever root used for automatic validation export. |
| `--scoring` | config value | `generation` or `marker_logprob`. |
| `--prompt` | config value | `rankgpt` or `marker`. |
| `--architecture` | config value | `lft` or `invarirank`; applies to marker scoring. |
| `--device` | config value | Runtime device. |
| `--torch-dtype` | config value | Model dtype. |
| `--max-new-tokens` | config value | Generation token limit. |
| `--max-length` | config value | Prompt token limit. |
| `--temperature` | config value | Generation temperature. |
| `--top-p` | config value | Nucleus sampling value. |
| `--parser-repair` | config value | `append_input_order` or `error`. |
| `--shuffle-candidates` | off | Shuffle candidate input order before reranking. |
| `--num-permutations` | `1` | Number of shuffled candidate orders per user. |
| `--permutation-seed` | `42` | Base seed for candidate permutations. |
| `--batch-size` | `1` | Number of user/permutation reranking requests per model batch. |
| `--save-prompts` | off | Include prompt text in the output artifact. |
| `--no-save-prompts` | on | Exclude prompt text from the output artifact. |
| `--save-raw-responses` | on | Include raw model/parser responses. |
| `--no-save-raw-responses` | off | Exclude raw model/parser responses. |
| `--no-progress` | off | Disable progress bars. |

Fine-tuned adapters can be wrapped by any inference method, but marker-trained adapters must use `--scoring marker_logprob --prompt marker`. Framework checkpoints restore their saved architecture automatically.
STELLA automatically creates its transition matrix when it is absent. Unannotated `.npy` files and legacy JSON
matrices are treated as incompatible and rebuilt because their compatibility cannot be established safely.

Default reranker output path:

```text
artifacts/reranking/{dataset}/{retriever}/{candidate_file_stem}/{reranker}_{scoring}.json
artifacts/reranking/{dataset}/{retriever}/{candidate_file_stem}/{reranker}_{scoring}_perm{n}.json
```

## Stage 7: Evaluate Reranker

Complete option template:

```powershell
python -m experiments.scripts.evaluate_reranker `
  --input artifacts/reranking/movielens/implicit_als/test_k10_users100/zero_shot_generation.json `
  --k 10 `
  --output artifacts/metrics/reranking/movielens/implicit_als/test_k10_users100/zero_shot_generation_k10.json `
  --include-robustness-values `
  --preference-buckets 3 `
  --minimum-bucket-observations 1 `
  --full-output
```

Options:

| Option | Default | Notes |
| --- | --- | --- |
| `--input` | required | Reranker output JSON from Stage 6. |
| `--k` | `10` | Evaluation cutoff. |
| `--output` | derived path | Metrics JSON path override. |
| `--include-robustness-values` | off | Include raw per-user robustness arrays when `--full-output` is enabled. |
| `--preference-buckets` | `3` | Number of input-position buckets used for PPI. |
| `--minimum-bucket-observations` | `1` | Minimum observations required for a bucket-conditioned PPI probability. |
| `--full-output` | off | Include all parser diagnostics, robustness diagnostics, and per-user robustness summaries. |

Default metrics path:

```text
artifacts/metrics/reranking/{dataset}/{retriever}/{candidate_file_stem}/{reranker_output_stem}_k{k}.json
```

Default output is compact and includes effectiveness, parse/repair health, ground-truth rank stability,
Kendall/Spearman, top-k overlap, and the RecSys preference-consistency metrics `ppi` and `gpi` when permutation
outputs are present. Add `--full-output` for the complete diagnostic schema.

Compact output schema:

```json
{
  "stage": "reranking",
  "dataset": "movielens",
  "source_retriever": "implicit_als",
  "reranker": "zero_shot",
  "reranking_mode": "experiment_methods",
  "model": "meta-llama/Llama-3.2-3B-Instruct",
  "num_users": 100,
  "num_effectiveness_rows": 2000,
  "k": 10,
  "hit_rate@10": 0.0,
  "recall@10": 0.0,
  "ndcg@10": 0.0,
  "mrr@10": 0.0,
  "num_parse_errors": 0,
  "parse_error_rate": 0.0,
  "num_parser_repair_attempts": 0,
  "parser_repair_attempt_rate": 0.0,
  "mean_ground_truth_rank": 4.2,
  "mean_ground_truth_rank_variance": 1.1,
  "mean_ground_truth_rank_std": 0.9,
  "mean_user_kendall_tau": 0.82,
  "mean_user_spearman": 0.79,
  "mean_top_10_jaccard": 0.75,
  "topk_overlap@10": 0.86,
  "mean_top_10_set_agreement": 0.54,
  "num_users_with_preference_consistency": 100,
  "preference_buckets": 3,
  "minimum_bucket_observations": 1,
  "ppi": 0.123,
  "gpi": 0.045
}
```

The ground-truth rank stability, Kendall/Spearman, top-k overlap, PPI, and GPI fields are present when the reranker
artifact contains permutation outputs.

## Stage 8: Analyze Position Bias

Position-bias analysis is intentionally separate from reranker effectiveness and preference-consistency evaluation.
This script estimates marginal top-k exposure by serialized input position across permutation runs.

Complete option template:

```powershell
python -m experiments.scripts.analyze_position_bias `
  --input artifacts/reranking/movielens/implicit_als/test_k10_users100/zero_shot_generation_perm5.json `
  --k 5 `
  --output artifacts/metrics/position_bias/movielens/implicit_als/test_k10_users100/zero_shot_generation_perm5_top5.json `
  --include-values
```

Options:

| Option | Default | Notes |
| --- | --- | --- |
| `--input` | required | Reranker output JSON from Stage 6 with permutation outputs. |
| `--k` | `5` | Exposure cutoff. |
| `--output` | derived path | Position-bias metrics JSON path override. |
| `--include-values` | off | Include per-item exposure observations. |

Default position-bias metrics path:

```text
artifacts/metrics/position_bias/{dataset}/{retriever}/{candidate_file_stem}/{reranker_output_stem}_top{k}.json
```

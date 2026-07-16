# InvariRank Framework Guide

The `invarirank` package provides two active capabilities:

1. Recommendation-specific InvariRank inference and training.
2. Domain-neutral controlled input-order experiments for compatible rerankers.

Position-bias analysis is a future downstream capability. The permutation suite executes rerankers and returns aligned
evidence; it does not calculate position-bias, effectiveness, or paper-specific metrics.

The published wheel contains this reusable package only. Dataset preparation, paper baselines, evaluation, experiment
configs, and reproduction commands under `research/` are available from a repository checkout with the `research`
dependency extra installed; they are not imported or installed by the core wheel.

## Ranking Data Contract

Both capabilities use `RankingSample` and `RankingResult`. Dictionary samples are accepted and converted
automatically:

```python
sample = {
    "user_id": "u1",
    "history": [
        {"item_id": "h1", "title": "Previously selected item"},
    ],
    "candidates": [
        {"item_id": "a", "title": "Alpha", "relevance": 0},
        {"item_id": "b", "title": "Beta", "relevance": 1},
        {"item_id": "c", "title": "Gamma", "relevance": 0},
    ],
}
```

A sample must contain at least one candidate. Candidate dictionaries are preserved in results. For
`InvariRankReranker`, each candidate must have a non-empty `title` or `name`, and candidate IDs must be unique.
IDs are read from `item_id`, `id`, `asin`, or `movie_id`; candidates without an explicit ID receive their original
candidate index as a stable fallback. Order callbacks additionally require every candidate to have an explicit,
unique, non-empty ID.

The InvariRank architecture uses `user_id`, `history`, and recommendation item metadata. External callbacks may
store domain-neutral context as an additional top-level field:

```python
sample = {
    "user_id": "query-7",
    "context": {"query": "position-invariant ranking"},
    "candidates": [
        {"item_id": "doc-a", "text": "First document"},
        {"item_id": "doc-b", "text": "Second document"},
    ],
}
```

Unknown top-level fields are available to callbacks through `sample.metadata`, so this example uses
`sample.metadata["context"]`. The package does not build external prompts or parse generated outputs.

## Normal and Batched Ranking

`InvariRankReranker` receives candidates from an upstream retriever; it does not retrieve from a full catalogue.

```python
from invarirank import InvariRankReranker, RerankerConfig

reranker = InvariRankReranker.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    adapter_path="path/to/adapter",
    config=RerankerConfig(device="cuda", max_length=4096),
)

result = reranker.rank(sample)
results = reranker.rank_many(samples, batch_size=8)
```

`rank_many` preserves sample order. Candidate counts and prompt lengths may differ inside a batch. Supply one
optional candidate-index permutation per sample when input orders are already known:

```python
results = reranker.rank_many(
    samples,
    permutations=[[2, 0, 1], [1, 2, 0]],
    batch_size=8,
)
```

Every permutation must contain each original candidate index exactly once, and indices must be integers (not strings,
floats, or booleans). Invalid recommendation inputs and permutations are rejected before model inference.

## Configuration Serialization

Inference and training configurations can be converted to ordinary dictionaries or saved as JSON:

```python
from invarirank import RerankerConfig, TrainingConfig

reranker_config = RerankerConfig(device="cuda", max_length=4096)
reranker_config.save_json("configs/reranker.json")
reranker_config = RerankerConfig.from_json("configs/reranker.json")

training_config = TrainingConfig(total_optimizer_steps=500)
training_config.save_json("configs/training.json")
training_config = TrainingConfig.from_json("configs/training.json")
```

Use `to_dict()` when a configuration mapping is needed directly. Unknown configuration fields retained in `extras`
are flattened back into the serialized mapping. Loading JSON reruns normal configuration validation. These files store
configuration only.

## Saved Model Lifecycle

Save a configured reranker, then reload it without reconstructing marker, attention, position-ID, tokenizer, or
adapter settings:

```python
reranker.save_pretrained("saved/invarirank")

reranker = InvariRankReranker.from_pretrained(
    "saved/invarirank",
    config={"device": "cuda"},
)
```

The optional configuration mapping applies partial runtime overrides to the saved configuration. Supplying
`adapter_path` separately is rejected because the saved directory already owns its artifact.

Every saved directory contains:

```text
saved/invarirank/
├── invarirank_config.json
├── framework_metadata.json
├── tokenizer_config.json
├── tokenizer files
└── model or adapter files
```

For a PEFT-backed reranker, `save_pretrained` stores adapter files and records the required base-model name. Reloading
that artifact still requires the recorded base model to be locally available or downloadable. For a non-PEFT
backbone, the directory stores the complete model through its Hugging Face `save_pretrained` implementation.

`framework_metadata.json` records the format version, package version, artifact type, and base-model identity. Loading
fails early for missing tokenizer/configuration files, unsupported format versions or artifact types, invalid
framework metadata, missing adapter provenance, and base-model mismatches.

## PermutationSuite

```python
from invarirank import PermutationSuite

suite = PermutationSuite(reranker)
```

All item indices and positions are zero-based:

- `item` is an index in the original `sample["candidates"]` list.
- `position` is a position in the input order presented to the reranker.
- Every operation returns `list[RankingResult]`.
- The suite preserves original candidate identity; temporary input positions never replace original indices.

### Random permutations

```python
results = suite.random(
    sample,
    count=6,
    seed=42,
    include_identity=True,
    batch_size=8,
)
```

`random` returns `count` unique input orders. With `include_identity=True`, the original order is returned first.
Set it to `False` to exclude that order. The remaining order is deterministic for a given seed.

For `n` candidates, at most `n!` unique permutations exist, or `n! - 1` when identity is excluded. Requesting
more raises `ValueError`. For the three-candidate example above, the maximum is six.

### Fixed-position permutations

```python
results = suite.fixed(
    sample,
    item=0,
    position=1,
    count=2,
    seed=42,
    batch_size=8,
)
```

`fixed` keeps one original candidate at one required input position and randomizes the remaining candidates. It
returns `count` unique completions in deterministic seeded order. At most `(n - 1)!` completions exist.

### Complete position sweep

```python
results = suite.sweep(
    sample,
    item=0,
    repeats=2,
    seed=42,
    batch_size=8,
)
```

`sweep` places the selected item at every input position. Results are grouped by ascending position; each position
has `repeats` adjacent unique completions. The result count is `n * repeats`, and `repeats` cannot exceed
`(n - 1)!`.

For the three-candidate example, the selected item's positions are:

```python
[result.permutation.index(0) for result in results]
# [0, 0, 1, 1, 2, 2]
```

### Exact and partial templates

```python
results = suite.templates(
    sample,
    templates=[
        [0, None, None],
        [None, 0, None],
        [None, None, 0],
        [2, 0, 1],
    ],
    seed=42,
    batch_size=8,
)
```

Each template describes one input order:

- An integer fixes that original candidate at that position.
- `None` fills the position from a seeded shuffle of the unused candidates.
- Fixed indices must be unique and in range.
- A template must have exactly one entry per candidate.
- A complete template is executed exactly as supplied.
- Templates are returned in the same order they were provided.

Each partial template produces one randomized completion, not every possible completion. Repeat a template explicitly
with a different suite call or seed when multiple independently seeded completions are needed.

## External Score Reranker

`CallableReranker.from_scores` adapts a model that returns one score per presented item:

```python
from invarirank import CallableReranker, PermutationSuite


def score_items(sample, ordered_items):
    query = sample.metadata["context"]["query"]
    return [my_model.score(query, item["text"]) for item in ordered_items]


reranker = CallableReranker.from_scores(
    score_items,
    higher_is_better=True,
    method_name="my_score_model",
)

result = reranker.rank(sample, permutation=[1, 0])
results = PermutationSuite(reranker).random(sample, count=2, seed=42)
```

`ordered_items` follows the tested input permutation. The callback must return exactly one finite numeric score per
item in that same order. Higher scores rank first by default. With `higher_is_better=False`, callback values are
negated in `RankingResult` so the shared result contract remains higher-is-better.

### Batched score callback

```python
def batch_score_items(samples, ordered_item_batches):
    return [
        [
            my_model.score(sample.metadata["context"]["query"], item["text"])
            for item in ordered_items
        ]
        for sample, ordered_items in zip(samples, ordered_item_batches)
    ]


reranker = CallableReranker.from_scores(
    score_items,
    batch_score_fn=batch_score_items,
)

results = reranker.rank_many(samples, batch_size=8)
```

The batch callback is invoked once per chunk, must preserve request order, and must return one score row per request.
Without it, `rank_many` safely calls the single-item callback repeatedly.

## External Generated-Order Reranker

`CallableReranker.from_order` adapts a model or parser that returns item IDs from best to worst:

```python
def generate_order(sample, ordered_items):
    prompt = build_my_prompt(sample.metadata["context"], ordered_items)
    output = my_model.generate(prompt)
    return parse_ordered_item_ids(output)


reranker = CallableReranker.from_order(
    generate_order,
    method_name="my_generated_reranker",
)

result = reranker.rank(sample, permutation=[1, 0])
```

The callback must return every input candidate ID exactly once. Unknown, duplicate, or missing IDs raise
`ValueError`. The adapter converts output ranks to deterministic higher-is-better scores.

### Batched order callback

```python
def batch_generate_orders(samples, ordered_item_batches):
    return [
        generate_order(sample, ordered_items)
        for sample, ordered_items in zip(samples, ordered_item_batches)
    ]


reranker = CallableReranker.from_order(
    generate_order,
    batch_order_fn=batch_generate_orders,
)

results = reranker.rank_many(samples, batch_size=8)
```

Without `batch_order_fn`, batched ranking falls back to repeated `generate_order` calls.

## Pairwise Ranking

Two candidates use the normal API; no separate pairwise abstraction is required:

```python
pairwise_results = PermutationSuite(reranker).random(
    pairwise_sample,
    count=2,
    seed=42,
)
```

The two possible input orders are tested and returned as ordinary `RankingResult` objects.

## Inspecting and Exporting Results

```python
for result in results:
    print("input permutation:", result.permutation)
    for output_rank, item in enumerate(result.items):
        print(output_rank, item.candidate_index, item.item_id, item.input_position, item.score)
```

Individual results provide `to_dict()`. A collection can be exported manually:

```python
import json
from pathlib import Path

records = [result.to_dict() for result in results]
Path("permutation_results.json").write_text(
    json.dumps(records, indent=2),
    encoding="utf-8",
)
```

This is a plain result export, not the planned versioned `PermutationRun` persistence format. It does not record the
experiment mode and settings automatically. Candidate and metadata values must also be JSON-compatible.

## Validation Summary

The suite rejects:

- Empty candidate sets.
- Invalid or incomplete permutations.
- Non-positive counts, repeats, and batch sizes.
- Out-of-range item indices and positions.
- Templates with the wrong length, duplicate fixed indices, or out-of-range indices.
- Requests for more unique permutations than mathematically possible.
- Score rows with the wrong length, non-numeric values, `NaN`, or infinity.
- Batch callbacks returning the wrong number of rows.
- Order callbacks with absent input IDs or unknown, duplicate, or missing output IDs.

## Metrics and Persistence Boundary

The suite returns aligned ranking evidence. General effectiveness and stability metrics are not calculated here;
paper-specific HR, NDCG, Kendall, Spearman, top-k overlap, PPI, GPI, PCR, and LRI remain in `research/`.

Versioned experiment persistence is also not implemented yet. A future `PermutationRun` may record mode, seed,
settings, sample identity, and results for later robustness or position-bias analysis without model access.

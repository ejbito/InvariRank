# InvariRank Framework Guide

The `invarirank` package provides the recommendation-specific InvariRank architecture and a domain-neutral controlled
permutation suite.

## Recommendation Inference

`InvariRankReranker` receives recommendation candidates from an upstream retriever. It does not retrieve from a full
catalogue itself.

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

`rank_many` preserves sample order and accepts `permutations=[...]` when each sample needs a specified candidate input
order. Candidate counts and prompt lengths may differ inside a batch; prompts are padded and reconstructed per row.

## Permutation Experiments

```python
from invarirank import PermutationSuite

suite = PermutationSuite(reranker)

random_results = suite.random(sample, count=6, seed=42, batch_size=8)
fixed_results = suite.fixed(sample, item=0, position=0, count=2, seed=42)
sweep_results = suite.sweep(sample, item=0, repeats=2, seed=42)
template_results = suite.templates(
    sample,
    templates=[[0, None, None], [None, 0, None], [2, 0, 1]],
    seed=42,
)
```

Random and controlled completions are unique. Requests exceeding the number of possible unique permutations raise an
error. `random` returns the identity first by default. `sweep` returns positions in ascending order, with each
position's repeats adjacent. `templates` preserves template order; complete templates run exactly as supplied.

## External Score Reranker

```python
from invarirank import CallableReranker, PermutationSuite


def score_items(sample, ordered_items):
    return my_model.score(sample, ordered_items)


reranker = CallableReranker.from_scores(score_items)
results = PermutationSuite(reranker).random(sample, count=6, seed=42)
```

The callback receives candidates in the tested input order and returns one finite score per candidate in that same
order. Higher scores rank first. Set `higher_is_better=False` for losses or distances; returned `RankingResult` scores
are then negated so the shared result contract still uses higher-is-better scores.

An optional `batch_score_fn(samples, ordered_item_batches)` enables one external callback per chunk.

## External Generated-Order Reranker

```python
def generate_order(sample, ordered_items):
    prompt = build_my_prompt(sample, ordered_items)
    output = my_model.generate(prompt)
    return parse_ordered_item_ids(output)


reranker = CallableReranker.from_order(generate_order)
```

Order callbacks return every candidate item ID exactly once, from best to worst. Missing input IDs and unknown,
duplicate, or missing output IDs raise errors. An optional `batch_order_fn` provides batched execution.

## Result Boundary

Every operation returns `RankingResult` objects containing original candidate identity, input position, exact input
permutation, output order, score, relevance, and candidate metadata. The suite does not calculate effectiveness,
robustness, paper-validity, or position-bias metrics. Paper evaluation remains under `research/`.

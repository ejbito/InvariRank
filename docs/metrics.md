# Metrics

Evaluation output is grouped into:

- `metadata`
- `effectiveness`
- `robustness`

## Effectiveness

- `HR@5`, `HR@10`: Hit Rate at k. Higher is better.
- `nDCG@5`, `nDCG@10`: Normalized Discounted Cumulative Gain at k. Higher is better.

## Robustness

- `kendall_tau`: mean Kendall rank correlation across permutation outputs. Higher is better.
- `spearman_rho`: mean Spearman rank correlation across permutation outputs. Higher is better.
- `top5_agreement`, `top10_agreement`: top-k overlap across permutation outputs. Higher is better.

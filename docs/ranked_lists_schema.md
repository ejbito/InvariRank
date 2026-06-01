# Ranked-List Schema

Ranking outputs use one record per dataset sample. Each record contains multiple permutation runs.

```json
{
  "sample_index": 0,
  "user_id": "u1",
  "split": "test",
  "list_length": 25,
  "num_items": 25,
  "history": [],
  "candidates": [],
  "permutations": [
    {
      "permutation_index": 0,
      "input": {
        "candidate_indices": [0, 1, 2],
        "item_ids": ["a", "b", "c"],
        "relevance": [4, 0, 3]
      },
      "scores": [0.7, 0.1, 0.4],
      "output_ranking": {
        "candidate_indices": [0, 2, 1],
        "item_ids": ["a", "c", "b"],
        "scores": [0.7, 0.4, 0.1]
      }
    }
  ]
}
```

This schema is shared by:

- raw ranking outputs
- evaluation
- plotting

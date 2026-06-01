# Data Format

Processed datasets are JSONL files. Each line is one ranking sample.

```json
{
  "user_id": "u1",
  "history": [
    {
      "item_id": "h1",
      "title": "The Dark Knight",
      "year": 2008,
      "genres": ["Action", "Crime"],
      "rating": 5,
      "timestamp": 1
    }
  ],
  "candidates": [
    {
      "item_id": "m1",
      "title": "Tenet",
      "year": 2020,
      "genres": ["Action", "Sci-Fi"],
      "relevance": 4
    }
  ],
  "target_ranking": {
    "item_ids": ["m1"],
    "relevance": [4]
  },
  "list_length": 25,
  "split": "test"
}
```

Required fields:

- `user_id`
- `history`
- `candidates`
- `candidates[*].item_id`
- `candidates[*].relevance`
- `list_length`
- `split`

Recommended display fields:

- `title`
- `year`
- `genres`
- `rating`
- `timestamp`


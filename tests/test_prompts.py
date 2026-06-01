from types import SimpleNamespace

from invarirank.data import build_prompt, extract_relevance_labels


def test_build_prompt_contains_markers_and_permutation():
    cfg = SimpleNamespace(
        span_start_token="[SPAN]",
        span_end_token="[/SPAN]",
        item_start_token="[ITEM]",
        item_end_token="[/ITEM]",
    )
    sample = {
        "history": [{"title": "A", "year": 2000, "rating": 5, "genres": ["Drama"]}],
        "candidates": [
            {"title": "First", "relevance": 1},
            {"title": "Second", "relevance": 0},
        ],
    }
    prompt = build_prompt(sample, [1, 0], cfg)
    assert "[SPAN]" in prompt
    assert prompt.index("Second") < prompt.index("First")
    assert extract_relevance_labels(sample, [1, 0]) == [0, 1]

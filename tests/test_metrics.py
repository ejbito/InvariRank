from invarirank.evaluation import (
    analyse_record_effectiveness,
    analyse_record_listwise,
    evaluate_ranked_list_records,
    hr_at_k,
    kendall_tau_from_rank_maps,
    ndcg_at_k,
    spearman_rho_from_rank_maps,
    topk_agreement_from_rank_maps,
)


def test_hr_and_ndcg_at_k():
    scores = [0.1, 0.9, 0.2]
    relevance = [0, 2, 1]
    assert hr_at_k(scores, relevance, 1) == 1.0
    assert ndcg_at_k(scores, relevance, 3) == 1.0


def test_rank_correlations_identical_and_reversed():
    a = {"a": 0, "b": 1, "c": 2}
    b = {"a": 0, "b": 1, "c": 2}
    c = {"a": 2, "b": 1, "c": 0}
    assert spearman_rho_from_rank_maps(a, b) == 1.0
    assert kendall_tau_from_rank_maps(a, b) == 1.0
    assert spearman_rho_from_rank_maps(a, c) == -1.0
    assert kendall_tau_from_rank_maps(a, c) == -1.0


def test_topk_agreement():
    a = {"a": 0, "b": 1, "c": 2}
    b = {"b": 0, "a": 1, "c": 2}
    assert topk_agreement_from_rank_maps(a, b, k=2) == 1.0


def test_notebook_style_record_evaluation_identical_permutations():
    record = {
        "sample_index": 0,
        "user_id": "u1",
        "list_length": 3,
        "num_items": 3,
        "candidates": [
            {"item_id": "a", "relevance": 2},
            {"item_id": "b", "relevance": 1},
            {"item_id": "c", "relevance": 0},
        ],
        "permutations": [
            {
                "permutation_index": 0,
                "input": {"candidate_indices": [0, 1, 2], "item_ids": ["a", "b", "c"], "relevance": [2, 1, 0]},
                "scores": [3.0, 2.0, 1.0],
                "output_ranking": {"candidate_indices": [0, 1, 2], "item_ids": ["a", "b", "c"], "scores": [3.0, 2.0, 1.0]},
            },
            {
                "permutation_index": 1,
                "input": {"candidate_indices": [2, 0, 1], "item_ids": ["c", "a", "b"], "relevance": [0, 2, 1]},
                "scores": [1.0, 3.0, 2.0],
                "output_ranking": {"candidate_indices": [0, 1, 2], "item_ids": ["a", "b", "c"], "scores": [3.0, 2.0, 1.0]},
            },
        ],
    }

    eff = analyse_record_effectiveness(record)
    lst = analyse_record_listwise(record)
    summary = evaluate_ranked_list_records([record])

    assert eff["hr@5"] == 1.0
    assert eff["ndcg@5"] == 1.0
    assert lst["perm_kendall"] == 1.0
    assert lst["perm_spearman"] == 1.0
    assert summary["effectiveness"]["HR@5"] == 1.0
    assert summary["effectiveness"]["nDCG@5"] == 1.0
    assert summary["robustness"]["kendall_tau"] == 1.0
    assert summary["robustness"]["spearman_rho"] == 1.0
    assert summary["robustness"]["top5_agreement"] == 1.0
    assert set(summary) == {"metadata", "effectiveness", "robustness"}


def test_notebook_style_record_evaluation_reversed_permutations():
    record = {
        "sample_index": 0,
        "list_length": 3,
        "num_items": 3,
        "candidates": [
            {"item_id": "a", "relevance": 2},
            {"item_id": "b", "relevance": 1},
            {"item_id": "c", "relevance": 0},
        ],
        "permutations": [
            {"input": {"candidate_indices": [0, 1, 2]}, "output_ranking": {"candidate_indices": [0, 1, 2]}},
            {"input": {"candidate_indices": [2, 1, 0]}, "output_ranking": {"candidate_indices": [2, 1, 0]}},
        ],
    }
    lst = analyse_record_listwise(record)
    assert lst["perm_kendall"] == -1.0
    assert lst["perm_spearman"] == -1.0

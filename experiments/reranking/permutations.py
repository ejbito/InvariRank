from __future__ import annotations

import random
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from experiments.reranking.parsers import candidate_item_ids


def permute_user_record(
    user_record: Mapping[str, Any],
    seed: int,
    avoid_original_order: bool = True,
) -> dict[str, Any]:
    output = deepcopy(dict(user_record))
    candidates = dict(user_record["candidates"])
    original_ids = candidate_item_ids(user_record)
    ordered_candidates = [
        dict(record)
        for _, record in sorted(candidates.items(), key=lambda item: int(item[0]))
    ]

    if len(ordered_candidates) <= 1:
        shuffled = ordered_candidates
    else:
        rng = random.Random(seed)
        shuffled = ordered_candidates.copy()
        for _ in range(20):
            rng.shuffle(shuffled)
            shuffled_ids = [str(record["item_id"]) for record in shuffled]
            if not avoid_original_order or shuffled_ids != original_ids:
                break

    output["candidates"] = {
        str(index): record
        for index, record in enumerate(shuffled, start=1)
    }
    return output

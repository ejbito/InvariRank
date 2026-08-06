from __future__ import annotations

import random
from collections.abc import Sequence


def select_user_ids(
    user_ids: Sequence[int],
    max_users: int | None = None,
    sample: bool = False,
    seed: int = 42,
) -> list[int]:
    selected = [int(user_id) for user_id in user_ids]
    if max_users is None or max_users >= len(selected):
        return selected
    if max_users < 1:
        raise ValueError("--max-users must be at least 1 when provided.")
    if sample:
        rng = random.Random(seed)
        return sorted(rng.sample(selected, max_users))
    return selected[:max_users]


def order_user_ids(
    user_ids: Sequence[int],
    sample: bool = False,
    seed: int = 42,
) -> list[int]:
    ordered = [int(user_id) for user_id in user_ids]
    if sample:
        rng = random.Random(seed)
        rng.shuffle(ordered)
    return ordered

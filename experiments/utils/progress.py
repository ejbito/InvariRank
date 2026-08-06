from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def progress(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    enabled: bool = True,
) -> Iterator[T]:
    if not enabled:
        yield from iterable
        return
    from tqdm.auto import tqdm

    yield from tqdm(iterable, desc=desc, total=total)

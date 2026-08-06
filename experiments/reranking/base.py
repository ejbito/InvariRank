from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class BaseReranker(ABC):
    @abstractmethod
    def rerank_user(
        self,
        user_record: Mapping[str, Any],
        user_history: list[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def rerank(
        self,
        users: Mapping[str, Mapping[str, Any]],
        user_histories: Mapping[int, list[Mapping[str, Any]]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        user_histories = user_histories or {}
        outputs = {}
        for user_key, user_record in users.items():
            user_id = int(user_record["user_id"])
            outputs[str(user_key)] = self.rerank_user(
                user_record,
                user_history=user_histories.get(user_id, []),
            )
        return outputs

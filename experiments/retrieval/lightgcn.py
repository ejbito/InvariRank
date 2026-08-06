from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.retrieval.base import BaseRetriever
from experiments.utils.progress import progress


class LightGCNRetriever(BaseRetriever):
    def __init__(
        self,
        embedding_dim: int = 64,
        num_layers: int = 3,
        learning_rate: float = 0.001,
        batch_size: int = 2048,
        epochs: int = 100,
        reg_weight: float = 0.0001,
        seed: int = 42,
        show_progress: bool = True,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.reg_weight = reg_weight
        self.seed = seed
        self.show_progress = show_progress
        self.user_embeddings_: np.ndarray | None = None
        self.item_embeddings_: np.ndarray | None = None
        self._positive_items: dict[int, np.ndarray] = {}

    def fit(self, interactions: pd.DataFrame) -> "LightGCNRetriever":
        interactions = self._prepare_fit(interactions)
        started_at = self._start_training_stats(interactions)
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "LightGCNRetriever requires 'torch'. "
                "Install the project normally or with: pip install -e \".[experiments]\""
            ) from exc

        torch.manual_seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        users = torch.tensor(interactions["user_id"].to_numpy(), dtype=torch.long, device=device)
        items = torch.tensor(interactions["item_id"].to_numpy(), dtype=torch.long, device=device)
        self._positive_items = {
            int(user_id): group["item_id"].astype(int).to_numpy()
            for user_id, group in interactions.groupby("user_id")
        }

        edge_index = self._build_edge_index(users, items, device, torch)
        norm_values = self._build_norm_values(edge_index, device, torch)
        adjacency = torch.sparse_coo_tensor(
            edge_index,
            norm_values,
            size=(self.n_users_ + self.n_items_, self.n_users_ + self.n_items_),
            device=device,
        ).coalesce()

        user_embedding = torch.nn.Embedding(self.n_users_, self.embedding_dim, device=device)
        item_embedding = torch.nn.Embedding(self.n_items_, self.embedding_dim, device=device)
        torch.nn.init.xavier_uniform_(user_embedding.weight)
        torch.nn.init.xavier_uniform_(item_embedding.weight)
        optimizer = torch.optim.Adam(
            list(user_embedding.parameters()) + list(item_embedding.parameters()),
            lr=self.learning_rate,
        )

        rng = np.random.default_rng(self.seed)
        train_pairs = interactions[["user_id", "item_id"]].astype(int).to_numpy()
        for epoch in progress(
            range(self.epochs),
            desc="Training LightGCN",
            total=self.epochs,
            enabled=getattr(self, "show_progress", True),
        ):
            rng.shuffle(train_pairs)
            batch_starts = range(0, len(train_pairs), self.batch_size)
            epoch_losses = []
            for start in progress(
                batch_starts,
                desc="LightGCN batches",
                total=len(batch_starts),
                enabled=False,
            ):
                batch = train_pairs[start : start + self.batch_size]
                batch_users = torch.tensor(batch[:, 0], dtype=torch.long, device=device)
                positive_items = torch.tensor(batch[:, 1], dtype=torch.long, device=device)
                negative_items = self._sample_negative_items(batch[:, 0], rng, torch, device)

                final_users, final_items = self._propagate(
                    user_embedding.weight,
                    item_embedding.weight,
                    adjacency,
                    torch,
                )
                u = final_users[batch_users]
                pos = final_items[positive_items]
                neg = final_items[negative_items]

                pos_scores = (u * pos).sum(dim=1)
                neg_scores = (u * neg).sum(dim=1)
                bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
                reg_loss = (
                    user_embedding(batch_users).norm(2).pow(2)
                    + item_embedding(positive_items).norm(2).pow(2)
                    + item_embedding(negative_items).norm(2).pow(2)
                ) / len(batch_users)
                loss = bpr_loss + self.reg_weight * reg_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))
            self.training_stats_["loss_history"].append(
                {
                    "epoch": int(epoch + 1),
                    "loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                }
            )

        with torch.no_grad():
            final_users, final_items = self._propagate(
                user_embedding.weight,
                item_embedding.weight,
                adjacency,
                torch,
            )
            self.user_embeddings_ = final_users.detach().cpu().numpy().astype("float32")
            self.item_embeddings_ = final_items.detach().cpu().numpy().astype("float32")
        self._finish_training_stats(started_at)
        return self

    def recommend(
        self,
        user_ids: list[int],
        k: int = 100,
        exclude_seen: bool = True,
    ) -> dict[int, list[int]]:
        if self.user_embeddings_ is None or self.item_embeddings_ is None:
            raise RuntimeError("LightGCNRetriever must be fitted before calling recommend.")

        recommendations = {}
        item_ids = np.arange(self.n_items_)
        for user_id in progress(
            user_ids,
            desc="LightGCN recommendations",
            total=len(user_ids),
            enabled=getattr(self, "show_progress", True),
        ):
            if not 0 <= int(user_id) < len(self.user_embeddings_):
                recommendations[int(user_id)] = []
                continue
            scores = self.item_embeddings_.dot(self.user_embeddings_[int(user_id)])
            ranked = np.lexsort((item_ids, -scores)).astype(int).tolist()
            ranked = self._filter_seen(int(user_id), ranked, exclude_seen)
            recommendations[int(user_id)] = ranked[:k]
        return recommendations

    def _build_edge_index(self, users, items, device, torch):
        item_nodes = items + self.n_users_
        sources = torch.cat([users, item_nodes])
        targets = torch.cat([item_nodes, users])
        return torch.stack([sources, targets], dim=0).to(device)

    def _build_norm_values(self, edge_index, device, torch):
        degrees = torch.bincount(
            edge_index[0],
            minlength=self.n_users_ + self.n_items_,
        ).float()
        degrees = torch.clamp(degrees, min=1.0)
        row, col = edge_index
        return torch.rsqrt(degrees[row]) * torch.rsqrt(degrees[col])

    def _propagate(self, user_weight, item_weight, adjacency, torch):
        embeddings = torch.cat([user_weight, item_weight], dim=0)
        all_embeddings = [embeddings]
        current = embeddings
        for _ in range(self.num_layers):
            current = torch.sparse.mm(adjacency, current)
            all_embeddings.append(current)
        final = torch.stack(all_embeddings, dim=0).mean(dim=0)
        return final[: self.n_users_], final[self.n_users_ :]

    def _sample_negative_items(self, batch_users, rng, torch, device):
        negatives = []
        for user_id in batch_users:
            seen = self.seen_items_.get(int(user_id), set())
            item_id = int(rng.integers(0, self.n_items_))
            while item_id in seen:
                item_id = int(rng.integers(0, self.n_items_))
            negatives.append(item_id)
        return torch.tensor(negatives, dtype=torch.long, device=device)

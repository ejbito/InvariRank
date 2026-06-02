from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from datasets.utils import cfg_get


class LightGCNRetriever:
    def __init__(self, cfg):
        self.cfg = cfg

        self.seed = int(cfg_get(cfg, "training.seed", 42))

        self.embedding_dim = int(cfg_get(cfg, "retrieval.embedding_dim", 128))
        self.num_layers = int(cfg_get(cfg, "retrieval.num_layers", 3))
        self.epochs = int(cfg_get(cfg, "retrieval.epochs", 100))
        self.learning_rate = float(cfg_get(cfg, "retrieval.learning_rate", 1e-3))
        self.reg = float(cfg_get(cfg, "retrieval.reg", 1e-5))

        self.edge_samples_per_epoch = int(cfg_get(cfg, "retrieval.edge_samples_per_epoch", 3_000_000))
        self.batch_size = int(cfg_get(cfg, "retrieval.batch_size", 8192))
        self.negatives = int(cfg_get(cfg, "retrieval.negatives_per_positive", 4))
        self.neg_rejection_max_tries = int(cfg_get(cfg, "retrieval.neg_rejection_max_tries", 10))

        self.filter_seen = bool(cfg_get(cfg, "retrieval.filter_seen", True))
        self.k_max = int(cfg_get(cfg, "retrieval.k_max", 1000))

        self.use_cuda = bool(cfg_get(cfg, "retrieval.use_cuda", True))
        self.use_amp = bool(cfg_get(cfg, "retrieval.use_amp", True))
        self.deterministic = bool(cfg_get(cfg, "retrieval.deterministic", False))

        root_det = cfg_get(cfg, "deterministic", None)
        if root_det is not None:
            self.deterministic = bool(root_det)

        if self.deterministic:
            self.use_cuda = False
            self.use_amp = False

        if self.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.edge_dropout = float(cfg_get(cfg, "retrieval.edge_dropout", 0.0))
        self.hard_negative_ratio = float(cfg_get(cfg, "retrieval.hard_negative_ratio", 0.5))
        self.hard_candidate_pool = int(cfg_get(cfg, "retrieval.hard_candidate_pool", 32))
        self.normalize_embeddings = bool(cfg_get(cfg, "retrieval.normalize_embeddings", True))

        self.np_rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        if self.deterministic:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

        self.user2idx: dict[Any, int] = {}
        self.idx2user: dict[int, Any] = {}
        self.item2idx: dict[Any, int] = {}
        self.idx2item: dict[int, Any] = {}

        self.num_users = 0
        self.num_items = 0

        self.user_seen_raw: dict[Any, set] = {}
        self.user_seen_idx: dict[int, np.ndarray] = {}
        self.user_pos_sets: dict[int, set[int]] = {}
        self.train_users: np.ndarray | None = None

        self.edge_u: np.ndarray | None = None
        self.edge_i: np.ndarray | None = None

        self.base_edge_u: np.ndarray | None = None
        self.base_edge_i: np.ndarray | None = None
        self.base_edge_w: np.ndarray | None = None

        self.norm_adj_ui: torch.Tensor | None = None
        self.norm_adj_iu: torch.Tensor | None = None

        self.user_embedding: torch.nn.Embedding | None = None
        self.item_embedding: torch.nn.Embedding | None = None

        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def fit(self, interactions: Iterable[tuple[int, int]]):
        """
        interactions: iterable of (user_id, item_id) implicit edges
        """
        interactions = list(interactions)
        if not interactions:
            raise ValueError("LightGCNRetriever requires at least one interaction")

        # Deduplicate edges to avoid overweighting repeated interactions
        interactions = list(dict.fromkeys(interactions))

        users = sorted({u for u, _ in interactions})
        items = sorted({i for _, i in interactions})

        self.user2idx = {u: idx for idx, u in enumerate(users)}
        self.idx2user = {idx: u for u, idx in self.user2idx.items()}
        self.item2idx = {it: idx for idx, it in enumerate(items)}
        self.idx2item = {idx: it for it, idx in self.item2idx.items()}

        self.num_users = len(users)
        self.num_items = len(items)

        rows = np.empty(len(interactions), dtype=np.int64)
        cols = np.empty(len(interactions), dtype=np.int64)

        user_seen_raw = defaultdict(set)
        user_pos_sets = defaultdict(set)

        edge_u = np.empty(len(interactions), dtype=np.int64)
        edge_i = np.empty(len(interactions), dtype=np.int64)

        for n, (u_raw, i_raw) in enumerate(tqdm(interactions, desc="[LightGCN] Encoding edges")):
            u = self.user2idx[u_raw]
            i = self.item2idx[i_raw]

            rows[n] = u
            cols[n] = i
            edge_u[n] = u
            edge_i[n] = i

            user_seen_raw[u_raw].add(i_raw)
            user_pos_sets[u].add(i)

        self.user_seen_raw = dict(user_seen_raw)
        self.user_pos_sets = {u: set(items) for u, items in user_pos_sets.items()}
        self.user_seen_idx = {
            self.user2idx[u_raw]: np.asarray(
                [self.item2idx[i_raw] for i_raw in seen if i_raw in self.item2idx],
                dtype=np.int64,
            )
            for u_raw, seen in self.user_seen_raw.items()
            if u_raw in self.user2idx
        }

        self.train_users = np.asarray(list(self.user_pos_sets.keys()), dtype=np.int64)

        self.edge_u = edge_u
        self.edge_i = edge_i

        mat = csr_matrix(
            (np.ones(len(interactions), dtype=np.float32), (rows, cols)),
            shape=(self.num_users, self.num_items),
        )

        self._prepare_base_graph(mat)
        self._refresh_adjacency(edge_keep_prob=1.0)
        self._init_model()
        self._train_model()
        self._materialize_embeddings()

    @torch.no_grad()
    def retrieve(self, user_id: int, k: int) -> list[int]:
        if user_id not in self.user2idx or k <= 0:
            return []
        if self.user_factors is None or self.item_factors is None:
            return []

        uidx = self.user2idx[user_id]
        limit = min(k, self.num_items, self.k_max if self.k_max else k)
        if limit <= 0:
            return []

        user_vec = self.user_factors[uidx]
        scores = self.item_factors @ user_vec

        if self.filter_seen:
            seen_idx = self.user_seen_idx.get(uidx)
            if seen_idx is not None and len(seen_idx) > 0:
                scores[seen_idx] = -np.inf

        finite_mask = np.isfinite(scores)
        if not finite_mask.any():
            return []

        limit2 = min(limit, int(finite_mask.sum()))
        if limit2 <= 0:
            return []

        idx = np.argpartition(-scores, limit2 - 1)[:limit2]
        idx = idx[np.argsort(-scores[idx])]
        return [self.idx2item[int(i)] for i in idx]

    def _prepare_base_graph(self, mat: csr_matrix):
        mat = mat.tocsr()

        user_deg = np.asarray(mat.sum(axis=1)).ravel().astype(np.float32)
        item_deg = np.asarray(mat.sum(axis=0)).ravel().astype(np.float32)
        user_deg[user_deg == 0] = 1.0
        item_deg[item_deg == 0] = 1.0

        coo = mat.tocoo()
        edge_w = 1.0 / np.sqrt(user_deg[coo.row] * item_deg[coo.col])
        edge_w = edge_w.astype(np.float32)

        self.base_edge_u = coo.row.astype(np.int64)
        self.base_edge_i = coo.col.astype(np.int64)
        self.base_edge_w = edge_w

    def _refresh_adjacency(self, edge_keep_prob: float = 1.0):
        if self.base_edge_u is None or self.base_edge_i is None or self.base_edge_w is None:
            raise RuntimeError("Base graph not prepared")

        edge_u = self.base_edge_u
        edge_i = self.base_edge_i
        edge_w = self.base_edge_w

        if edge_keep_prob < 1.0:
            keep_mask = self.np_rng.random(len(edge_u)) < edge_keep_prob
            if keep_mask.sum() == 0:
                keep_mask[self.np_rng.integers(0, len(edge_u))] = True
            edge_u = edge_u[keep_mask]
            edge_i = edge_i[keep_mask]
            edge_w = edge_w[keep_mask] / edge_keep_prob

        idx_ui = torch.tensor(np.vstack([edge_u, edge_i]), dtype=torch.int64)
        val_ui = torch.tensor(edge_w, dtype=torch.float32)
        ui = torch.sparse_coo_tensor(
            idx_ui,
            val_ui,
            (self.num_users, self.num_items),
            device=self.device,
        ).coalesce()

        idx_iu = torch.tensor(np.vstack([edge_i, edge_u]), dtype=torch.int64)
        iu = torch.sparse_coo_tensor(
            idx_iu,
            val_ui,
            (self.num_items, self.num_users),
            device=self.device,
        ).coalesce()

        # sparse.mm works reliably with COO too
        self.norm_adj_ui = ui
        self.norm_adj_iu = iu

    def _init_model(self):
        self.user_embedding = torch.nn.Embedding(self.num_users, self.embedding_dim, device=self.device)
        self.item_embedding = torch.nn.Embedding(self.num_items, self.embedding_dim, device=self.device)

        torch.nn.init.xavier_uniform_(self.user_embedding.weight)
        torch.nn.init.xavier_uniform_(self.item_embedding.weight)

    def _propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.norm_adj_ui is None or self.norm_adj_iu is None:
            raise RuntimeError("Adjacency not built")
        if self.user_embedding is None or self.item_embedding is None:
            raise RuntimeError("Embeddings not initialized")

        users0 = self.user_embedding.weight
        items0 = self.item_embedding.weight

        user_layers = [users0]
        item_layers = [items0]

        u_prev = users0
        i_prev = items0

        for _ in range(self.num_layers):
            u_next = torch.sparse.mm(self.norm_adj_ui, i_prev)
            i_next = torch.sparse.mm(self.norm_adj_iu, u_prev)
            user_layers.append(u_next)
            item_layers.append(i_next)
            u_prev, i_prev = u_next, i_next

        users = torch.stack(user_layers, dim=0).mean(dim=0)
        items = torch.stack(item_layers, dim=0).mean(dim=0)
        return users, items

    def _train_model(self):
        assert self.edge_u is not None and self.edge_i is not None
        assert self.user_embedding is not None and self.item_embedding is not None

        optimizer = torch.optim.Adam(
            list(self.user_embedding.parameters()) + list(self.item_embedding.parameters()),
            lr=self.learning_rate,
        )

        n_edges = self.edge_u.shape[0]
        pbar = tqdm(range(self.epochs), desc="[LightGCN] Training")

        for _epoch in pbar:
            edge_keep_prob = 1.0 - self.edge_dropout if self.edge_dropout > 0 else 1.0
            self._refresh_adjacency(edge_keep_prob=edge_keep_prob)

            optimizer.zero_grad(set_to_none=True)

            user_all, item_all = self._propagate()

            take = min(self.edge_samples_per_epoch, n_edges)
            idx = self.np_rng.integers(0, n_edges, size=take, dtype=np.int64)
            pos_u = self.edge_u[idx]
            pos_i = self.edge_i[idx]

            total_loss = None
            loss_epoch = 0.0
            seen_batches = 0

            for s in range(0, take, self.batch_size):
                e = min(s + self.batch_size, take)

                u_np = pos_u[s:e]
                i_np = pos_i[s:e]
                neg_np = self._sample_negatives(u_np, user_all, item_all)

                u = torch.from_numpy(u_np).to(self.device)
                pos = torch.from_numpy(i_np).to(self.device)
                neg = torch.from_numpy(neg_np).to(self.device)

                u_emb = user_all[u]  # [B, D]
                pos_emb = item_all[pos]  # [B, D]
                neg_emb = item_all[neg]  # [B, N, D]

                pos_scores = (u_emb * pos_emb).sum(dim=1, keepdim=True)  # [B, 1]
                neg_scores = (u_emb.unsqueeze(1) * neg_emb).sum(dim=2)  # [B, N]

                bpr = -F.logsigmoid(pos_scores - neg_scores).mean()

                reg_loss = self.reg * (
                    u_emb.pow(2).sum(dim=1).mean() + pos_emb.pow(2).sum(dim=1).mean() + neg_emb.pow(2).sum(dim=2).mean()
                )

                loss = bpr + reg_loss

                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss = total_loss + loss

                loss_epoch += float(loss.detach().cpu())
                seen_batches += 1

            total_loss = total_loss / max(1, seen_batches)
            total_loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss_epoch / max(1, seen_batches))

    def _sample_negatives(
        self,
        users_np: np.ndarray,
        user_all: torch.Tensor,
        item_all: torch.Tensor,
    ) -> np.ndarray:
        """
        Mixed negative sampler:
        - some hard negatives: choose highest-scoring sampled valid candidates
        - rest random negatives
        Returns shape [B, N]
        """
        batch_size = users_np.shape[0]
        n_neg = self.negatives
        if n_neg <= 0:
            raise ValueError("retrieval.negatives_per_positive must be >= 1")

        hard_count = int(round(n_neg * self.hard_negative_ratio))
        hard_count = max(0, min(n_neg, hard_count))
        rand_count = n_neg - hard_count

        out = np.empty((batch_size, n_neg), dtype=np.int64)
        col = 0

        if hard_count > 0:
            hard = self._sample_hard_negatives(
                users_np=users_np,
                user_all=user_all,
                item_all=item_all,
                n_hard=hard_count,
            )
            out[:, col : col + hard_count] = hard
            col += hard_count

        if rand_count > 0:
            rand = self._sample_random_negatives(users_np, rand_count)
            out[:, col : col + rand_count] = rand

        return out

    def _sample_random_negatives(self, users_np: np.ndarray, n_neg: int) -> np.ndarray:
        n = users_np.shape[0]
        neg = self.np_rng.integers(0, self.num_items, size=(n, n_neg), dtype=np.int64)

        for _ in range(self.neg_rejection_max_tries):
            bad = np.zeros((n, n_neg), dtype=bool)
            for r, u in enumerate(users_np):
                pos = self.user_pos_sets[int(u)]
                for c in range(n_neg):
                    if int(neg[r, c]) in pos:
                        bad[r, c] = True

            if not bad.any():
                return neg

            num_bad = int(bad.sum())
            neg[bad] = self.np_rng.integers(0, self.num_items, size=num_bad, dtype=np.int64)

        for r, u in enumerate(users_np):
            pos = self.user_pos_sets[int(u)]
            for c in range(n_neg):
                if int(neg[r, c]) in pos:
                    while True:
                        cand = int(self.np_rng.integers(0, self.num_items))
                        if cand not in pos:
                            neg[r, c] = cand
                            break

        return neg

    @torch.no_grad()
    def _sample_hard_negatives(
        self,
        users_np: np.ndarray,
        user_all: torch.Tensor,
        item_all: torch.Tensor,
        n_hard: int,
    ) -> np.ndarray:
        """
        For each user:
        1. sample candidate_pool random items
        2. filter positives
        3. choose highest-scoring valid items under current model
        """
        n = users_np.shape[0]
        cand_pool = max(self.hard_candidate_pool, n_hard)

        cand = self.np_rng.integers(0, self.num_items, size=(n, cand_pool), dtype=np.int64)
        valid = np.ones((n, cand_pool), dtype=bool)

        for r, u in enumerate(users_np):
            pos = self.user_pos_sets[int(u)]
            for c in range(cand_pool):
                if int(cand[r, c]) in pos:
                    valid[r, c] = False

        u = torch.from_numpy(users_np).to(self.device)
        cand_t = torch.from_numpy(cand).to(self.device)

        u_emb = user_all[u]  # [B, D]
        cand_emb = item_all[cand_t]  # [B, C, D]
        scores = (u_emb.unsqueeze(1) * cand_emb).sum(dim=2)  # [B, C]

        valid_t = torch.from_numpy(valid).to(self.device)
        scores = scores.masked_fill(~valid_t, float("-inf"))

        hard = np.empty((n, n_hard), dtype=np.int64)

        for r in range(n):
            row_scores = scores[r]
            finite = torch.isfinite(row_scores)
            valid_count = int(finite.sum().item())

            if valid_count >= n_hard:
                top_idx = torch.topk(row_scores, k=n_hard, dim=0).indices.detach().cpu().numpy()
                hard[r] = cand[r, top_idx]
            else:
                picked = []
                if valid_count > 0:
                    valid_idx = torch.where(finite)[0].detach().cpu().numpy().tolist()
                    valid_idx = sorted(valid_idx, key=lambda j: float(scores[r, j].item()), reverse=True)
                    picked.extend(cand[r, valid_idx[:valid_count]].tolist())

                needed = n_hard - len(picked)
                if needed > 0:
                    fallback = self._sample_random_negatives(
                        users_np=np.asarray([users_np[r]], dtype=np.int64),
                        n_neg=needed,
                    )[0].tolist()
                    picked.extend(fallback)

                hard[r] = np.asarray(picked[:n_hard], dtype=np.int64)

        return hard

    def _materialize_embeddings(self):
        with torch.no_grad():
            self._refresh_adjacency(edge_keep_prob=1.0)
            user_all, item_all = self._propagate()

            if self.normalize_embeddings:
                user_all = F.normalize(user_all, dim=1)
                item_all = F.normalize(item_all, dim=1)

        self.user_factors = user_all.detach().float().cpu().numpy()
        self.item_factors = item_all.detach().float().cpu().numpy()

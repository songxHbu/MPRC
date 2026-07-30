# coding: utf-8


import argparse
import math
import os
import random
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ----------------------------- Utilities -----------------------------

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ----------------------------- Data -----------------------------

class ClothingInterData:
    """Data loader for the uploaded Clothing .inter dataset."""

    def __init__(
        self,
        data_path: str,
        inter_file: str,
        text_feat_file: str,
        image_feat_file: Optional[str],
        pop_seq_len: int,
        device: torch.device,
    ):
        self.data_path = data_path
        self.inter_path = os.path.join(data_path, inter_file)
        self.text_path = os.path.join(data_path, text_feat_file)
        self.image_path = os.path.join(data_path, image_feat_file) if image_feat_file else None
        self.pop_seq_len = pop_seq_len
        self.device = device

        self.n_users = 0
        self.n_items = 0
        self.raw_user2idx: Dict[int, int] = {}
        self.raw_item2idx: Dict[int, int] = {}
        self.idx_item2raw: List[int] = []

        self.train_pairs: np.ndarray = np.empty((0, 2), dtype=np.int64)
        self.train_dict: Dict[int, List[int]] = defaultdict(list)
        self.valid_dict: Dict[int, List[int]] = defaultdict(list)
        self.test_dict: Dict[int, List[int]] = defaultdict(list)
        self.valid_mask_dict: Dict[int, set] = defaultdict(set)
        self.test_mask_dict: Dict[int, set] = defaultdict(set)

        self.pop_feat: torch.Tensor
        self.user_pop_seq: torch.Tensor
        self.text_feat: torch.Tensor
        self.visual_feat: Optional[torch.Tensor]
        self.mm_raw_feat: torch.Tensor
        self.norm_adj: sp.coo_matrix

        self._load_interactions()
        self._load_features()
        self._build_popularity()
        self._build_user_pop_seq()
        self._build_ui_graph()

    def _load_interactions(self) -> None:
        if not os.path.exists(self.inter_path):
            raise FileNotFoundError(f"Cannot find interaction file: {self.inter_path}")

        df = pd.read_csv(self.inter_path, sep="\t")
        required = {"userID", "itemID", "timestamp", "x_label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns {missing}. Existing columns: {list(df.columns)}")

        df = df.sort_values(["userID", "timestamp"]).reset_index(drop=True)
        raw_users = sorted(df["userID"].unique().tolist())
        raw_items = sorted(df["itemID"].unique().tolist())

        self.raw_user2idx = {u: i for i, u in enumerate(raw_users)}
        self.raw_item2idx = {it: i for i, it in enumerate(raw_items)}
        self.idx_item2raw = raw_items
        self.n_users = len(raw_users)
        self.n_items = len(raw_items)

        df["uid"] = df["userID"].map(self.raw_user2idx).astype(np.int64)
        df["iid"] = df["itemID"].map(self.raw_item2idx).astype(np.int64)

        train_df = df[df["x_label"] == 0].copy()
        valid_df = df[df["x_label"] == 1].copy()
        test_df = df[df["x_label"] == 2].copy()

        if len(train_df) == 0:
            raise ValueError("No training rows found: expected x_label=0.")
        if len(test_df) == 0:
            raise ValueError("No testing rows found: expected x_label=2.")

        self.train_pairs = train_df[["uid", "iid"]].drop_duplicates().values.astype(np.int64)

        for uid, g in train_df.groupby("uid", sort=False):
            self.train_dict[int(uid)] = g.sort_values("timestamp")["iid"].astype(int).tolist()
        for uid, g in valid_df.groupby("uid", sort=False):
            self.valid_dict[int(uid)] = g.sort_values("timestamp")["iid"].astype(int).tolist()
        for uid, g in test_df.groupby("uid", sort=False):
            self.test_dict[int(uid)] = g.sort_values("timestamp")["iid"].astype(int).tolist()

        # Validation masks training items only. Test masks training + validation items.
        for u in range(self.n_users):
            self.valid_mask_dict[u].update(self.train_dict.get(u, []))
            self.test_mask_dict[u].update(self.train_dict.get(u, []))
            self.test_mask_dict[u].update(self.valid_dict.get(u, []))

        print(f"[{now()}] users={self.n_users}, items={self.n_items}")
        print(f"[{now()}] train_pairs={len(self.train_pairs)}, valid_users={len(self.valid_dict)}, test_users={len(self.test_dict)}")
        print(f"[{now()}] x_label counts: {df['x_label'].value_counts().sort_index().to_dict()}")

    @staticmethod
    def _load_npy(path: str):
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.ndim == 0:
            return arr.item()
        return arr

    @staticmethod
    def _try_keys(raw_item) -> List:
        keys = [raw_item, str(raw_item)]
        try:
            keys.append(int(raw_item))
        except Exception:
            pass
        # Preserve order and uniqueness.
        out = []
        for k in keys:
            if k not in out:
                out.append(k)
        return out

    def _fetch_feature(self, feat_obj, raw_item):
        if isinstance(feat_obj, dict):
            for k in self._try_keys(raw_item):
                if k in feat_obj:
                    return np.asarray(feat_obj[k], dtype=np.float32)
            return None
        raw_int = int(raw_item)
        if 0 <= raw_int < len(feat_obj):
            return np.asarray(feat_obj[raw_int], dtype=np.float32)
        return None

    def _matrix_from_feature_obj(self, feat_obj, name: str) -> np.ndarray:
        if isinstance(feat_obj, dict):
            first = next(iter(feat_obj.values()))
            dim = len(first)
        else:
            if len(feat_obj.shape) != 2:
                raise ValueError(f"{name} must be 2-D array or dict. Got shape={getattr(feat_obj, 'shape', None)}")
            dim = int(feat_obj.shape[1])

        mat = np.zeros((self.n_items, dim), dtype=np.float32)
        missing = 0
        for idx, raw_item in enumerate(self.idx_item2raw):
            feat = self._fetch_feature(feat_obj, raw_item)
            if feat is None:
                missing += 1
                feat = np.zeros(dim, dtype=np.float32)
            mat[idx] = feat[:dim]
        print(f"[{now()}] {name}: dim={dim}, missing={missing}/{self.n_items} ({missing / self.n_items:.4%})")
        return mat

    @staticmethod
    def _standardize_per_dim(x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True) + 1e-8
        return ((x - mean) / std).astype(np.float32)

    def _load_features(self) -> None:
        if not os.path.exists(self.text_path):
            raise FileNotFoundError(f"Cannot find text feature file: {self.text_path}")

        text_obj = self._load_npy(self.text_path)
        text = self._matrix_from_feature_obj(text_obj, "text_feat")
        text = self._standardize_per_dim(text)
        self.text_feat = torch.from_numpy(text).float()

        visual = None
        if self.image_path and os.path.exists(self.image_path):
            image_obj = self._load_npy(self.image_path)
            visual = self._matrix_from_feature_obj(image_obj, "image_feat")
            visual = self._standardize_per_dim(visual)
            self.visual_feat = torch.from_numpy(visual).float()
        else:
            self.visual_feat = None
            print(f"[{now()}] image feature file not found. Using text-only semantic branch.")

        # Raw feature space for KNN item graph. Do not use randomly initialized projections.
        text_norm = text / (np.linalg.norm(text, axis=1, keepdims=True) + 1e-8)
        if visual is None:
            raw = text_norm.astype(np.float32)
        else:
            vis_norm = visual / (np.linalg.norm(visual, axis=1, keepdims=True) + 1e-8)
            raw = np.concatenate([0.5 * text_norm, 0.5 * vis_norm], axis=1).astype(np.float32)
        self.mm_raw_feat = torch.from_numpy(raw).float()

    def _build_popularity(self) -> None:
        counts = np.zeros(self.n_items, dtype=np.float32)
        for _, iid in self.train_pairs:
            counts[int(iid)] += 1.0
        pop = np.log1p(counts)
        if pop.max() > pop.min():
            pop = (pop - pop.min()) / (pop.max() - pop.min())
        self.pop_feat = torch.from_numpy(pop.astype(np.float32)).float()
        self.raw_pop_counts = counts
        print(f"[{now()}] train popularity: mean={pop.mean():.4f}, max={pop.max():.4f}, nonzero={(counts > 0).mean():.4%}")

    def _build_user_pop_seq(self) -> None:
        out = torch.zeros((self.n_users, self.pop_seq_len), dtype=torch.float32)
        for u in range(self.n_users):
            items = self.train_dict.get(u, [])
            if not items:
                continue
            # Use most recent interactions.
            items = items[-self.pop_seq_len:]
            vals = self.pop_feat[torch.LongTensor(items)]
            if len(vals) < self.pop_seq_len:
                pad = torch.zeros(self.pop_seq_len - len(vals), dtype=torch.float32)
                vals = torch.cat([pad, vals], dim=0)
            out[u] = vals[-self.pop_seq_len:]
        self.user_pop_seq = out

    def _build_ui_graph(self) -> None:
        n_nodes = self.n_users + self.n_items
        rows, cols = [], []
        for u, i in self.train_pairs:
            u = int(u)
            i = int(i)
            rows.extend([u, self.n_users + i])
            cols.extend([self.n_users + i, u])
        data = np.ones(len(rows), dtype=np.float32)
        adj = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)
        degree = np.asarray(adj.sum(axis=1)).flatten()
        degree[degree == 0] = 1e-8
        d_inv = np.power(degree, -0.5)
        self.norm_adj = sp.diags(d_inv).dot(adj).dot(sp.diags(d_inv)).tocoo()
        print(f"[{now()}] UI graph edges={len(rows)}")


class BPRDataset(Dataset):
    def __init__(self, data: ClothingInterData):
        self.pairs = torch.LongTensor(data.train_pairs)
        self.n_items = data.n_items
        self.train_sets = {u: set(items) for u, items in data.train_dict.items()}

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        u = int(self.pairs[idx, 0])
        pos = int(self.pairs[idx, 1])
        seen = self.train_sets.get(u, set())
        neg = random.randint(0, self.n_items - 1)
        while neg in seen:
            neg = random.randint(0, self.n_items - 1)
        return torch.tensor(u, dtype=torch.long), torch.tensor(pos, dtype=torch.long), torch.tensor(neg, dtype=torch.long)


# ----------------------------- Model -----------------------------

class UserPopEncoder(nn.Module):
    def __init__(self, seq_len: int, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaptivePopDebiasMMGCN(nn.Module):
    def __init__(self, data: ClothingInterData, args: argparse.Namespace, device: torch.device):
        super().__init__()
        self.args = args
        self.device = device
        self.n_users = data.n_users
        self.n_items = data.n_items
        self.embed_dim = args.embed_dim

        self.user_emb = nn.Embedding(self.n_users, args.embed_dim)
        self.item_emb = nn.Embedding(self.n_items, args.embed_dim)

        self.register_buffer("pop_feat", data.pop_feat.to(device))
        self.register_buffer("user_pop_seq", data.user_pop_seq.to(device))
        self.register_buffer("text_feat", data.text_feat.to(device))
        self.has_visual = data.visual_feat is not None
        if self.has_visual:
            self.register_buffer("visual_feat", data.visual_feat.to(device))
        else:
            self.visual_feat = None

        self.text_proj = nn.Sequential(
            nn.Linear(data.text_feat.shape[1], args.embed_dim),
            nn.LayerNorm(args.embed_dim),
            nn.Dropout(args.dropout),
        )
        if self.has_visual:
            self.visual_proj = nn.Sequential(
                nn.Linear(data.visual_feat.shape[1], args.embed_dim),
                nn.LayerNorm(args.embed_dim),
                nn.Dropout(args.dropout),
            )
            self.modal_gate = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        else:
            self.visual_proj = None
            self.modal_gate = None

        self.pop_proj = nn.Sequential(nn.Linear(1, args.embed_dim), nn.LayerNorm(args.embed_dim))
        self.user_pop_encoder = UserPopEncoder(args.pop_seq_len, args.embed_dim, args.dropout)
        self.pop_pred = nn.Linear(args.embed_dim, 1)

        # Adaptive residual gates. Gate factor is user_gate(u) * item_gate(i).
        self.user_conform_gate = nn.Sequential(
            nn.Linear(args.embed_dim, args.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(args.embed_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.item_pop_gate = nn.Sequential(
            nn.Linear(args.embed_dim * 2 + 1, args.embed_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.embed_dim, 1),
            nn.Sigmoid(),
        )

        # Stable global mixture weights.
        self.semantic_scale = nn.Parameter(torch.tensor(float(args.semantic_w), dtype=torch.float32))
        self.train_pop_scale = nn.Parameter(torch.tensor(float(args.train_pop_w), dtype=torch.float32))

        # UI graph.
        adj = data.norm_adj
        idx = torch.LongTensor(np.vstack([adj.row, adj.col]))
        val = torch.FloatTensor(adj.data)
        self.register_buffer("ui_adj_idx", idx.to(device))
        self.register_buffer("ui_adj_val", val.to(device))
        self.ui_adj_shape = adj.shape

        # Item-item graph from original feature space.
        mm_adj = self._build_mm_knn_graph(data.mm_raw_feat, args.mm_knn_k, args.knn_block_size)
        self.register_buffer("mm_adj_idx", mm_adj.indices().to(device))
        self.register_buffer("mm_adj_val", mm_adj.values().to(device))
        self.mm_adj_shape = mm_adj.shape

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _build_mm_knn_graph(raw_feat_cpu: torch.Tensor, k: int, block_size: int) -> torch.Tensor:
        print(f"[{now()}] Building item-item KNN graph from raw feature space, k={k}")
        x = F.normalize(raw_feat_cpu.float().cpu(), dim=1)
        n = x.size(0)
        rows, cols, vals = [], [], []
        for start in tqdm(range(0, n, block_size), desc="KNN blocks"):
            end = min(start + block_size, n)
            sim = x[start:end] @ x.t()
            ar = torch.arange(start, end)
            sim[torch.arange(end - start), ar] = -1e9
            topv, topi = torch.topk(sim, k=min(k, n - 1), dim=1)
            r = torch.arange(start, end).view(-1, 1).expand(-1, topi.shape[1]).reshape(-1)
            rows.append(r.cpu())
            cols.append(topi.reshape(-1).cpu())
            vals.append(topv.reshape(-1).clamp(min=0).cpu() + 1e-6)
        row = torch.cat(rows)
        col = torch.cat(cols)
        val = torch.cat(vals).float()

        # Symmetric weighted adjacency.
        row2 = torch.cat([row, col])
        col2 = torch.cat([col, row])
        val2 = torch.cat([val, val])
        adj = torch.sparse_coo_tensor(torch.stack([row2, col2]), val2, (n, n)).coalesce()
        deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1e-8)
        ri, ci = adj.indices()
        norm_val = adj.values() * deg[ri].pow(-0.5) * deg[ci].pow(-0.5)
        return torch.sparse_coo_tensor(adj.indices(), norm_val, adj.shape).coalesce()

    def _ui_adj(self) -> torch.Tensor:
        return torch.sparse_coo_tensor(self.ui_adj_idx, self.ui_adj_val, self.ui_adj_shape, device=self.device).coalesce()

    def _mm_adj(self) -> torch.Tensor:
        return torch.sparse_coo_tensor(self.mm_adj_idx, self.mm_adj_val, self.mm_adj_shape, device=self.device).coalesce()

    def _lightgcn(self, ego: torch.Tensor, layers: int) -> torch.Tensor:
        adj = self._ui_adj()
        outs = [ego]
        h = ego
        for _ in range(layers):
            h = torch.sparse.mm(adj, h)
            outs.append(h)
        return torch.stack(outs, dim=1).mean(dim=1)

    def _mm_graph_prop(self, item_x: torch.Tensor, layers: int) -> torch.Tensor:
        adj = self._mm_adj()
        outs = [item_x]
        h = item_x
        for _ in range(layers):
            h = torch.sparse.mm(adj, h)
            outs.append(h)
        return torch.stack(outs, dim=1).mean(dim=1)

    def _semantic_item_init(self) -> torch.Tensor:
        t = self.text_proj(self.text_feat)
        if self.has_visual:
            v = self.visual_proj(self.visual_feat)
            w = F.softmax(self.modal_gate, dim=0)
            return w[0] * t + w[1] * v
        return t

    def compute_all(self) -> Tuple[torch.Tensor, ...]:
        # Collaborative ID branch.
        id_ego = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        id_all = self._lightgcn(id_ego, self.args.lgcn_layers)
        u_id = id_all[:self.n_users]
        i_id = id_all[self.n_users:]

        # Multimodal semantic branch.
        sem_item0 = self._semantic_item_init()
        sem_item = self._mm_graph_prop(sem_item0, self.args.mm_adj_layers)
        sem_ego = torch.cat([self.user_emb.weight, sem_item], dim=0)
        sem_all = self._lightgcn(sem_ego, self.args.sem_ui_layers)
        u_sem = sem_all[:self.n_users]
        i_sem = sem_all[self.n_users:]

        # Popularity / conformity branch.
        pop_item0 = self.pop_proj(self.pop_feat.view(-1, 1))
        pop_ego = torch.cat([self.user_emb.weight, pop_item0], dim=0)
        pop_all = self._lightgcn(pop_ego, self.args.pop_ui_layers)
        i_pop = pop_all[self.n_users:]
        u_pop = self.user_pop_encoder(self.user_pop_seq)

        # Gates.
        u_gate = self.user_conform_gate(u_pop).squeeze(-1).clamp(0.0, 1.0)
        item_gate_input = torch.cat([i_sem, i_pop, self.pop_feat.view(-1, 1)], dim=-1)
        i_gate = self.item_pop_gate(item_gate_input).squeeze(-1).clamp(0.0, 1.0)

        return u_id, i_id, u_sem, i_sem, u_pop, i_pop, u_gate, i_gate

    def score_pairs(self, users: torch.Tensor, items: torch.Tensor, reps: Tuple[torch.Tensor, ...], beta: float = 0.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        u_id, i_id, u_sem, i_sem, u_pop, i_pop, u_gate, i_gate = reps
        base = (u_id[users] * i_id[items]).sum(dim=-1)
        sem = (u_sem[users] * i_sem[items]).sum(dim=-1)
        conform = (u_pop[users] * i_pop[items]).sum(dim=-1)
        gate = u_gate[users] * i_gate[items]
        residual = gate * conform
        sem_w = self.semantic_scale.clamp(0.0, 2.0)
        train_pop_w = self.train_pop_scale.clamp(0.0, 1.0)
        score = base + sem_w * sem + train_pop_w * residual - float(beta) * residual
        parts = {"base": base, "semantic": sem, "conform": conform, "gate": gate, "residual": residual}
        return score, parts

    @torch.no_grad()
    def full_sort_scores(self, users: torch.Tensor, reps: Tuple[torch.Tensor, ...], beta: float = 0.0) -> torch.Tensor:
        u_id, i_id, u_sem, i_sem, u_pop, i_pop, u_gate, i_gate = reps
        base = u_id[users] @ i_id.t()
        sem = u_sem[users] @ i_sem.t()
        conform = u_pop[users] @ i_pop.t()
        gate = u_gate[users].view(-1, 1) * i_gate.view(1, -1)
        residual = gate * conform
        sem_w = self.semantic_scale.clamp(0.0, 2.0)
        train_pop_w = self.train_pop_scale.clamp(0.0, 1.0)
        return base + sem_w * sem + train_pop_w * residual - float(beta) * residual

    def auxiliary_losses(self, reps: Tuple[torch.Tensor, ...]) -> Dict[str, torch.Tensor]:
        _, _, _, i_sem, _, i_pop, u_gate, i_gate = reps
        pred_pop = torch.sigmoid(self.pop_pred(i_pop)).squeeze(-1)
        pop_loss = F.mse_loss(pred_pop, self.pop_feat)
        orth_loss = torch.abs(F.cosine_similarity(F.normalize(i_sem, dim=-1), F.normalize(i_pop, dim=-1), dim=-1)).mean()
        # Gate sparsity prevents always-removing all popularity signal.
        gate_sparse = 0.5 * (u_gate.mean() + i_gate.mean())
        return {"pop": pop_loss, "orth": orth_loss, "gate_sparse": gate_sparse}


# ----------------------------- Metrics -----------------------------

def bpr_loss(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(pos - neg).mean()


def _rank_metrics_from_topk(topk_items: np.ndarray, positives: set, top_ks: Sequence[int]) -> Tuple[Dict[int, float], Dict[int, float]]:
    hr = {k: 0.0 for k in top_ks}
    ndcg = {k: 0.0 for k in top_ks}
    for k in top_ks:
        hit_items = [it for it in topk_items[:k] if int(it) in positives]
        if hit_items:
            hr[k] = 1.0
        dcg = 0.0
        for rank, it in enumerate(topk_items[:k]):
            if int(it) in positives:
                dcg += 1.0 / math.log2(rank + 2)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(positives))))
        ndcg[k] = dcg / idcg if idcg > 0 else 0.0
    return hr, ndcg


@torch.no_grad()
def evaluate_full_sort(
    model: AdaptivePopDebiasMMGCN,
    data: ClothingInterData,
    split: str,
    beta: float,
    top_ks: Sequence[int],
    user_batch_size: int,
    device: torch.device,
    show_progress: bool = False,
) -> Dict[str, float]:
    model.eval()
    reps = model.compute_all()
    if split == "valid":
        eval_dict = data.valid_dict
        mask_dict = data.valid_mask_dict
    elif split == "test":
        eval_dict = data.test_dict
        mask_dict = data.test_mask_dict
    else:
        raise ValueError("split must be valid or test")

    users = [u for u in eval_dict.keys() if len(eval_dict[u]) > 0]
    max_k = max(top_ks)
    hr_sum = {k: 0.0 for k in top_ks}
    ndcg_sum = {k: 0.0 for k in top_ks}
    arp_sum = {k: 0.0 for k in top_ks}       # average recommendation popularity
    tail_sum = {k: 0.0 for k in top_ks}      # fraction of long-tail items in top-k
    coverage_sets = {k: set() for k in top_ks}

    pop_np = data.pop_feat.cpu().numpy()
    # Long-tail threshold: lower 80% popularity by normalized train popularity.
    tail_thr = float(np.quantile(pop_np, 0.80))

    iterator = range(0, len(users), user_batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Eval-{split}-beta={beta:.3f}")

    valid_user_count = 0
    for start in iterator:
        batch_users = users[start:start + user_batch_size]
        u_tensor = torch.LongTensor(batch_users).to(device)
        scores = model.full_sort_scores(u_tensor, reps, beta=beta)
        scores = scores.detach().cpu().numpy()

        for row, u in enumerate(batch_users):
            positives = set(eval_dict.get(u, []))
            if not positives:
                continue
            # Mask seen items but keep current split positives.
            for it in mask_dict.get(u, set()):
                scores[row, int(it)] = -np.inf
            top_items = np.argpartition(-scores[row], kth=max_k - 1)[:max_k]
            top_items = top_items[np.argsort(-scores[row][top_items])]

            hr, ndcg = _rank_metrics_from_topk(top_items, positives, top_ks)
            valid_user_count += 1
            for k in top_ks:
                hr_sum[k] += hr[k]
                ndcg_sum[k] += ndcg[k]
                recs = top_items[:k]
                arp_sum[k] += float(np.mean(pop_np[recs]))
                tail_sum[k] += float(np.mean(pop_np[recs] <= tail_thr))
                coverage_sets[k].update(map(int, recs))

    denom = max(valid_user_count, 1)
    out: Dict[str, float] = {"users": float(valid_user_count), "beta": float(beta)}
    for k in top_ks:
        out[f"HR@{k}"] = hr_sum[k] / denom
        out[f"NDCG@{k}"] = ndcg_sum[k] / denom
        out[f"ARP@{k}"] = arp_sum[k] / denom
        out[f"TAIL@{k}"] = tail_sum[k] / denom
        out[f"COV@{k}"] = len(coverage_sets[k]) / data.n_items
    return out


def format_metrics(m: Dict[str, float], top_ks: Sequence[int]) -> str:
    parts = []
    for k in top_ks:
        parts.append(f"HR@{k}={m[f'HR@{k}']:.4f}, NDCG@{k}={m[f'NDCG@{k}']:.4f}")
    parts.append(f"ARP@{max(top_ks)}={m[f'ARP@{max(top_ks)}']:.4f}")
    parts.append(f"TAIL@{max(top_ks)}={m[f'TAIL@{max(top_ks)}']:.4f}")
    return " | ".join(parts)


def choose_beta(
    model: AdaptivePopDebiasMMGCN,
    data: ClothingInterData,
    betas: Sequence[float],
    top_ks: Sequence[int],
    user_batch_size: int,
    device: torch.device,
    mode: str,
    pop_penalty: float,
) -> Tuple[float, Dict[float, Dict[str, float]]]:
    results = {}
    key_k = max(top_ks)
    best_beta = 0.0
    best_score = -1e18
    raw_arp = None
    for beta in betas:
        m = evaluate_full_sort(model, data, split="valid", beta=beta, top_ks=top_ks, user_batch_size=user_batch_size, device=device)
        results[float(beta)] = m
        if raw_arp is None and abs(beta) < 1e-12:
            raw_arp = m[f"ARP@{key_k}"]
        if mode == "accuracy":
            select_score = m[f"NDCG@{key_k}"]
        elif mode == "tradeoff":
            # Reward accuracy, and mildly reward lower recommendation popularity than raw.
            pop_reduction = 0.0 if raw_arp is None else max(0.0, raw_arp - m[f"ARP@{key_k}"])
            select_score = m[f"NDCG@{key_k}"] + pop_penalty * pop_reduction
        else:
            raise ValueError("beta_selection must be accuracy or tradeoff")
        if select_score > best_score:
            best_score = select_score
            best_beta = float(beta)
    return best_beta, results


# ----------------------------- Training -----------------------------

def train_one_epoch(
    model: AdaptivePopDebiasMMGCN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total = defaultdict(float)
    n_batches = 0
    for users, pos, neg in tqdm(loader, desc="Train", leave=False):
        users = users.to(device)
        pos = pos.to(device)
        neg = neg.to(device)

        reps = model.compute_all()
        pos_score, _ = model.score_pairs(users, pos, reps, beta=0.0)
        neg_score, _ = model.score_pairs(users, neg, reps, beta=0.0)
        loss_bpr = bpr_loss(pos_score, neg_score)
        aux = model.auxiliary_losses(reps)
        reg = model.user_emb(users).pow(2).mean() + model.item_emb(pos).pow(2).mean() + model.item_emb(neg).pow(2).mean()
        loss = (
            args.lambda_bpr * loss_bpr
            + args.lambda_pop * aux["pop"]
            + args.lambda_orth * aux["orth"]
            + args.lambda_gate_sparse * aux["gate_sparse"]
            + args.lambda_reg * reg
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        total["loss"] += float(loss.item())
        total["bpr"] += float(loss_bpr.item())
        total["pop"] += float(aux["pop"].item())
        total["orth"] += float(aux["orth"].item())
        total["gate"] += float(aux["gate_sparse"].item())
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total.items()}


def save_checkpoint(path: str, model: nn.Module, args: argparse.Namespace, epoch: int, best_beta: float, best_valid: Dict[str, float]) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save({
        "model_state": model.state_dict(),
        "args": vars(args),
        "epoch": epoch,
        "best_beta": best_beta,
        "best_valid": best_valid,
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/Sports", help="Dataset directory.")
    parser.add_argument("--inter_file", type=str, default="sports.inter")
    parser.add_argument("--text_feat_file", type=str, default="text_feat.npy")
    parser.add_argument("--image_feat_file", type=str, default="image_feat.npy",
                        help="Optional. Set empty string to disable image features.")

    # parser.add_argument("--inter_file", type=str, default="baby.inter")
    # parser.add_argument("--text_feat_file", type=str, default="text_feat.npy")
    # parser.add_argument("--image_feat_file", type=str, default="image_feat.npy",
    #                     help="Optional. Set empty string to disable image features.")

    # parser.add_argument("--inter_file", type=str, default="Copy of clothing.inter")
    # parser.add_argument("--text_feat_file", type=str, default="Copy of text_feat.npy")
    # parser.add_argument("--image_feat_file", type=str, default="Copy of image_feat.npy", help="Optional. Set empty string to disable image features.")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Model.
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--pop_seq_len", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lgcn_layers", type=int, default=2)
    parser.add_argument("--sem_ui_layers", type=int, default=1)
    parser.add_argument("--pop_ui_layers", type=int, default=1)
    parser.add_argument("--mm_knn_k", type=int, default=10)
    parser.add_argument("--mm_adj_layers", type=int, default=1)
    parser.add_argument("--knn_block_size", type=int, default=1024)
    parser.add_argument("--semantic_w", type=float, default=0.20)
    parser.add_argument("--train_pop_w", type=float, default=0.05)

    # Training.
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--patience", type=int, default=30)

    # Loss weights.
    parser.add_argument("--lambda_bpr", type=float, default=1.0)
    parser.add_argument("--lambda_pop", type=float, default=0.02)
    parser.add_argument("--lambda_orth", type=float, default=0.01)
    parser.add_argument("--lambda_gate_sparse", type=float, default=0.001)
    parser.add_argument("--lambda_reg", type=float, default=1e-4)

    # Evaluation.
    parser.add_argument("--topks", type=str, default="5,10")
    parser.add_argument("--eval_user_batch", type=int, default=512)
    parser.add_argument("--beta_candidates", type=str, default="0,0.01,0.02,0.05,0.1,0.2")
    parser.add_argument("--beta_selection", type=str, default="accuracy", choices=["accuracy", "tradeoff"])
    parser.add_argument("--pop_penalty", type=float, default=0.20)
    parser.add_argument("--save_path", type=str, default="./checkpoints/adaptive_pop_debias_best.pt")

    args = parser.parse_args()
    if args.image_feat_file.strip() == "":
        args.image_feat_file = None
    top_ks = [int(x) for x in args.topks.split(",") if x.strip()]
    betas = [float(x) for x in args.beta_candidates.split(",") if x.strip()]

    device = torch.device(args.device)
    set_seed(args.seed)
    print(f"[{now()}] device={device}")
    print(f"[{now()}] args={args}")

    data = ClothingInterData(
        data_path=args.data_path,
        inter_file=args.inter_file,
        text_feat_file=args.text_feat_file,
        image_feat_file=args.image_feat_file,
        pop_seq_len=args.pop_seq_len,
        device=device,
    )
    dataset = BPRDataset(data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))

    model = AdaptivePopDebiasMMGCN(data, args, device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_key = f"NDCG@{max(top_ks)}"
    best_valid_score = -1.0
    best_epoch = 0
    best_beta = 0.0
    best_valid_metrics: Dict[str, float] = {}
    bad_count = 0

    print(f"[{now()}] Start training")
    for epoch in range(1, args.epochs + 1):
        losses = train_one_epoch(model, loader, optimizer, args, device)
        if epoch % args.eval_interval == 0 or epoch == 1:
            chosen_beta, beta_results = choose_beta(
                model, data, betas, top_ks, args.eval_user_batch, device,
                mode=args.beta_selection, pop_penalty=args.pop_penalty
            )
            valid_raw = beta_results.get(0.0) or evaluate_full_sort(model, data, "valid", 0.0, top_ks, args.eval_user_batch, device)
            valid_deb = beta_results[chosen_beta]
            current_score = valid_deb[best_key]

            print(
                f"[{now()}] Epoch {epoch:03d} "
                f"loss={losses['loss']:.4f} bpr={losses['bpr']:.4f} pop={losses['pop']:.4f} "
                f"orth={losses['orth']:.4f} gate={losses['gate']:.4f}"
            )
            print(f"  Valid Raw(beta=0):        {format_metrics(valid_raw, top_ks)}")
            print(f"  Valid Selected(beta={chosen_beta:.3f}): {format_metrics(valid_deb, top_ks)}")

            if current_score > best_valid_score:
                best_valid_score = current_score
                best_epoch = epoch
                best_beta = chosen_beta
                best_valid_metrics = valid_deb
                bad_count = 0
                save_checkpoint(args.save_path, model, args, epoch, best_beta, best_valid_metrics)
                print(f"  New best checkpoint saved: {args.save_path}")
            else:
                bad_count += args.eval_interval
                if bad_count >= args.patience:
                    print(f"[{now()}] Early stopping at epoch {epoch}. Best epoch={best_epoch}, best_beta={best_beta:.3f}")
                    break

    # Load best checkpoint and evaluate on test.
    if os.path.exists(args.save_path):
        ckpt = torch.load(args.save_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        best_beta = float(ckpt.get("best_beta", best_beta))
        print(f"[{now()}] Loaded best checkpoint from epoch={ckpt.get('epoch')} beta={best_beta:.3f}")

    test_raw = evaluate_full_sort(model, data, "test", 0.0, top_ks, args.eval_user_batch, device, show_progress=True)
    test_deb = evaluate_full_sort(model, data, "test", best_beta, top_ks, args.eval_user_batch, device, show_progress=True)

    print("\n================ Final Results ================")
    print(f"Best Valid(beta={best_beta:.3f}): {format_metrics(best_valid_metrics, top_ks) if best_valid_metrics else 'N/A'}")
    print(f"Final Raw Test(beta=0):        {format_metrics(test_raw, top_ks)}")
    print(f"Final Debiased Test(beta={best_beta:.3f}): {format_metrics(test_deb, top_ks)}")
    print("\nNotes:")
    print("  Raw keeps the learned popularity/conformity residual.")
    print("  Debiased subtracts beta * adaptive_gate * conformity_residual.")
    print("  If beta=0 is selected, validation accuracy indicates that removing popularity hurts NDCG.")
    print("  Use --beta_selection tradeoff to prefer lower recommendation popularity with mild accuracy tradeoff.")


if __name__ == "__main__":
    main()

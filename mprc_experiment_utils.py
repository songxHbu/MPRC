# coding: utf-8
"""Shared utilities for paper-oriented MPRC experiments.

This module intentionally does not modify the original training script. It loads
that script dynamically and subclasses its model only inside the experiment
process.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


DEFAULT_BASE_FILE = str(Path(__file__).with_name("2d932a83-daec-46d4-ac68-b3e11adf9bb4.py"))

VARIANT_DISPLAY = {
    "id_only": "ID-only",
    "wo_semantic": "w/o Semantic Path",
    "wo_popularity": "w/o Popularity Path",
    "wo_item_graph": "w/o Item Graph",
    "wo_orthogonality": "w/o Orthogonality",
    "wo_gate_sparsity": "w/o Gate Sparsity",
    "wo_user_gate": "w/o User Gate",
    "wo_item_gate": "w/o Item Gate",
    "wo_pop_sequence": "w/o Popularity Sequence",
    "fixed_r": "Fixed-R",
    "full": "MPRC",
    "mprc_raw": "MPRC-Raw",
    "mprc_calibrated": "MPRC-Calibrated",
}

PANEL_A_VARIANTS = ["id_only", "wo_semantic", "wo_popularity", "wo_item_graph"]
PANEL_B_VARIANTS = [
    "wo_orthogonality",
    "wo_gate_sparsity",
    "wo_user_gate",
    "wo_item_gate",
    "wo_pop_sequence",
    "fixed_r",
]

_MM_GRAPH_CACHE: Dict[Tuple[int, int, int, int], torch.Tensor] = {}


def load_base_module(base_file: str) -> ModuleType:
    base_file = os.path.abspath(base_file)
    if not os.path.exists(base_file):
        raise FileNotFoundError(f"Base script not found: {base_file}")
    spec = importlib.util.spec_from_file_location("mprc_base_script", base_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import base script: {base_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_common_data_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base_file", type=str, default=DEFAULT_BASE_FILE)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--inter_file", type=str, required=True)
    parser.add_argument("--text_feat_file", type=str, default="text_feat.npy")
    parser.add_argument("--image_feat_file", type=str, default="image_feat.npy")
    parser.add_argument("--dataset_name", type=str, default="dataset")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2024)

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
    parser.add_argument("--fixed_r", type=float, default=0.5,
                        help="Constant gate product used by the Fixed-R ablation.")


def add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--patience", type=int, default=30)

    parser.add_argument("--lambda_bpr", type=float, default=1.0)
    parser.add_argument("--lambda_pop", type=float, default=0.02)
    parser.add_argument("--lambda_orth", type=float, default=0.01)
    parser.add_argument("--lambda_gate_sparse", type=float, default=0.001)
    parser.add_argument("--lambda_reg", type=float, default=1e-4)

    parser.add_argument("--topks", type=str, default="5,10")
    parser.add_argument("--eval_user_batch", type=int, default=512)
    parser.add_argument("--beta_candidates", type=str, default="0,0.01,0.02,0.05,0.1,0.2")
    parser.add_argument("--beta_selection", type=str, default="tradeoff", choices=["accuracy", "tradeoff"])
    parser.add_argument("--pop_penalty", type=float, default=0.20)


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if isinstance(args.image_feat_file, str) and args.image_feat_file.strip() == "":
        args.image_feat_file = None
    return args


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def copy_namespace(args: argparse.Namespace, **updates) -> argparse.Namespace:
    out = copy.deepcopy(args)
    for key, value in updates.items():
        setattr(out, key, value)
    return out


def configure_variant(args: argparse.Namespace, variant: str) -> argparse.Namespace:
    if variant not in VARIANT_DISPLAY:
        raise ValueError(f"Unknown variant: {variant}")
    out = copy_namespace(args, experiment_variant=variant)

    if variant == "wo_item_graph":
        out.mm_adj_layers = 0
    elif variant == "wo_orthogonality":
        out.lambda_orth = 0.0
    elif variant == "wo_gate_sparsity":
        out.lambda_gate_sparse = 0.0
    elif variant == "id_only":
        out.semantic_w = 0.0
        out.train_pop_w = 0.0
        out.lambda_pop = 0.0
        out.lambda_orth = 0.0
        out.lambda_gate_sparse = 0.0
    elif variant == "wo_semantic":
        out.semantic_w = 0.0
        out.lambda_orth = 0.0
    elif variant == "wo_popularity":
        out.train_pop_w = 0.0
        out.lambda_pop = 0.0
        out.lambda_orth = 0.0
        out.lambda_gate_sparse = 0.0

    return out


def make_experiment_model_class(base: ModuleType):
    class ExperimentModel(base.AdaptivePopCalibratMMGCN):
        """A runtime-only subclass implementing paper ablations.

        State-dict keys stay compatible with the original model because no new
        trainable modules are introduced.
        """

        def _build_mm_knn_graph(self, raw_feat_cpu: torch.Tensor, k: int, block_size: int) -> torch.Tensor:
            key = (int(raw_feat_cpu.data_ptr()), int(raw_feat_cpu.shape[0]), int(k), int(block_size))
            if key not in _MM_GRAPH_CACHE:
                _MM_GRAPH_CACHE[key] = super()._build_mm_knn_graph(raw_feat_cpu, k, block_size).cpu().coalesce()
            else:
                print(f"[{base.now()}] Reusing cached item-item KNN graph.")
            return _MM_GRAPH_CACHE[key]

        @property
        def variant(self) -> str:
            return getattr(self.args, "experiment_variant", "full")

        def compute_all(self) -> Tuple[torch.Tensor, ...]:
            reps = list(super().compute_all())
            u_id, i_id, u_sem, i_sem, u_pop, i_pop, u_gate, i_gate = reps

            if self.variant == "wo_pop_sequence":
                # Replace sequence-derived conformity with a static user-ID
                # representation, while retaining identical dimensionality.
                u_pop = F.layer_norm(self.user_emb.weight, (self.embed_dim,))
                u_gate = self.user_conform_gate(u_pop).squeeze(-1).clamp(0.0, 1.0)

            if self.variant == "wo_user_gate":
                u_gate = torch.ones_like(u_gate)
            if self.variant == "wo_item_gate":
                i_gate = torch.ones_like(i_gate)
            if self.variant == "fixed_r":
                fixed_r = float(np.clip(getattr(self.args, "fixed_r", 0.5), 0.0, 1.0))
                root = math.sqrt(fixed_r)
                u_gate = torch.full_like(u_gate, root)
                i_gate = torch.full_like(i_gate, root)

            return u_id, i_id, u_sem, i_sem, u_pop, i_pop, u_gate, i_gate

        def _use_semantic(self) -> bool:
            return self.variant not in {"id_only", "wo_semantic"}

        def _use_popularity(self) -> bool:
            return self.variant not in {"id_only", "wo_popularity"}

        def score_pairs(
            self,
            users: torch.Tensor,
            items: torch.Tensor,
            reps: Tuple[torch.Tensor, ...],
            beta: float = 0.0,
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            u_id, i_id, u_sem, i_sem, u_pop, i_pop, u_gate, i_gate = reps
            base_score = (u_id[users] * i_id[items]).sum(dim=-1)
            sem_score = (u_sem[users] * i_sem[items]).sum(dim=-1)
            conform = (u_pop[users] * i_pop[items]).sum(dim=-1)
            gate = u_gate[users] * i_gate[items]
            residual = gate * conform

            score = base_score
            if self._use_semantic():
                score = score + self.semantic_scale.clamp(0.0, 2.0) * sem_score
            if self._use_popularity():
                score = score + self.train_pop_scale.clamp(0.0, 1.0) * residual
                score = score - float(beta) * residual

            parts = {
                "base": base_score,
                "semantic": sem_score,
                "conform": conform,
                "gate": gate,
                "residual": residual,
            }
            return score, parts

        @torch.no_grad()
        def full_sort_scores(
            self,
            users: torch.Tensor,
            reps: Tuple[torch.Tensor, ...],
            beta: float = 0.0,
        ) -> torch.Tensor:
            u_id, i_id, u_sem, i_sem, u_pop, i_pop, u_gate, i_gate = reps
            score = u_id[users] @ i_id.t()
            if self._use_semantic():
                score = score + self.semantic_scale.clamp(0.0, 2.0) * (u_sem[users] @ i_sem.t())
            if self._use_popularity():
                conform = u_pop[users] @ i_pop.t()
                gate = u_gate[users].view(-1, 1) * i_gate.view(1, -1)
                residual = gate * conform
                score = score + self.train_pop_scale.clamp(0.0, 1.0) * residual
                score = score - float(beta) * residual
            return score

        def auxiliary_losses(self, reps: Tuple[torch.Tensor, ...]) -> Dict[str, torch.Tensor]:
            losses = super().auxiliary_losses(reps)
            zero = self.user_emb.weight.sum() * 0.0
            if not self._use_popularity():
                losses["pop"] = zero
                losses["gate_sparse"] = zero
            if not self._use_semantic() or not self._use_popularity() or self.variant == "wo_orthogonality":
                losses["orth"] = zero
            if self.variant == "wo_gate_sparsity":
                losses["gate_sparse"] = zero
            return losses

    return ExperimentModel


def build_data(base: ModuleType, args: argparse.Namespace, device: torch.device):
    return base.ClothingInterData(
        data_path=args.data_path,
        inter_file=args.inter_file,
        text_feat_file=args.text_feat_file,
        image_feat_file=args.image_feat_file,
        pop_seq_len=args.pop_seq_len,
        device=device,
    )


def build_model(base: ModuleType, data, args: argparse.Namespace, device: torch.device):
    cls = make_experiment_model_class(base)
    return cls(data, args, device).to(device)


def metric_dict_to_row(
    dataset_name: str,
    variant: str,
    split: str,
    beta: float,
    metrics: Mapping[str, float],
    seed: int,
    panel: str = "",
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "dataset": dataset_name,
        "panel": panel,
        "variant_key": variant,
        "variant": VARIANT_DISPLAY.get(variant, variant),
        "split": split,
        "seed": seed,
        "beta": beta,
    }
    for key, value in metrics.items():
        if key not in {"users", "beta"}:
            row[key] = float(value)
    row["users"] = int(metrics.get("users", 0))
    return row


def train_experiment(
    base: ModuleType,
    data,
    args: argparse.Namespace,
    variant: str,
    checkpoint_path: str,
    selection_betas: Optional[Sequence[float]] = None,
    fixed_eval_beta: Optional[float] = None,
    save_extra: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Train one variant and return raw/calibrated validation and test metrics."""
    args = configure_variant(args, variant)
    device = torch.device(args.device)
    base.set_seed(args.seed)

    top_ks = parse_int_list(args.topks)
    candidate_betas = list(selection_betas) if selection_betas is not None else parse_float_list(args.beta_candidates)
    if fixed_eval_beta is not None:
        candidate_betas = [float(fixed_eval_beta)]
    if variant in {"id_only", "wo_semantic", "wo_popularity", "wo_item_graph", "mprc_raw"}:
        candidate_betas = [0.0]

    dataset = base.BPRDataset(data)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(base, data, args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_key = f"NDCG@{max(top_ks)}"
    best_valid_score = -float("inf")
    best_epoch = 0
    best_beta = float(candidate_betas[0])
    best_valid_metrics: Dict[str, float] = {}
    bad_count = 0

    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        losses = base.train_one_epoch(model, loader, optimizer, args, device)
        if epoch % args.eval_interval != 0 and epoch != 1:
            continue

        if len(candidate_betas) == 1:
            chosen_beta = float(candidate_betas[0])
            valid_selected = base.evaluate_full_sort(
                model, data, "valid", chosen_beta, top_ks, args.eval_user_batch, device
            )
            valid_raw = valid_selected if abs(chosen_beta) < 1e-12 else base.evaluate_full_sort(
                model, data, "valid", 0.0, top_ks, args.eval_user_batch, device
            )
        else:
            chosen_beta, beta_results = base.choose_beta(
                model,
                data,
                candidate_betas,
                top_ks,
                args.eval_user_batch,
                device,
                mode=args.beta_selection,
                pop_penalty=args.pop_penalty,
            )
            valid_selected = beta_results[chosen_beta]
            valid_raw = beta_results.get(0.0) or base.evaluate_full_sort(
                model, data, "valid", 0.0, top_ks, args.eval_user_batch, device
            )

        current_score = float(valid_selected[best_key])
        print(
            f"[{base.now()}] variant={VARIANT_DISPLAY.get(variant, variant)} epoch={epoch:03d} "
            f"loss={losses['loss']:.4f} selected_beta={chosen_beta:.4f} "
            f"valid_{best_key}={current_score:.4f}"
        )

        if current_score > best_valid_score:
            best_valid_score = current_score
            best_epoch = epoch
            best_beta = float(chosen_beta)
            best_valid_metrics = dict(valid_selected)
            bad_count = 0
            payload = {
                "model_state": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "best_beta": best_beta,
                "best_valid": best_valid_metrics,
                "variant": variant,
            }
            if save_extra:
                payload.update(dict(save_extra))
            torch.save(payload, checkpoint_path)
        else:
            bad_count += args.eval_interval
            if bad_count >= args.patience:
                break

    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"No checkpoint was produced for variant {variant}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    best_beta = float(ckpt.get("best_beta", best_beta))
    best_epoch = int(ckpt.get("epoch", best_epoch))

    valid_raw = base.evaluate_full_sort(model, data, "valid", 0.0, top_ks, args.eval_user_batch, device)
    valid_selected = base.evaluate_full_sort(model, data, "valid", best_beta, top_ks, args.eval_user_batch, device)
    test_raw = base.evaluate_full_sort(model, data, "test", 0.0, top_ks, args.eval_user_batch, device, show_progress=True)
    test_selected = base.evaluate_full_sort(
        model, data, "test", best_beta, top_ks, args.eval_user_batch, device, show_progress=True
    )

    return {
        "variant": variant,
        "display_name": VARIANT_DISPLAY.get(variant, variant),
        "best_epoch": best_epoch,
        "best_beta": best_beta,
        "checkpoint": checkpoint_path,
        "args": vars(args),
        "valid_raw": valid_raw,
        "valid_selected": valid_selected,
        "test_raw": test_raw,
        "test_selected": test_selected,
    }


def load_model_from_checkpoint(
    base: ModuleType,
    checkpoint_path: str,
    cli_args: argparse.Namespace,
):
    device = torch.device(cli_args.device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = dict(ckpt.get("args", {}))

    # Architecture must match the saved model. Runtime dataset paths and device
    # are taken from the current command when supplied.
    merged = dict(ckpt_args)
    runtime_override_keys = {
        "base_file", "data_path", "inter_file", "text_feat_file",
        "image_feat_file", "dataset_name", "device", "eval_user_batch",
        "topks", "beta_candidates", "output_dir", "checkpoint",
        "split", "k", "group_count", "tail_quantile", "head_quantile",
        "save_plots", "seed", "calibrated_beta", "user_indices",
        "num_users", "candidate_users", "splits",
    }
    for key, value in vars(cli_args).items():
        if key in runtime_override_keys and value is not None:
            merged[key] = value
        elif key not in merged:
            merged[key] = value
    merged.setdefault("experiment_variant", ckpt.get("variant", "full"))
    merged.setdefault("fixed_r", 0.5)
    args = normalize_args(argparse.Namespace(**merged))

    data = build_data(base, args, device)
    model = build_model(base, data, args, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return args, data, model, ckpt


def write_csv(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: str, obj: object) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def topk_for_users(
    model,
    data,
    users: Sequence[int],
    split: str,
    beta: float,
    k: int,
    user_batch_size: int,
    device: torch.device,
) -> Dict[int, np.ndarray]:
    model.eval()
    reps = model.compute_all()
    mask_dict = data.valid_mask_dict if split == "valid" else data.test_mask_dict
    out: Dict[int, np.ndarray] = {}
    for start in range(0, len(users), user_batch_size):
        batch_users = list(users[start:start + user_batch_size])
        scores = model.full_sort_scores(torch.LongTensor(batch_users).to(device), reps, beta=beta)
        scores = scores.detach().cpu().numpy()
        for row, u in enumerate(batch_users):
            for item in mask_dict.get(u, set()):
                scores[row, int(item)] = -np.inf
            kk = min(k, data.n_items)
            idx = np.argpartition(-scores[row], kth=kk - 1)[:kk]
            idx = idx[np.argsort(-scores[row][idx])]
            out[int(u)] = idx.astype(np.int64)
    return out


def rank_metrics_single(top_items: Sequence[int], positives: set, k: int) -> Tuple[float, float]:
    top_items = list(top_items[:k])
    hit = 1.0 if any(int(i) in positives for i in top_items) else 0.0
    dcg = 0.0
    for rank, item in enumerate(top_items):
        if int(item) in positives:
            dcg += 1.0 / math.log2(rank + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(positives))))
    return hit, (dcg / idcg if idcg > 0 else 0.0)

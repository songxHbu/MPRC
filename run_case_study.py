# coding: utf-8
"""Generate user-level Raw-vs-Calibrated recommendation case studies."""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from mprc_experiment_utils import (
    add_common_data_model_args,
    add_training_args,
    load_base_module,
    load_model_from_checkpoint,
    normalize_args,
    write_csv,
)


def parse_user_ids(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def masked_topk(scores: np.ndarray, mask: Sequence[int], k: int) -> np.ndarray:
    scores = scores.copy()
    for item in mask:
        scores[int(item)] = -np.inf
    kk = min(k, len(scores))
    idx = np.argpartition(-scores, kth=kk - 1)[:kk]
    return idx[np.argsort(-scores[idx])]


def main() -> None:
    parser = argparse.ArgumentParser(description="MPRC recommendation case study")
    add_common_data_model_args(parser)
    add_training_args(parser)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./paper_results/case_study")
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--user_indices", type=str, default="",
                        help="Internal user indices. Empty means automatic selection.")
    parser.add_argument("--num_users", type=int, default=3)
    parser.add_argument("--candidate_users", type=int, default=500,
                        help="Maximum users scanned for automatic selection.")
    parser.add_argument("--calibrated_beta", type=float, default=None)
    args = normalize_args(parser.parse_args())

    os.makedirs(args.output_dir, exist_ok=True)
    base = load_base_module(args.base_file)
    args, data, model, ckpt = load_model_from_checkpoint(base, args.checkpoint, args)
    device = torch.device(args.device)
    beta = float(args.calibrated_beta if args.calibrated_beta is not None else ckpt.get("best_beta", 0.0))

    eval_dict = data.valid_dict if args.split == "valid" else data.test_dict
    mask_dict = data.valid_mask_dict if args.split == "valid" else data.test_mask_dict
    available_users = sorted(u for u, positives in eval_dict.items() if positives)

    model.eval()
    with torch.no_grad():
        reps = model.compute_all()
        u_id, i_id, u_sem, i_sem, u_pop, i_pop, u_gate, i_gate = reps

    def scores_for_user(u: int, b: float) -> np.ndarray:
        with torch.no_grad():
            s = model.full_sort_scores(torch.tensor([u], device=device), reps, beta=b)[0]
        return s.detach().cpu().numpy()

    if args.user_indices.strip():
        selected = parse_user_ids(args.user_indices)
    else:
        changes: List[Tuple[float, int]] = []
        for u in available_users[:args.candidate_users]:
            raw_top = masked_topk(scores_for_user(u, 0.0), mask_dict.get(u, set()), args.k)
            cal_top = masked_topk(scores_for_user(u, beta), mask_dict.get(u, set()), args.k)
            overlap = len(set(map(int, raw_top)) & set(map(int, cal_top))) / max(args.k, 1)
            changes.append((1.0 - overlap, u))
        changes.sort(reverse=True)
        selected = [u for _, u in changes[:args.num_users]]

    idx_user2raw = {idx: raw for raw, idx in data.raw_user2idx.items()}
    pop_np = data.pop_feat.cpu().numpy()
    tail_thr = float(np.quantile(pop_np, 0.80))
    u_gate_np = u_gate.detach().cpu().numpy()
    i_gate_np = i_gate.detach().cpu().numpy()
    u_pop_cpu = u_pop.detach().cpu()
    i_pop_cpu = i_pop.detach().cpu()

    rows: List[Dict[str, object]] = []
    for u in selected:
        raw_scores = scores_for_user(u, 0.0)
        cal_scores = scores_for_user(u, beta)
        raw_top = masked_topk(raw_scores, mask_dict.get(u, set()), args.k)
        cal_top = masked_topk(cal_scores, mask_dict.get(u, set()), args.k)
        raw_rank = {int(item): rank + 1 for rank, item in enumerate(raw_top)}
        cal_rank = {int(item): rank + 1 for rank, item in enumerate(cal_top)}
        union = sorted(set(raw_rank) | set(cal_rank), key=lambda i: (cal_rank.get(i, 10**9), raw_rank.get(i, 10**9)))

        conform = (u_pop_cpu[u].view(1, -1) * i_pop_cpu[union]).sum(dim=-1).numpy()
        residual = u_gate_np[u] * i_gate_np[union] * conform
        positives = set(map(int, eval_dict.get(u, [])))

        for pos, item in enumerate(union):
            rows.append({
                "dataset": args.dataset_name,
                "split": args.split,
                "user_idx": u,
                "raw_user_id": idx_user2raw.get(u, u),
                "item_idx": item,
                "raw_item_id": data.idx_item2raw[item],
                "raw_rank": raw_rank.get(item, ""),
                "calibrated_rank": cal_rank.get(item, ""),
                "raw_score": float(raw_scores[item]),
                "calibrated_score": float(cal_scores[item]),
                "score_change": float(cal_scores[item] - raw_scores[item]),
                "normalized_popularity": float(pop_np[item]),
                "is_tail": int(pop_np[item] <= tail_thr),
                "is_test_positive": int(item in positives),
                "user_gate": float(u_gate_np[u]),
                "item_gate": float(i_gate_np[item]),
                "conformity_score": float(conform[pos]),
                "adaptive_residual": float(residual[pos]),
                "beta": beta,
            })

    csv_path = os.path.join(args.output_dir, f"{args.dataset_name}_{args.split}_case_study.csv")
    write_csv(csv_path, rows)
    print(f"Case-study details: {csv_path}")
    print(f"Selected users: {selected}")


if __name__ == "__main__":
    main()

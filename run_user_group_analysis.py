# coding: utf-8
"""User heterogeneity analysis based on historical average popularity.

For each user u, the grouping variable is
    mean_{i in H_u} p_i,
which matches the paper's user historical-popularity statistic. Users are split
into equal-frequency Low/Medium/High groups by default. The script reports raw
and calibrated ranking metrics, average user gate, and recommendation-item gate.
"""

from __future__ import annotations

import argparse
import math
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
    rank_metrics_single,
    topk_for_users,
    write_csv,
)


def equal_frequency_groups(values: Dict[int, float], group_count: int) -> Dict[int, str]:
    if group_count < 2:
        raise ValueError("group_count must be at least 2")
    labels = ["Low", "Medium", "High"] if group_count == 3 else [f"G{i + 1}" for i in range(group_count)]
    ordered = sorted(values.items(), key=lambda x: (x[1], x[0]))
    groups: Dict[int, str] = {}
    n = len(ordered)
    for rank, (u, _) in enumerate(ordered):
        idx = min(group_count - 1, int(rank * group_count / max(n, 1)))
        groups[u] = labels[idx]
    return groups


def summarize_group(
    users: Sequence[int],
    recs: Dict[int, np.ndarray],
    eval_dict: Dict[int, List[int]],
    pop_np: np.ndarray,
    tail_thr: float,
    u_gate_np: np.ndarray,
    i_gate_np: np.ndarray,
    k: int,
) -> Dict[str, float]:
    hr_sum = ndcg_sum = arp_sum = tail_sum = user_gate_sum = item_gate_sum = 0.0
    count = 0
    for u in users:
        positives = set(eval_dict.get(u, []))
        if not positives or u not in recs:
            continue
        top = recs[u][:k]
        hr, ndcg = rank_metrics_single(top, positives, k)
        hr_sum += hr
        ndcg_sum += ndcg
        arp_sum += float(np.mean(pop_np[top]))
        tail_sum += float(np.mean(pop_np[top] <= tail_thr))
        user_gate_sum += float(u_gate_np[u])
        item_gate_sum += float(np.mean(i_gate_np[top]))
        count += 1
    denom = max(count, 1)
    return {
        "users": count,
        f"HR@{k}": hr_sum / denom,
        f"NDCG@{k}": ndcg_sum / denom,
        f"ARP@{k}": arp_sum / denom,
        f"TAIL@{k}": tail_sum / denom,
        "mean_user_gate": user_gate_sum / denom,
        "mean_recommended_item_gate": item_gate_sum / denom,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MPRC user-group analysis")
    add_common_data_model_args(parser)
    add_training_args(parser)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./paper_results/user_groups")
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--group_count", type=int, default=3)
    parser.add_argument("--calibrated_beta", type=float, default=None,
                        help="Default: beta stored in the checkpoint.")
    args = normalize_args(parser.parse_args())

    os.makedirs(args.output_dir, exist_ok=True)
    base = load_base_module(args.base_file)
    args, data, model, ckpt = load_model_from_checkpoint(base, args.checkpoint, args)
    device = torch.device(args.device)
    beta = float(args.calibrated_beta if args.calibrated_beta is not None else ckpt.get("best_beta", 0.0))

    eval_dict = data.valid_dict if args.split == "valid" else data.test_dict
    users = sorted(u for u, positives in eval_dict.items() if positives)
    pop_np = data.pop_feat.cpu().numpy()
    tail_thr = float(np.quantile(pop_np, 0.80))

    historical_mean: Dict[int, float] = {}
    for u in users:
        items = data.train_dict.get(u, [])
        historical_mean[u] = float(np.mean(pop_np[items])) if items else 0.0
    user_group = equal_frequency_groups(historical_mean, args.group_count)

    model.eval()
    with torch.no_grad():
        reps = model.compute_all()
        u_gate_np = reps[6].detach().cpu().numpy()
        i_gate_np = reps[7].detach().cpu().numpy()

    raw_recs = topk_for_users(model, data, users, args.split, 0.0, args.k, args.eval_user_batch, device)
    cal_recs = topk_for_users(model, data, users, args.split, beta, args.k, args.eval_user_batch, device)

    labels = ["Low", "Medium", "High"] if args.group_count == 3 else [f"G{i + 1}" for i in range(args.group_count)]
    rows: List[Dict[str, object]] = []
    group_metrics: Dict[Tuple[str, str], Dict[str, float]] = {}
    for label in labels:
        group_users = [u for u in users if user_group[u] == label]
        mean_hist = float(np.mean([historical_mean[u] for u in group_users])) if group_users else 0.0
        for mode, mode_beta, recs in [("Raw", 0.0, raw_recs), ("Calibrated", beta, cal_recs)]:
            metrics = summarize_group(
                group_users, recs, eval_dict, pop_np, tail_thr, u_gate_np, i_gate_np, args.k
            )
            group_metrics[(label, mode)] = metrics
            row: Dict[str, object] = {
                "dataset": args.dataset_name,
                "split": args.split,
                "group": label,
                "mode": mode,
                "beta": mode_beta,
                "mean_historical_popularity": mean_hist,
            }
            row.update(metrics)
            rows.append(row)

    # Add within-group calibrated-minus-raw differences.
    for row in rows:
        if row["mode"] != "Calibrated":
            continue
        label = str(row["group"])
        raw = group_metrics[(label, "Raw")]
        row[f"delta_NDCG@{args.k}"] = float(row[f"NDCG@{args.k}"]) - raw[f"NDCG@{args.k}"]
        row[f"ARP@{args.k}_reduction"] = raw[f"ARP@{args.k}"] - float(row[f"ARP@{args.k}"])
        row[f"TAIL@{args.k}_increase"] = float(row[f"TAIL@{args.k}"]) - raw[f"TAIL@{args.k}"]

    csv_path = os.path.join(args.output_dir, f"{args.dataset_name}_{args.split}_user_groups.csv")
    write_csv(csv_path, rows)
    print(f"User-group analysis: {csv_path}")


if __name__ == "__main__":
    main()

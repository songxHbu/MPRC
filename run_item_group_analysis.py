# coding: utf-8
"""Head/Middle/Tail target-item and exposure analysis for MPRC.

The default boundaries are configurable. By default, items at or below the 80th
popularity percentile are Tail, items between the 80th and 95th percentiles are
Middle, and the top 5 percent are Head. Change the quantiles to match the exact
paper protocol when necessary.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Sequence

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


def item_group_map(pop: np.ndarray, tail_q: float, head_q: float) -> Dict[int, str]:
    if not 0.0 < tail_q < head_q < 1.0:
        raise ValueError("Require 0 < tail_quantile < head_quantile < 1")
    tail_thr = float(np.quantile(pop, tail_q))
    head_thr = float(np.quantile(pop, head_q))
    out: Dict[int, str] = {}
    for i, value in enumerate(pop):
        if value <= tail_thr:
            out[i] = "Tail"
        elif value <= head_thr:
            out[i] = "Middle"
        else:
            out[i] = "Head"
    return out


def summarize_target_group(
    users: Sequence[int],
    recs: Dict[int, np.ndarray],
    eval_dict: Dict[int, List[int]],
    group_map: Dict[int, str],
    target_group: str,
    k: int,
) -> Dict[str, float]:
    hr_sum = ndcg_sum = 0.0
    count = 0
    for u in users:
        positives = {int(i) for i in eval_dict.get(u, []) if group_map[int(i)] == target_group}
        if not positives or u not in recs:
            continue
        hr, ndcg = rank_metrics_single(recs[u], positives, k)
        hr_sum += hr
        ndcg_sum += ndcg
        count += 1
    denom = max(count, 1)
    return {"users": count, f"HR@{k}": hr_sum / denom, f"NDCG@{k}": ndcg_sum / denom}


def exposure_share(recs: Dict[int, np.ndarray], group_map: Dict[int, str], group: str, k: int) -> float:
    total = 0
    matched = 0
    for items in recs.values():
        for item in items[:k]:
            total += 1
            matched += int(group_map[int(item)] == group)
    return matched / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="MPRC item popularity group analysis")
    add_common_data_model_args(parser)
    add_training_args(parser)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./paper_results/item_groups")
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--tail_quantile", type=float, default=0.80)
    parser.add_argument("--head_quantile", type=float, default=0.95)
    parser.add_argument("--calibrated_beta", type=float, default=None)
    args = normalize_args(parser.parse_args())

    os.makedirs(args.output_dir, exist_ok=True)
    base = load_base_module(args.base_file)
    args, data, model, ckpt = load_model_from_checkpoint(base, args.checkpoint, args)
    device = torch.device(args.device)
    beta = float(args.calibrated_beta if args.calibrated_beta is not None else ckpt.get("best_beta", 0.0))

    eval_dict = data.valid_dict if args.split == "valid" else data.test_dict
    users = sorted(u for u, positives in eval_dict.items() if positives)
    pop = data.pop_feat.cpu().numpy()
    groups = item_group_map(pop, args.tail_quantile, args.head_quantile)

    raw_recs = topk_for_users(model, data, users, args.split, 0.0, args.k, args.eval_user_batch, device)
    cal_recs = topk_for_users(model, data, users, args.split, beta, args.k, args.eval_user_batch, device)

    rows: List[Dict[str, object]] = []
    for mode, mode_beta, recs in [("Raw", 0.0, raw_recs), ("Calibrated", beta, cal_recs)]:
        for group in ["Head", "Middle", "Tail"]:
            metrics = summarize_target_group(users, recs, eval_dict, groups, group, args.k)
            row: Dict[str, object] = {
                "dataset": args.dataset_name,
                "split": args.split,
                "mode": mode,
                "beta": mode_beta,
                "item_group": group,
                "tail_quantile": args.tail_quantile,
                "head_quantile": args.head_quantile,
                "exposure_share": exposure_share(recs, groups, group, args.k),
            }
            row.update(metrics)
            rows.append(row)

    csv_path = os.path.join(args.output_dir, f"{args.dataset_name}_{args.split}_item_groups.csv")
    write_csv(csv_path, rows)
    print(f"Item-group analysis: {csv_path}")


if __name__ == "__main__":
    main()

# coding: utf-8
"""Evaluate beta sensitivity from an existing MPRC checkpoint.

Outputs validation/test CSV files and three separate plots for NDCG, ARP and
TAIL. The checkpoint is never modified.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch

from mprc_experiment_utils import (
    add_common_data_model_args,
    add_training_args,
    load_base_module,
    load_model_from_checkpoint,
    normalize_args,
    parse_float_list,
    parse_int_list,
    write_csv,
)


def save_metric_plot(df: pd.DataFrame, metric: str, path: str) -> None:
    plt.figure(figsize=(6.4, 4.4))
    for split, group in df.groupby("split"):
        group = group.sort_values("beta")
        plt.plot(group["beta"], group[metric], marker="o", label=split)
    plt.xlabel(r"Calibration strength $\beta$")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="MPRC beta sensitivity")
    add_common_data_model_args(parser)
    add_training_args(parser)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./paper_results/beta_sensitivity")
    parser.add_argument("--splits", type=str, default="valid,test")
    parser.add_argument("--save_plots", action=argparse.BooleanOptionalAction, default=True)
    args = normalize_args(parser.parse_args())

    os.makedirs(args.output_dir, exist_ok=True)
    base = load_base_module(args.base_file)
    args, data, model, ckpt = load_model_from_checkpoint(base, args.checkpoint, args)
    device = torch.device(args.device)
    top_ks = parse_int_list(args.topks)
    betas = parse_float_list(args.beta_candidates)
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]

    rows: List[Dict[str, object]] = []
    raw_by_split: Dict[str, Dict[str, float]] = {}
    for split in splits:
        for beta in betas:
            metrics = base.evaluate_full_sort(
                model,
                data,
                split=split,
                beta=beta,
                top_ks=top_ks,
                user_batch_size=args.eval_user_batch,
                device=device,
                show_progress=(split == "test"),
            )
            if abs(beta) < 1e-12:
                raw_by_split[split] = metrics
            row: Dict[str, object] = {
                "dataset": args.dataset_name,
                "split": split,
                "beta": beta,
                "checkpoint_epoch": int(ckpt.get("epoch", -1)),
                "checkpoint_best_beta": float(ckpt.get("best_beta", 0.0)),
            }
            row.update({k: float(v) for k, v in metrics.items() if k not in {"users", "beta"}})
            row["users"] = int(metrics.get("users", 0))
            rows.append(row)

    key_k = max(top_ks)
    for row in rows:
        raw = raw_by_split.get(str(row["split"]))
        if raw is None:
            continue
        raw_ndcg = raw[f"NDCG@{key_k}"]
        raw_arp = raw[f"ARP@{key_k}"]
        raw_tail = raw[f"TAIL@{key_k}"]
        row[f"NDCG@{key_k}_change_pct"] = 100.0 * (float(row[f"NDCG@{key_k}"]) - raw_ndcg) / max(abs(raw_ndcg), 1e-12)
        row[f"ARP@{key_k}_reduction_pct"] = 100.0 * (raw_arp - float(row[f"ARP@{key_k}"])) / max(abs(raw_arp), 1e-12)
        row[f"TAIL@{key_k}_increase_pct"] = 100.0 * (float(row[f"TAIL@{key_k}"]) - raw_tail) / max(abs(raw_tail), 1e-12)

    csv_path = os.path.join(args.output_dir, f"{args.dataset_name}_beta_sensitivity.csv")
    write_csv(csv_path, rows)

    if args.save_plots:
        df = pd.DataFrame(rows)
        for metric, suffix in [
            (f"NDCG@{key_k}", "ndcg"),
            (f"ARP@{key_k}", "arp"),
            (f"TAIL@{key_k}", "tail"),
        ]:
            save_metric_plot(
                df,
                metric,
                os.path.join(args.output_dir, f"{args.dataset_name}_beta_{suffix}.png"),
            )

    print(f"Beta sensitivity results: {csv_path}")


if __name__ == "__main__":
    main()

# coding: utf-8
"""Repeat an MPRC experiment over multiple random seeds and report mean/std."""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch

from mprc_experiment_utils import (
    VARIANT_DISPLAY,
    add_common_data_model_args,
    add_training_args,
    build_data,
    load_base_module,
    metric_dict_to_row,
    normalize_args,
    parse_int_list,
    train_experiment,
    write_csv,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="MPRC multi-seed experiment")
    add_common_data_model_args(parser)
    add_training_args(parser)
    parser.add_argument("--variant", type=str, default="full")
    parser.add_argument("--seeds", type=str, default="2024,2025,2026")
    parser.add_argument("--report_mode", type=str, default="both", choices=["raw", "calibrated", "both"])
    parser.add_argument("--output_dir", type=str, default="./paper_results/multi_seed")
    args = normalize_args(parser.parse_args())

    os.makedirs(args.output_dir, exist_ok=True)
    base = load_base_module(args.base_file)
    device = torch.device(args.device)
    data = build_data(base, args, device)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    rows: List[Dict[str, object]] = []
    details: Dict[str, object] = {}
    for seed in seeds:
        run_args = argparse.Namespace(**vars(args))
        run_args.seed = seed
        ckpt = os.path.join(args.output_dir, f"{args.dataset_name}_{args.variant}_seed{seed}.pt")
        result = train_experiment(
            base=base,
            data=data,
            args=run_args,
            variant=args.variant,
            checkpoint_path=ckpt,
        )
        details[str(seed)] = result

        if args.report_mode in {"raw", "both"}:
            rows.append(
                metric_dict_to_row(
                    args.dataset_name,
                    "mprc_raw" if args.variant == "full" else args.variant,
                    "test_raw",
                    0.0,
                    result["test_raw"],
                    seed,
                )
            )
        if args.report_mode in {"calibrated", "both"}:
            rows.append(
                metric_dict_to_row(
                    args.dataset_name,
                    "mprc_calibrated" if args.variant == "full" else args.variant,
                    "test_calibrated",
                    float(result["best_beta"]),
                    result["test_selected"],
                    seed,
                )
            )

    numeric_metrics = sorted({k for row in rows for k, v in row.items() if isinstance(v, (int, float)) and k not in {"seed", "users"}})
    summary_rows: List[Dict[str, object]] = []
    modes = sorted({str(row["split"]) for row in rows})
    for mode in modes:
        subset = [row for row in rows if str(row["split"]) == mode]
        summary: Dict[str, object] = {
            "dataset": args.dataset_name,
            "variant": VARIANT_DISPLAY.get(args.variant, args.variant),
            "mode": mode,
            "num_seeds": len(subset),
        }
        for metric in numeric_metrics:
            vals = [float(row[metric]) for row in subset if metric in row]
            if not vals:
                continue
            summary[f"{metric}_mean"] = float(np.mean(vals))
            summary[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary_rows.append(summary)

    seed_csv = os.path.join(args.output_dir, f"{args.dataset_name}_{args.variant}_per_seed.csv")
    summary_csv = os.path.join(args.output_dir, f"{args.dataset_name}_{args.variant}_mean_std.csv")
    json_path = os.path.join(args.output_dir, f"{args.dataset_name}_{args.variant}_multi_seed.json")
    write_csv(seed_csv, rows)
    write_csv(summary_csv, summary_rows)
    write_json(json_path, {"rows": rows, "summary": summary_rows, "details": details})

    print(f"Per-seed results: {seed_csv}")
    print(f"Mean/std results: {summary_csv}")


if __name__ == "__main__":
    main()

# coding: utf-8
"""Run the dataset-specific ablation study used in the paper.

The original training file remains unchanged. This script trains the full model
once, evaluates Raw and Calibrated inference from the same checkpoint, then
trains representation and calibration ablations under explicit protocols.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

from mprc_experiment_utils import (
    PANEL_A_VARIANTS,
    PANEL_B_VARIANTS,
    VARIANT_DISPLAY,
    add_common_data_model_args,
    add_training_args,
    build_data,
    load_base_module,
    metric_dict_to_row,
    normalize_args,
    train_experiment,
    write_csv,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="MPRC paper ablation study")
    add_common_data_model_args(parser)
    add_training_args(parser)
    parser.add_argument("--output_dir", type=str, default="./paper_results/ablation")
    parser.add_argument(
        "--panel_a_variants",
        type=str,
        default=",".join(PANEL_A_VARIANTS),
        help="Comma-separated representation ablations.",
    )
    parser.add_argument(
        "--panel_b_variants",
        type=str,
        default=",".join(PANEL_B_VARIANTS),
        help="Comma-separated calibration ablations.",
    )
    parser.add_argument(
        "--shared_beta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the beta selected by the full model for every Panel-B variant.",
    )
    args = normalize_args(parser.parse_args())

    os.makedirs(args.output_dir, exist_ok=True)
    base = load_base_module(args.base_file)
    device = __import__("torch").device(args.device)
    base.set_seed(args.seed)
    data = build_data(base, args, device)

    rows: List[Dict[str, object]] = []
    details: Dict[str, object] = {}

    # Train the complete model only once. Raw and calibrated rows are evaluated
    # from the same learned parameters, which gives a fair inference comparison.
    full_ckpt = os.path.join(args.output_dir, f"{args.dataset_name}_full_seed{args.seed}.pt")
    full_result = train_experiment(
        base=base,
        data=data,
        args=args,
        variant="full",
        checkpoint_path=full_ckpt,
    )
    shared_beta = float(full_result["best_beta"])
    details["full"] = full_result

    rows.append(
        metric_dict_to_row(
            args.dataset_name,
            "mprc_raw",
            "test",
            0.0,
            full_result["test_raw"],
            args.seed,
            panel="A",
        )
    )
    rows.append(
        metric_dict_to_row(
            args.dataset_name,
            "mprc_calibrated",
            "test",
            shared_beta,
            full_result["test_selected"],
            args.seed,
            panel="B",
        )
    )

    panel_a = [x.strip() for x in args.panel_a_variants.split(",") if x.strip()]
    for variant in panel_a:
        ckpt = os.path.join(args.output_dir, f"{args.dataset_name}_{variant}_seed{args.seed}.pt")
        result = train_experiment(
            base=base,
            data=data,
            args=args,
            variant=variant,
            checkpoint_path=ckpt,
            selection_betas=[0.0],
            fixed_eval_beta=0.0,
        )
        details[variant] = result
        rows.append(
            metric_dict_to_row(
                args.dataset_name,
                variant,
                "test",
                0.0,
                result["test_raw"],
                args.seed,
                panel="A",
            )
        )

    panel_b = [x.strip() for x in args.panel_b_variants.split(",") if x.strip()]
    for variant in panel_b:
        ckpt = os.path.join(args.output_dir, f"{args.dataset_name}_{variant}_seed{args.seed}.pt")
        fixed_beta = shared_beta if args.shared_beta else None
        result = train_experiment(
            base=base,
            data=data,
            args=args,
            variant=variant,
            checkpoint_path=ckpt,
            fixed_eval_beta=fixed_beta,
        )
        details[variant] = result
        rows.append(
            metric_dict_to_row(
                args.dataset_name,
                variant,
                "test",
                float(result["best_beta"]),
                result["test_selected"],
                args.seed,
                panel="B",
            )
        )

    # Stable paper order.
    order = PANEL_A_VARIANTS + ["mprc_raw"] + PANEL_B_VARIANTS + ["mprc_calibrated"]
    pos = {name: idx for idx, name in enumerate(order)}
    rows.sort(key=lambda row: (str(row["panel"]), pos.get(str(row["variant_key"]), 999)))

    csv_path = os.path.join(args.output_dir, f"{args.dataset_name}_ablation_seed{args.seed}.csv")
    json_path = os.path.join(args.output_dir, f"{args.dataset_name}_ablation_seed{args.seed}.json")
    write_csv(csv_path, rows)
    write_json(
        json_path,
        {
            "dataset": args.dataset_name,
            "seed": args.seed,
            "shared_beta": shared_beta,
            "shared_beta_protocol": bool(args.shared_beta),
            "rows": rows,
            "details": details,
        },
    )

    print("\nAblation study completed")
    print(f"Shared beta selected by full model: {shared_beta:.6f}")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    report_k = max(int(x.strip()) for x in args.topks.split(",") if x.strip())
    for row in rows:
        print(
            f"[{row['panel']}] {row['variant']}: "
            f"NDCG@{report_k}={row.get(f'NDCG@{report_k}', float('nan')):.4f}, "
            f"ARP@{report_k}={row.get(f'ARP@{report_k}', float('nan')):.4f}, "
            f"TAIL@{report_k}={row.get(f'TAIL@{report_k}', float('nan')):.4f}"
        )


if __name__ == "__main__":
    main()

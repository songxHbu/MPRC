# coding: utf-8
"""Orchestrate the MPRC paper experiments for multiple datasets.

The suite intentionally invokes the specialized scripts as separate processes so
that every experiment has its own logs and output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent


def run_command(cmd: List[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN:", " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed; inspect {log_path}")


def common_args(dataset: Dict[str, object], config: Dict[str, object]) -> List[str]:
    args = [
        "--data_path", str(dataset["data_path"]),
        "--inter_file", str(dataset["inter_file"]),
        "--text_feat_file", str(dataset.get("text_feat_file", "text_feat.npy")),
        "--image_feat_file", str(dataset.get("image_feat_file", "image_feat.npy")),
        "--dataset_name", str(dataset["name"]),
        "--device", str(config.get("device", "cuda")),
    ]
    for key, value in dict(config.get("common_overrides", {})).items():
        args.extend([f"--{key}", str(value)])
    for key, value in dict(dataset.get("overrides", {})).items():
        args.extend([f"--{key}", str(value)])
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MPRC paper experiment suite")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tasks", type=str, default="ablation,beta,user_groups,item_groups,case_study")
    parser.add_argument("--output_root", type=str, default="./paper_results/suite")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    tasks = {x.strip() for x in args.tasks.split(",") if x.strip()}
    output_root = Path(args.output_root).resolve()
    python = sys.executable

    for dataset in config["datasets"]:
        name = str(dataset["name"])
        ds_root = output_root / name
        common = common_args(dataset, config)

        ablation_dir = ds_root / "ablation"
        full_ckpt = ablation_dir / f"{name}_full_seed{config.get('seed', 2024)}.pt"

        if "ablation" in tasks:
            cmd = [python, str(ROOT / "run_ablation_study.py"), *common,
                   "--seed", str(config.get("seed", 2024)),
                   "--output_dir", str(ablation_dir)]
            run_command(cmd, ds_root / "logs" / "ablation.log")

        if not full_ckpt.exists() and tasks - {"ablation"}:
            raise FileNotFoundError(
                f"Full checkpoint not found: {full_ckpt}. Run the ablation task first."
            )

        eval_tasks = {
            "beta": ("run_beta_sensitivity.py", "beta_sensitivity"),
            "user_groups": ("run_user_group_analysis.py", "user_groups"),
            "item_groups": ("run_item_group_analysis.py", "item_groups"),
            "case_study": ("run_case_study.py", "case_study"),
        }
        for task, (script, folder) in eval_tasks.items():
            if task not in tasks:
                continue
            cmd = [python, str(ROOT / script), *common,
                   "--checkpoint", str(full_ckpt),
                   "--output_dir", str(ds_root / folder)]
            run_command(cmd, ds_root / "logs" / f"{task}.log")

        if "multi_seed" in tasks:
            cmd = [python, str(ROOT / "run_multi_seed.py"), *common,
                   "--seeds", str(config.get("seeds", "2024,2025,2026")),
                   "--output_dir", str(ds_root / "multi_seed")]
            run_command(cmd, ds_root / "logs" / "multi_seed.log")

    print(f"All requested tasks completed under: {output_root}")


if __name__ == "__main__":
    main()

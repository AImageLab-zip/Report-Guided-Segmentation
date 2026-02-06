"""
Summarize QaTaCov2D sweep results into Excel.

Expected run folder naming (example):
  baseline_64-5_AdamW_S2_A123456_7

Each run folder contains:
  test_metrics.csv with columns:
    epoch,DSC_Covid,DSC_mean,HD95_Covid,HD95_mean

Outputs:
  1) results_per_run.xlsx  (one row per run)
  2) results_avg_over_splits.xlsx (mean/std over splits for each config)

Usage:
  python summarize_qatacov_results.py \
    --runs_root /leonardo_work/IscrC_narc2/reports_project/trained_models/QaTaCov2D \
    --out_dir   /leonardo/home/userexternal/ldelgaud/projects/Report-Guided-Segmentation/summaries \
    --job_id    123456
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


RUN_RE = re.compile(
    r"""
    ^baseline_                      # prefix
    (?P<bs>\d+)-(?P<depth>\d+)_     # batch size and depth
    (?P<optim>SGD|AdamW)_           # optimizer
    S(?P<split>\d+)_                # split id
    A(?P<job>\d+)_(?P<task>\d+)     # array job id and task id
    $""",
    re.VERBOSE,
)


def read_last_metrics(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Empty CSV: {csv_path}")
    last = df.iloc[-1]  # use last epoch row
    return {
        "DSC_Covid": float(last.get("DSC_Covid", np.nan)),
        "DSC_mean": float(last.get("DSC_mean", np.nan)),
        "HD95_Covid": float(last.get("HD95_Covid", np.nan)),
        "HD95_mean": float(last.get("HD95_mean", np.nan)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, required=True, help="Folder containing run subfolders")
    ap.add_argument("--out_dir", type=str, required=True, help="Where to write Excel summaries")
    ap.add_argument("--job_id", type=str, default=None, help="If set, only include runs with A<job_id>_<task>")
    ap.add_argument("--round", dest="round_n", type=int, default=4, help="Decimal rounding for metrics")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []

    for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()]):
        name = run_dir.name

        # Optional filter by array job id
        if args.job_id is not None and f"_A{args.job_id}_" not in f"_{name}_":
            continue

        m = RUN_RE.match(name)
        if not m:
            # skip folders not from this naming scheme
            continue

        csv_path = run_dir / "test_metrics.csv"
        if not csv_path.exists():
            missing.append(str(csv_path))
            continue

        try:
            metrics = read_last_metrics(csv_path)
        except Exception as e:
            missing.append(f"{csv_path} ({e})")
            continue

        row = {
            "run_name": name,
            "batch_size": int(m.group("bs")),
            "depth": int(m.group("depth")),
            "optimizer": m.group("optim"),
            "split": int(m.group("split")),
            "job_array_id": int(m.group("job")),
            "task_id": int(m.group("task")),
            **metrics,
            "path": str(run_dir),
        }
        rows.append(row)

    if not rows:
        raise SystemExit(
            "No runs found. Check --runs_root, --job_id, and folder naming.\n"
            f"runs_root={runs_root}"
        )

    df = pd.DataFrame(rows)

    # Round metrics
    metric_cols = ["DSC_Covid", "DSC_mean", "HD95_Covid", "HD95_mean"]
    df[metric_cols] = df[metric_cols].round(args.round_n)

    # --------------------------
    # 1) Per-run Excel
    # --------------------------
    df_per_run = df.sort_values(["job_array_id", "task_id"]).set_index("run_name")
    out1 = out_dir / f"results_per_run_{row['job_array_id']}.xlsx"
    with pd.ExcelWriter(out1, engine="openpyxl") as w:
        df_per_run.to_excel(w, sheet_name="per_run")
        # optional: missing sheet
        if missing:
            pd.DataFrame({"missing_or_failed": missing}).to_excel(w, sheet_name="missing", index=False)

    # --------------------------
    # 2) Average + std over splits per configuration
    #    (ignore job/task, group by config only)
    # --------------------------
    group_cols = ["batch_size", "depth", "optimizer"]

    agg = df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()

    # Flatten MultiIndex columns: DSC_Covid_mean, DSC_Covid_std, ...
    agg.columns = [
        "_".join([c for c in col if c]).rstrip("_") if isinstance(col, tuple) else col
        for col in agg.columns
    ]

    # Round aggregated metrics
    for c in agg.columns:
        if any(c.startswith(m + "_") for m in metric_cols):
            agg[c] = agg[c].round(args.round_n)

    # Add how many splits contributed (useful if something missing)
    n_splits = df.groupby(group_cols)["split"].nunique().reset_index(name="n_splits")
    agg = agg.merge(n_splits, on=group_cols, how="left")

    out2 = out_dir / f"results_avg_over_splits_{row['job_array_id']}.xlsx"
    with pd.ExcelWriter(out2, engine="openpyxl") as w:
        agg.sort_values(group_cols).to_excel(w, sheet_name="avg_std_over_splits", index=False)

    print(f"Wrote: {out1}")
    print(f"Wrote: {out2}")
    if missing:
        print(f"Warning: {len(missing)} runs missing/failed CSV. See 'missing' sheet in {out1}.")


if __name__ == "__main__":
    main()

"""
Summarize QaTaCov2D results into Excel (compatible with baseline + textft runs).

Supported run folder naming:

1) Baseline:
   baseline_64-5_AdamW_S2_A123456_7

2) Text fine-tuning:
   textft_from_baseline_32-3_SGD_S1_A33634707_0_AdamW_l0.1-V1_A33643046_2

Each run folder contains:
  test_metrics.csv with columns:
    epoch,DSC_Covid,DSC_mean,HD95_Covid,HD95_mean

Outputs:
  1) results_per_run_<JOB>.xlsx
  2) results_avg_over_splits_<JOB>.xlsx

Usage:
  python summarize_qatacov_results_any.py \
    --runs_root /path/to/runs \
    --out_dir   /path/to/out \
    --job_id    123456
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Optional, Dict



BASELINE_RE = re.compile(
    r"""
    ^baseline_
    (?P<bs>\d+)-(?P<depth>\d+)_
    (?P<optim>SGD|AdamW)_
    S(?P<split>\d+)_
    A(?P<job>\d+)_(?P<task>\d+)
    $
    """,
    re.VERBOSE,
)

TEXTFT_RE = re.compile(
    r"""
    ^(?P<tag>textft_from_baseline)_
    (?P<bs>\d+)-(?P<depth>\d+)_
    (?P<base_optim>SGD|AdamW)_
    S(?P<split>\d+)_
    A(?P<base_job>\d+)_(?P<base_task>\d+)_
    (?P<ft_optim>SGD|AdamW)_
    l(?P<lambda>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)      # float or sci
    -V(?P<version>\d+)_
    A(?P<ft_job>\d+)_(?P<ft_task>\d+)
    $
    """,
    re.VERBOSE,
)


def read_last_metrics(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Empty CSV: {csv_path}")
    last = df.iloc[-1]
    return {
        "DSC_Covid": float(last.get("DSC_Covid", np.nan)),
        "DSC_mean": float(last.get("DSC_mean", np.nan)),
        "HD95_Covid": float(last.get("HD95_Covid", np.nan)),
        "HD95_mean": float(last.get("HD95_mean", np.nan)),
    }


def parse_run_name(name: str) -> Optional[Dict]:
    """
    Return a unified dict of parsed fields, or None if not matching.
    """
    m = BASELINE_RE.match(name)
    if m:
        return {
            "mode": "baseline",
            "tag": "baseline",
            "batch_size": int(m.group("bs")),
            "depth": int(m.group("depth")),
            "split": int(m.group("split")),
            "optimizer": m.group("optim"),
            "job_array_id": int(m.group("job")),
            "task_id": int(m.group("task")),
            # textft-specific fields kept for schema compatibility
            "base_optimizer": "",
            "ft_optimizer": "",
            "lambda": np.nan,
            "loss_version": np.nan,
            "base_job_array_id": np.nan,
            "base_task_id": np.nan,
            "ft_job_array_id": np.nan,
            "ft_task_id": np.nan,
        }

    m = TEXTFT_RE.match(name)
    if m:
        lam_raw = m.group("lambda")
        try:
            lam_val = float(lam_raw)
        except Exception:
            lam_val = np.nan

        return {
            "mode": "textft",
            "tag": m.group("tag"),
            "batch_size": int(m.group("bs")),
            "depth": int(m.group("depth")),
            "split": int(m.group("split")),
            # baseline-vs-ft optimizers
            "optimizer": "",  # not used in textft grouping
            "base_optimizer": m.group("base_optim"),
            "ft_optimizer": m.group("ft_optim"),
            "lambda": lam_val,
            "loss_version": int(m.group("version")),
            # baseline job/task + finetune job/task
            "base_job_array_id": int(m.group("base_job")),
            "base_task_id": int(m.group("base_task")),
            "ft_job_array_id": int(m.group("ft_job")),
            "ft_task_id": int(m.group("ft_task")),
            # for compatibility with output sorting
            "job_array_id": int(m.group("ft_job")),
            "task_id": int(m.group("ft_task")),
        }

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--job_id", type=str, default=None, help="Filter: include runs containing _A<job_id>_")
    ap.add_argument("--round", dest="round_n", type=int, default=4)
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []

    for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()]):
        name = run_dir.name

        if args.job_id is not None and f"_A{args.job_id}_" not in f"_{name}_":
            continue

        parsed = parse_run_name(name)
        if parsed is None:
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

        rows.append(
            {
                "run_name": name,
                **parsed,
                **metrics,
                "path": str(run_dir),
            }
        )

    if not rows:
        raise SystemExit(
            "No runs found. Check --runs_root / --job_id and folder naming.\n"
            f"runs_root={runs_root}"
        )

    df = pd.DataFrame(rows)

    metric_cols = ["DSC_Covid", "DSC_mean", "HD95_Covid", "HD95_mean"]
    df[metric_cols] = df[metric_cols].round(args.round_n)

    job_ids = sorted(df["job_array_id"].dropna().unique().tolist())
    job_suffix = str(job_ids[0]) if len(job_ids) == 1 else "MULTI"

    # --------------------------
    # 1) Per-run Excel
    # --------------------------
    df_per_run = df.sort_values(["mode", "job_array_id", "task_id"]).set_index("run_name")
    out1 = out_dir / f"results_per_run_{parsed['mode']}_{job_suffix}.xlsx"
    with pd.ExcelWriter(out1, engine="openpyxl") as w:
        df_per_run.to_excel(w, sheet_name="per_run")
        if missing:
            pd.DataFrame({"missing_or_failed": missing}).to_excel(w, sheet_name="missing", index=False)

    # --------------------------
    # 2) Average + std over splits per configuration
    # --------------------------
    baseline_group = ["mode", "batch_size", "depth", "optimizer"]
    textft_group = ["mode", "batch_size", "depth", "base_optimizer", "ft_optimizer", "lambda", "loss_version"]

    # Build a single "group_key" that lets us group both modes in one dataframe
    df["group_key"] = np.where(
        df["mode"] == "baseline",
        df[baseline_group].astype(str).agg("|".join, axis=1),
        df[textft_group].astype(str).agg("|".join, axis=1),
    )

    # Keep representative config columns (for readability)
    rep_cols = [
        "mode", "batch_size", "depth", "optimizer",
        "base_optimizer", "ft_optimizer", "lambda", "loss_version"
    ]

    rep = df.groupby("group_key", dropna=False)[rep_cols].first().reset_index()

    agg = df.groupby("group_key", dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = [
        "_".join([c for c in col if c]).rstrip("_") if isinstance(col, tuple) else col
        for col in agg.columns
    ]

    for c in agg.columns:
        if any(c.startswith(m + "_") for m in metric_cols):
            agg[c] = agg[c].round(args.round_n)

    n_splits = df.groupby("group_key")["split"].nunique().reset_index(name="n_splits")

    out_df = rep.merge(agg, on="group_key", how="left").merge(n_splits, on="group_key", how="left")

    # nicer ordering
    out_df = out_df.drop(columns=["group_key"]).sort_values(["mode", "batch_size", "depth"])

    out2 = out_dir / f"results_avg_over_splits_{parsed['mode']}_{job_suffix}.xlsx"
    with pd.ExcelWriter(out2, engine="openpyxl") as w:
        out_df.to_excel(w, sheet_name="avg_std_over_splits", index=False)

    print(f"Wrote: {out1}")
    print(f"Wrote: {out2}")
    if missing:
        print(f"Warning: {len(missing)} runs missing/failed CSV. See 'missing' sheet in {out1}.")


if __name__ == "__main__":
    main()

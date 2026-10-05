"""Phase-9 support: export result tables and plots for the thesis chapters.

    python -m experiments.export_thesis_tables
    python -m experiments.export_thesis_tables --out results/thesis --plots

Reads **only** ``results/experiment_log.parquet`` and writes CSV + LaTeX tables
plus the two headline figures.  Nothing is recomputed here, which is what makes
the Phase-9 gate checkable: every exported cell carries the number of logged
rows behind it, so a number in the thesis can be traced back to the log.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from core.io import load_results
from core.logging_utils import get_logger
from eval.metrics import price_of_robustness, summarise
from eval.significance_tests import paired_compare

log = get_logger(__name__)

MAIN_METRICS = [
    "erp", "seq_dev", "reliability_index", "expected_lateness",
    "expected_duration", "route_travel_time", "objective", "runtime_s",
]


def _similarity_metric(df: pd.DataFrame) -> str:
    """Prefer the official ERP; fall back to the labelled stand-in."""
    if "erp" in df.columns and df["erp"].notna().any():
        return "erp"
    log.warning(
        "no ERP values in the log: exporting seq_dev instead. Tables built this "
        "way must say so -- seq_dev is not the official route score."
    )
    return "seq_dev"


def table_main_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Table 1: every method, mean +/- sd over routes, all folds pooled."""
    present = [m for m in MAIN_METRICS if m in df.columns]
    g = df.groupby("method")[present].agg(["mean", "std", "count"])
    g.columns = ["_".join(c) for c in g.columns]
    return g.reset_index().sort_values("method")


def table_per_fold(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Table 2: leave-one-metro-out, mean per fold plus the spread across folds."""
    per = df.pivot_table(index="method", columns="fold", values=metric,
                         aggfunc="mean")
    per["fold_mean"] = per.mean(axis=1)
    per["fold_std"] = per.std(axis=1)
    per["n_folds"] = per.notna().sum(axis=1) - 2
    return per.reset_index()


def table_frontier(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Table 3: the RQ2 frontier -- similarity cost against reliability gain."""
    sweep = df[df["lam"].notna()].copy()
    if sweep["lam"].nunique() < 2:
        return pd.DataFrame()
    por = price_of_robustness(sweep, metric=metric)
    out = por.groupby("lam")[[metric, f"d_{metric}", "reliability_index",
                              "d_reliability", "expected_lateness"]].agg(
        ["mean", "std", "count"]
    )
    out.columns = ["_".join(c) for c in out.columns]
    return out.reset_index().sort_values("lam", ascending=False)


def table_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Table 4: certified optimality gaps, where a bound was computed."""
    d = df[df["optimality_gap"].notna()]
    if d.empty:
        return pd.DataFrame()
    g = d.groupby(["solver", "method"])["optimality_gap"].agg(
        ["mean", "median", "max", "count"]
    )
    return g.reset_index()


def plot_frontier(frontier: pd.DataFrame, metric: str, path: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed: skipping plots")
        return None
    if frontier.empty:
        return None

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.errorbar(
        frontier[f"d_{metric}_mean"], frontier["d_reliability_mean"],
        xerr=frontier[f"d_{metric}_std"] / np.sqrt(frontier[f"{metric}_count"]),
        yerr=frontier["d_reliability_std"] / np.sqrt(frontier[f"{metric}_count"]),
        fmt="o-", capsize=3,
    )
    for _, r in frontier.iterrows():
        ax.annotate(f"$\\lambda$={r['lam']:.2f}",
                    (r[f"d_{metric}_mean"], r["d_reliability_mean"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel(f"change in {metric} vs. pure imitation (lower is better)")
    ax.set_ylabel("change in reliability index")
    ax.set_title("Price of reliability")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_method_bars(main: pd.DataFrame, metric: str, path: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    col = f"{metric}_mean"
    if col not in main.columns:
        return None
    d = main.dropna(subset=[col]).sort_values(col)
    fig, ax = plt.subplots(figsize=(7.0, 0.4 * len(d) + 1.5))
    ax.barh(d["method"], d[col], xerr=d[f"{metric}_std"], capsize=3)
    ax.set_xlabel(f"{metric} (mean over routes)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, default=Path("results/experiment_log.parquet"))
    ap.add_argument("--out", type=Path, default=Path("results/thesis"))
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--latex", action="store_true", default=True)
    args = ap.parse_args(argv)

    df = load_results(args.log)
    if df.empty:
        log.error("%s is empty: run the experiments first", args.log)
        return 1
    df = df[df.get("error").isna()] if "error" in df.columns else df
    args.out.mkdir(parents=True, exist_ok=True)
    metric = _similarity_metric(df)

    tables = {
        "table1_main_comparison": table_main_comparison(df),
        "table2_per_fold": table_per_fold(df, metric),
        "table3_frontier": table_frontier(df, metric),
        "table4_optimality_gaps": table_gaps(df),
        "table5_significance": paired_compare(df, metric),
        "table6_summary_by_method_lambda": summarise(df),
    }
    written = {}
    for name, t in tables.items():
        if t is None or t.empty:
            log.warning("%s is empty: skipped", name)
            continue
        t.to_csv(args.out / f"{name}.csv", index=False)
        if args.latex:
            (args.out / f"{name}.tex").write_text(
                t.to_latex(index=False, float_format="%.3f")
            )
        written[name] = len(t)

    if args.plots:
        plot_frontier(tables["table3_frontier"], metric,
                      args.out / "fig_price_of_reliability.png")
        plot_method_bars(tables["table1_main_comparison"], metric,
                         args.out / "fig_method_comparison.png")

    provenance = {
        "log": str(args.log),
        "rows": int(len(df)),
        "routes": int(df["route_id"].nunique()),
        "methods": sorted(df["method"].dropna().unique().tolist()),
        "folds": sorted(df["fold"].dropna().unique().tolist()),
        "similarity_metric": metric,
        "erp_rows": int(df["erp"].notna().sum()) if "erp" in df else 0,
        "tables": written,
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

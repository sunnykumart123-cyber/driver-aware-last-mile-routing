"""RQ2: the price of reliability (§7, §9 Phase 8).

    python -m experiments.lambda_sweep_rq2 --folds Austin --limit 25
    python -m experiments.lambda_sweep_rq2 --all-folds --lams 1.0 0.9 0.75 0.5 0.25 0.0

For each route the same instance is solved at every lambda on the grid with
everything else held fixed -- same ambiguity set, same behaviour cost, same
search budget, same seed -- so the differences are attributable to lambda alone.

Outputs
-------
* rows in ``results/experiment_log.parquet`` tagged ``lambda_sweep``;
* ``results/rq2_price_of_robustness.parquet``: per-route paired differences
  against ``lambda = 1``, which is the price-of-robustness curve;
* paired Wilcoxon tests between adjacent lambda levels, with effect sizes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from core.config import METROS
from core.io import metro_slug
from core.logging_utils import get_logger
from eval.metrics import price_of_robustness, summarise
from eval.significance_tests import lambda_trend_test, paired_compare
from experiments.common import base_config, fold_instances, log_rows, make_runner, objective_for

log = get_logger(__name__)
DEFAULT_LAMS = (1.0, 0.9, 0.75, 0.5, 0.25, 0.1, 0.0)


def run_fold(metro: str, args) -> pd.DataFrame:
    cfg = base_config(metro, tag=args.tag, eps=args.eps, delta=args.delta,
                      scenarios=args.scenarios, time_limit=args.time_limit,
                      iterations=args.iterations)
    runner = make_runner(cfg)
    insts = fold_instances(args.processed, metro, args.on, args.limit, args.max_stops)
    fold = f"ho_{metro_slug(metro)}"
    rows: list[dict] = []
    for k, inst in enumerate(insts):
        for lam in args.lams:
            obj = objective_for(runner, inst, lam)
            sol = runner.solve(inst, obj, args.method)
            from eval.metrics import evaluate_solution

            rows.append(
                evaluate_solution(inst, sol, obj, runner.ctx, "mtp_lambda_sweep", fold)
            )
        if (k + 1) % 5 == 0:
            log.info("fold %s: %d/%d routes", metro, k + 1, len(insts))
    return log_rows(rows, cfg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--folds", nargs="*", default=["Austin"], choices=list(METROS))
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--on", default="test", choices=["test", "val", "train"])
    ap.add_argument("--lams", type=float, nargs="+", default=list(DEFAULT_LAMS))
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--scenarios", type=int, default=64)
    ap.add_argument("--method", default="alns", choices=["alns", "auto", "benders"])
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--max-stops", type=int, default=None)
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--tag", default="lambda_sweep")
    args = ap.parse_args(argv)

    folds = list(METROS) if args.all_folds else args.folds
    frames = [run_fold(m, args) for m in folds]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if df.empty:
        log.error("no rows produced")
        return 1

    results = Path("results")
    results.mkdir(parents=True, exist_ok=True)

    metric = "erp" if df["erp"].notna().any() else "seq_dev"
    por = price_of_robustness(df, metric=metric)
    por.to_parquet(results / "rq2_price_of_robustness.parquet", index=False)

    curve = (
        por.groupby("lam")[[f"d_{metric}", "d_reliability", "reliability_index",
                            metric, "expected_lateness", "runtime_s"]]
        .agg(["mean", "std", "count"])
    )
    curve.columns = ["_".join(c) for c in curve.columns]
    curve = curve.reset_index().sort_values("lam", ascending=False)
    curve.to_csv(results / "rq2_price_of_robustness_curve.csv", index=False)
    print(curve.to_string(index=False))

    df["method"] = "lam_" + df["lam"].round(3).astype(str)
    tests = pd.concat(
        [paired_compare(df, m) for m in (metric, "reliability_index",
                                         "expected_lateness")],
        ignore_index=True,
    )
    tests.to_csv(results / "rq2_wilcoxon.csv", index=False)

    trends = [lambda_trend_test(df, m) for m in
              ("reliability_index", "expected_lateness", metric)]
    (results / "rq2_summary.json").write_text(json.dumps(
        {"similarity_metric": metric, "folds": folds,
         "n_rows": int(len(df)), "trends": trends}, indent=2
    ))
    print(json.dumps(trends, indent=2))
    print(summarise(df, by=("lam",)).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

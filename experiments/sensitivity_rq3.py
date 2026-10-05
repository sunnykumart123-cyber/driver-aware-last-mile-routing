"""RQ3: sensitivity to the quality of the uncertainty estimates (§7, Phase 8).

    python -m experiments.sensitivity_rq3 --folds Austin --limit 20

Two axes are varied, both through the §4b degradation hooks, so the degraded
models are genuinely *fitted worse* rather than perturbed after the fact:

``full``            all covariates, all quantile levels, all training routes
``few_features``    the two-covariate travel/service heads
``coarse_grid``     three quantile levels instead of five
``small_sample``    a fraction of the training routes
``all_degraded``    every hook at once

and, orthogonally, the ambiguity radius ``delta`` on a grid, since the point of
the DRO layer is that a larger radius should buy back reliability when ``P_hat``
is poor.  Each (degradation, delta) cell is solved on the same routes at the
same lambda, and the reliability index is always measured against the *full*
model's independent scenario draw -- the ground truth must not degrade with the
optimizer's view of it, or the experiment measures nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from core.config import METROS, UncertaintyConfig
from core.io import metro_slug
from core.logging_utils import get_logger
from eval.metrics import EvalContext, evaluate_solution, summarise
from eval.significance_tests import paired_compare
from experiments.common import base_config, fold_instances, log_rows, make_runner, objective_for
from models.uncertainty.quantile_model import QuantileUncertaintyModel, fit_historical_stats

log = get_logger(__name__)

DEGRADATIONS = {
    "full": {},
    "few_features": {"degrade_features": True},
    "coarse_grid": {"degrade_quantiles": True},
    "small_sample": {"degrade_train_fraction": 0.25},
    "all_degraded": {
        "degrade_features": True,
        "degrade_quantiles": True,
        "degrade_train_fraction": 0.25,
    },
}


def _fit_degraded(train_instances, cfg, overrides: dict) -> QuantileUncertaintyModel:
    unc = UncertaintyConfig(**{**cfg.uncertainty.__dict__, **overrides})
    stats = fit_historical_stats(train_instances)
    return QuantileUncertaintyModel.fit(train_instances, unc, stats)


def run_fold(metro: str, args) -> pd.DataFrame:
    cfg = base_config(metro, tag=args.tag, eps=args.eps, scenarios=args.scenarios,
                      time_limit=args.time_limit, iterations=args.iterations)
    cfg.lam = args.lam
    runner = make_runner(cfg)
    if runner.uncertainty_model is None:
        log.error("Stage-1b artefact missing: run train_uncertainty.py first")
        return pd.DataFrame()

    train = fold_instances(args.processed, metro, "train", args.train_limit)
    insts = fold_instances(args.processed, metro, args.on, args.limit, args.max_stops)
    fold = f"ho_{metro_slug(metro)}"

    # The yardstick: the full model's independent draw, fixed across all arms.
    reference_ctx = EvalContext(cfg, runner.uncertainty_model)

    rows: list[dict] = []
    for name, overrides in DEGRADATIONS.items():
        if name not in args.arms:
            continue
        model = runner.uncertainty_model if name == "full" else _fit_degraded(
            train, cfg, overrides
        )
        degraded_runner = runner
        degraded_runner.uncertainty_model = model
        for delta in args.deltas:
            unc = UncertaintyConfig(**{**cfg.uncertainty.__dict__, "delta": delta})
            for inst in insts:
                obj = objective_for(degraded_runner, inst, cfg.lam,
                                    uncertainty_override=unc)
                sol = degraded_runner.solve(inst, obj, args.method)
                row = evaluate_solution(
                    inst, sol, obj, reference_ctx, f"rq3_{name}", fold,
                    {"degradation": name, "delta": delta},
                )
                rows.append(row)
        log.info("arm %s done (%d routes x %d deltas)", name, len(insts), len(args.deltas))

    runner.uncertainty_model = None  # avoid leaking a degraded model to a caller
    return log_rows(rows, cfg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--folds", nargs="*", default=["Austin"], choices=list(METROS))
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--on", default="test", choices=["test", "val", "train"])
    ap.add_argument("--arms", nargs="*", default=list(DEGRADATIONS),
                    choices=list(DEGRADATIONS))
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.0, 0.1, 0.5])
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--scenarios", type=int, default=64)
    ap.add_argument("--method", default="alns", choices=["alns", "auto"])
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--train-limit", type=int, default=400)
    ap.add_argument("--max-stops", type=int, default=None)
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--tag", default="sensitivity_rq3")
    args = ap.parse_args(argv)

    folds = list(METROS) if args.all_folds else args.folds
    frames = [run_fold(m, args) for m in folds]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if df.empty:
        log.error("no rows produced")
        return 1

    results = Path("results")
    curve = (
        df.groupby(["degradation", "delta"])[
            ["reliability_index", "expected_lateness", "total_eta", "objective"]
        ].agg(["mean", "std", "count"])
    )
    curve.columns = ["_".join(c) for c in curve.columns]
    curve = curve.reset_index()
    curve.to_csv(results / "rq3_degradation_curve.csv", index=False)
    print(curve.to_string(index=False))

    tests = paired_compare(df, "reliability_index")
    tests.to_csv(results / "rq3_wilcoxon.csv", index=False)
    (results / "rq3_summary.json").write_text(json.dumps(
        {"arms": args.arms, "deltas": args.deltas, "folds": folds,
         "n_rows": int(len(df))}, indent=2
    ))
    print(summarise(df, by=("method", "delta")).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

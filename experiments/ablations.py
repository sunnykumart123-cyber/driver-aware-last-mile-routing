"""Baselines + ablations + leave-one-metro-out (§7, §9 Phase 7).

    python -m experiments.ablations --folds Austin --limit 30
    python -m experiments.ablations --all-folds --limit 100 --time-limit 60

Arms produced per route:

``naive_tsp``               shortest-time TSP, zone-respecting variant too
``imitation_pointer``       lambda = 1, greedy pointer rollout
``imitation_gbm``           lambda = 1, greedy GBM rollout
``imitation_alns``          lambda = 1, same search as the full method
``robust_no_behavior``      lambda = 0, uniform behaviour cost
``mtp_full``                the thesis method at the configured lambda
``ablation_no_behavior``    full method, uniform behaviour cost
``ablation_no_robustness``  full method with the reliability term switched off
``ablation_geometry_box``   box ambiguity set
``ablation_geometry_ellip`` ellipsoidal ambiguity set

Everything lands in ``results/experiment_log.parquet`` with ``fold`` set to the
held-out metro, so the five folds aggregate to the mean +/- spread §7 wants.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from core.config import METROS, UncertaintyConfig
from core.logging_utils import get_logger
from baselines.imitation_baseline import gbm_greedy, pointer_greedy, alns_imitation
from baselines.naive_tsp import solve_naive_tsp, zone_respecting_naive_tsp
from baselines.robust_no_behavior import solve_robust_no_behavior
from experiments.common import (
    base_config,
    fold_instances,
    log_rows,
    make_runner,
    objective_for,
)
from models.behavior.gbm_transition import GBMTransitionModel
from models.behavior.pointer_net import PointerNet
from core.io import metro_slug

log = get_logger(__name__)


def run_fold(metro: str, args) -> pd.DataFrame:
    cfg = base_config(
        metro, tag=args.tag, behavior_model="pointer", eps=args.eps,
        delta=args.delta, scenarios=args.scenarios,
        time_limit=args.time_limit, iterations=args.iterations,
    )
    cfg.lam = args.lam
    runner = make_runner(cfg)
    insts = fold_instances(args.processed, metro, args.on, args.limit, args.max_stops)
    if not insts:
        log.error("fold %s has no instances", metro)
        return pd.DataFrame()
    fold = f"ho_{metro_slug(metro)}"
    art = Path(cfg.paths.artifacts)
    tag = f"ho_{metro_slug(metro)}"

    gbm = None
    gbm_path = art / f"gbm_transition_{tag}.pkl"
    if gbm_path.exists():
        gbm = GBMTransitionModel.load(gbm_path)
    pointer = runner.behavior_model if isinstance(runner.behavior_model, PointerNet) else None

    rows: list[dict] = []
    for k, inst in enumerate(insts):
        obj_full = objective_for(runner, inst, cfg.lam)
        obj_lam1 = objective_for(runner, inst, 1.0)
        obj_lam0 = objective_for(runner, inst, 0.0)

        # ---- baselines -------------------------------------------------
        sol = solve_naive_tsp(inst)
        rows.append(_row(runner, inst, sol, obj_full, "naive_tsp", fold))
        sol = zone_respecting_naive_tsp(inst)
        rows.append(_row(runner, inst, sol, obj_full, "naive_tsp_zoned", fold))

        if pointer is not None:
            sol = pointer_greedy(inst, pointer)
            rows.append(_row(runner, inst, sol, obj_lam1, "imitation_pointer", fold))
        if gbm is not None:
            sol = gbm_greedy(inst, gbm)
            rows.append(_row(runner, inst, sol, obj_lam1, "imitation_gbm", fold))

        sol = alns_imitation(inst, obj_lam1, cfg.solver)
        rows.append(_row(runner, inst, sol, obj_lam1, "imitation_alns", fold))

        sol, obj_rb = solve_robust_no_behavior(inst, obj_lam0.evaluator, cfg.solver)
        rows.append(_row(runner, inst, sol, obj_rb, "robust_no_behavior", fold))

        # ---- the method ------------------------------------------------
        sol = runner.solve(inst, obj_full, args.method)
        rows.append(_row(runner, inst, sol, obj_full, "mtp_full", fold))

        # ---- ablations -------------------------------------------------
        obj_nb = objective_for(runner, inst, cfg.lam, behavior="uniform")
        sol = runner.solve(inst, obj_nb, args.method)
        rows.append(_row(runner, inst, sol, obj_nb, "ablation_no_behavior", fold))

        sol = runner.solve(inst, obj_lam1, args.method)
        rows.append(_row(runner, inst, sol, obj_lam1, "ablation_no_robustness", fold))

        for geom, name in (("box", "ablation_geometry_box"),
                           ("ellipsoidal", "ablation_geometry_ellip")):
            unc = UncertaintyConfig(**{**cfg.uncertainty.__dict__, "geometry": geom})
            obj_g = objective_for(runner, inst, cfg.lam, uncertainty_override=unc)
            sol = runner.solve(inst, obj_g, args.method)
            rows.append(_row(runner, inst, sol, obj_g, name, fold,
                             {"geometry": geom}))

        if (k + 1) % 10 == 0:
            log.info("fold %s: %d/%d routes", metro, k + 1, len(insts))

    rows = [r for r in rows if isinstance(r, dict)]
    return log_rows(rows, cfg)


def _row(runner, inst, sol, obj, method: str, fold: str, extra: dict | None = None):
    from eval.metrics import evaluate_solution

    return evaluate_solution(inst, sol, obj, runner.ctx, method, fold, extra)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--folds", nargs="*", default=["Austin"], choices=list(METROS))
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--on", default="test", choices=["test", "val", "train"])
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--scenarios", type=int, default=64)
    ap.add_argument("--method", default="alns",
                    choices=["alns", "auto", "exact", "benders"])
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--max-stops", type=int, default=None)
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--tag", default="ablations")
    args = ap.parse_args(argv)

    folds = list(METROS) if args.all_folds else args.folds
    frames = []
    for m in folds:
        log.info("=== ablation fold: held out %s ===", m)
        frames.append(run_fold(m, args))
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if df.empty:
        log.error("no rows produced")
        return 1

    from eval.metrics import summarise

    view = summarise(df, by=("method",))
    print(view.to_string(index=False))
    print(json.dumps({"rows": len(df), "folds": folds}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

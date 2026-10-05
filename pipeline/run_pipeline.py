"""Pipeline glue (§5d): Stage-1 artefacts -> costs + ambiguity set -> solver -> rows.

    python -m pipeline.run_pipeline --held-out-metro Austin --lam 0.5 --limit 50
    python -m pipeline.run_pipeline --method exact --limit 5 --max-stops 10

One route is one unit of work.  The solver is chosen by instance size unless one
is named explicitly, following §5: exact for toy sizes, Benders for the
medium band where a certified gap is wanted, ALNS everywhere else.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from core.config import METROS, Config, Paths
from core.io import append_results
from core.logging_utils import get_logger
from core.objective import CombinedObjective
from core.types import Instance, Solution
from data.splits import held_out_metro_split, model_split
from eval.metrics import EvalContext, evaluate_solution
from models.behavior.cost import build_behavior_cost, load_behavior_model
from models.uncertainty.ambiguity_set import AmbiguitySet
from models.uncertainty.quantile_model import QuantileUncertaintyModel
from solvers.alns.alns_solver import solve_alns, zone_nearest_neighbour
from solvers.benders.benders_solver import solve_benders
from solvers.exact_milp import solve_bruteforce, solve_exact
from solvers.reliability import ReliabilityEvaluator

log = get_logger(__name__)

EXACT_MAX_STOPS = 12
BENDERS_MAX_STOPS = 80


@dataclass
class PipelineRunner:
    """Holds the Stage-1 artefacts and turns instances into result rows."""

    cfg: Config
    behavior_model: object
    uncertainty_model: QuantileUncertaintyModel | None
    ctx: EvalContext

    @classmethod
    def from_artifacts(cls, cfg: Config, tag: str | None = None) -> "PipelineRunner":
        art = Path(cfg.paths.artifacts)
        tag = tag if tag is not None else _fold_tag(cfg.held_out_metro)
        behavior = load_behavior_model(cfg.behavior, art, tag)
        unc_path = art / f"quantile_model_{tag}.pkl" if tag else art / "quantile_model.pkl"
        unc = QuantileUncertaintyModel.load(unc_path) if unc_path.exists() else None
        if unc is None:
            log.warning(
                "no Stage-1b artefact at %s: falling back to the planned times "
                "as a degenerate one-scenario P_hat", unc_path,
            )
        return cls(cfg, behavior, unc, EvalContext(cfg, unc))

    # ------------------------------------------------------------------ per route
    def build_objective(self, inst: Instance, lam: float | None = None) -> CombinedObjective:
        cost = build_behavior_cost(self.behavior_model, inst)
        if self.uncertainty_model is None:
            amb = AmbiguitySet.deterministic(inst, self.cfg.uncertainty)
        else:
            amb = AmbiguitySet.from_model(
                inst, self.uncertainty_model, self.cfg.uncertainty,
                np.random.default_rng(self.cfg.uncertainty.seed),
            )
        return CombinedObjective(
            inst=inst,
            behavior_cost=cost,
            evaluator=ReliabilityEvaluator(inst, amb),
            lam=self.cfg.lam if lam is None else lam,
        )

    def choose_solver(self, inst: Instance, requested: str) -> str:
        if requested != "auto":
            return requested
        if inst.n <= EXACT_MAX_STOPS:
            return "exact"
        if inst.n <= BENDERS_MAX_STOPS:
            return "benders"
        return "alns"

    def solve(self, inst: Instance, obj: CombinedObjective,
              method: str = "auto") -> Solution:
        which = self.choose_solver(inst, method)
        cfg = self.cfg.solver
        if which == "exact":
            return solve_exact(inst, obj, cfg).solution
        if which == "bruteforce":
            return solve_bruteforce(inst, obj, cfg).solution
        if which == "benders":
            warm = solve_alns(inst, obj, cfg).solution
            res = solve_benders(inst, obj, cfg, initial=warm.order)
            return res.solution if res.solution.objective <= warm.objective else warm
        if which == "alns":
            return solve_alns(inst, obj, cfg).solution
        if which == "greedy":
            order = zone_nearest_neighbour(inst, obj, cfg.zone_precedence)
            return obj.to_solution(order, "greedy", 0.0)
        raise ValueError(f"unknown solver {which!r}")

    def run_route(self, inst: Instance, method: str = "auto",
                  lam: float | None = None, fold: str = "",
                  label: str | None = None) -> dict:
        obj = self.build_objective(inst, lam)
        t0 = time.perf_counter()
        sol = self.solve(inst, obj, method)
        wall = time.perf_counter() - t0
        name = label or f"mtp_{self.cfg.behavior.model}"
        row = evaluate_solution(inst, sol, obj, self.ctx, name, fold)
        row["wall_s"] = wall
        row["solver_requested"] = method
        return row

    # --------------------------------------------------------------------- batch
    def run_split(
        self,
        instances: Iterable[Instance],
        method: str = "auto",
        lams: Sequence[float] | None = None,
        fold: str = "",
        label: str | None = None,
        max_stops: int | None = None,
        log_path: str | Path | None = None,
    ) -> pd.DataFrame:
        lams = list(lams) if lams is not None else [self.cfg.lam]
        rows: list[dict] = []
        for k, inst in enumerate(instances):
            if max_stops is not None and inst.n > max_stops:
                continue
            for lam in lams:
                try:
                    rows.append(self.run_route(inst, method, lam, fold, label))
                except Exception as exc:  # one bad route must not kill a sweep
                    log.exception("route %s failed: %s", inst.route_id, exc)
                    rows.append({
                        "method": label or "mtp", "lam": lam, "metro": inst.metro,
                        "fold": fold, "route_id": inst.route_id, "n_stops": inst.n,
                        "error": repr(exc),
                    })
            if (k + 1) % 20 == 0:
                log.info("processed %d routes", k + 1)
        df = pd.DataFrame(rows)
        if log_path is not None and not df.empty:
            append_results(df.to_dict("records"), log_path)
        return df


def _fold_tag(held_out_metro: str | None) -> str:
    from core.io import metro_slug

    return f"ho_{metro_slug(held_out_metro)}" if held_out_metro else ""


# ------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--held-out-metro", default="Austin", choices=list(METROS))
    ap.add_argument("--behavior-model", default="pointer",
                    choices=["pointer", "gbm", "uniform"])
    ap.add_argument("--method", default="auto",
                    choices=["auto", "alns", "exact", "benders", "bruteforce", "greedy"])
    ap.add_argument("--lam", type=float, nargs="+", default=[0.5])
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--geometry", default="wasserstein",
                    choices=["wasserstein", "box", "ellipsoidal"])
    ap.add_argument("--scenarios", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-stops", type=int, default=None)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--iterations", type=int, default=4000)
    ap.add_argument("--on", default="test", choices=["test", "val", "train"])
    ap.add_argument("--tag", default="pipeline")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args(argv)

    cfg = Config()
    cfg.held_out_metro = args.held_out_metro
    cfg.behavior.model = args.behavior_model
    cfg.uncertainty.eps = args.eps
    cfg.uncertainty.delta = args.delta
    cfg.uncertainty.geometry = args.geometry
    cfg.uncertainty.n_scenarios = args.scenarios
    cfg.solver.time_limit_s = args.time_limit
    cfg.solver.alns_iterations = args.iterations
    cfg.tag = args.tag
    cfg.paths = Paths().ensure()

    train, val = model_split(args.processed, args.held_out_metro)
    test = held_out_metro_split(args.processed, args.held_out_metro)
    split = {"train": train, "val": val, "test": test}[args.on]
    if len(split) == 0:
        log.error("split %s is empty: run `python -m data.preprocess` first", split.name)
        return 1

    runner = PipelineRunner.from_artifacts(cfg)
    df = runner.run_split(
        split.iter_instances(args.limit),
        method=args.method,
        lams=args.lam,
        fold=f"ho_{args.held_out_metro}",
        max_stops=args.max_stops,
        log_path=None if args.no_log else cfg.paths.experiment_log,
    )
    if df.empty:
        log.error("no rows produced")
        return 1
    cols = [c for c in ("route_id", "lam", "objective", "reliability_index",
                        "expected_lateness", "seq_dev", "runtime_s") if c in df]
    print(df[cols].to_string(index=False))
    print(json.dumps(
        {"routes": int(df["route_id"].nunique()),
         "mean_objective": float(df["objective"].mean()),
         "mean_reliability": float(df["reliability_index"].mean())},
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())

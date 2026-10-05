"""Phase-2 and Phase-3 gate reports (§9).

    python -m scripts.phase_gates --phase 2 --limit 300
    python -m scripts.phase_gates --phase 3 --limit 300 \
        --reference 0.0479 --reference-source "da Costa et al. (2023), Table 4"

**Phase 2** — the naive shortest-time TSP must produce an ERP of a plausible
*order of magnitude* on the official scorer.  There is no published number to
match here; the gate is a sanity check that the pipeline, the parser and the
scorer are wired together correctly.  The report prints the distribution and
flags the degenerate outcomes that would indicate a wiring bug (all zeros, all
identical, NaNs).

**Phase 3** — the GBM transition model must reproduce a published ERP within a
defensible tolerance.  This script will not invent the reference: pass
``--reference`` and ``--reference-source`` from the paper you are reproducing,
and the report states the gap and whether it falls inside ``--tolerance``.  The
tolerance argument you write in the thesis is yours to defend; the script only
does the arithmetic and records what was compared.

Both gates write a JSON report to ``results/`` so the claim is auditable later,
and append their per-route rows to the experiment log like any other method.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import METROS, Config, Paths, SolverConfig
from core.io import append_results, metro_slug
from core.logging_utils import get_logger
from data.splits import held_out_metro_split, model_split
from eval.erp_official import erp_score, scorer_available, sequence_deviation
from baselines.naive_tsp import solve_naive_tsp, zone_respecting_naive_tsp

log = get_logger(__name__)


def _instances(processed: Path, metro: str, on: str, limit: int | None):
    train, val = model_split(processed, metro)
    test = held_out_metro_split(processed, metro)
    split = {"train": train, "val": val, "test": test}[on]
    return split.instances(limit)


def _score_rows(instances, solve, method: str, fold: str) -> list[dict]:
    rows = []
    for inst in instances:
        if inst.actual_sequence is None:
            continue
        sol = solve(inst)
        row = {
            "method": method, "fold": fold, "route_id": inst.route_id,
            "metro": inst.metro, "n_stops": inst.n,
            "n_zones": len(inst.zone_groups()),
            "seq_dev": sequence_deviation(inst, sol.order),
            "runtime_s": sol.runtime_s,
            "erp": float("nan"),
        }
        if scorer_available():
            try:
                row["erp"] = erp_score(inst, sol.order)
            except Exception as exc:  # pragma: no cover
                log.warning("ERP failed on %s: %s", inst.route_id, exc)
        rows.append(row)
    return rows


def _describe(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "min": float(v.min()),
        "p25": float(np.percentile(v, 25)),
        "median": float(np.median(v)),
        "p75": float(np.percentile(v, 75)),
        "max": float(v.max()),
        "n_zero": int((v == 0).sum()),
        "frac_zero": float((v == 0).mean()),
    }


def _degeneracy_warnings(stats: dict) -> list[str]:
    out = []
    if stats.get("n", 0) == 0:
        out.append("no finite scores at all: the scorer or the parser is broken")
        return out
    if stats["frac_zero"] > 0.5:
        out.append(
            f"{stats['frac_zero']:.0%} of routes score exactly 0 -- suspicious "
            "unless the solver is reproducing sigma* (remember the metric is "
            "reversal-invariant, see docs/DATA_NOTES.md §7)"
        )
    if stats.get("std", 0.0) < 1e-12 and stats["n"] > 1:
        out.append("every route scores identically: the submission is probably "
                   "not varying with the instance")
    if stats["max"] > 10:
        out.append(f"max score {stats['max']:.3g} is implausibly large for the "
                   "challenge metric: check the cost matrix units")
    return out


def phase2(args, cfg: Config) -> dict:
    insts = _instances(args.processed, args.metro, args.on, args.limit)
    if not insts:
        raise SystemExit("no instances: run data.preprocess first")
    fold = f"ho_{metro_slug(args.metro)}"

    rows = _score_rows(insts, solve_naive_tsp, "gate2_naive_tsp", fold)
    rows += _score_rows(insts, zone_respecting_naive_tsp,
                        "gate2_naive_tsp_zoned", fold)
    df = pd.DataFrame(rows)
    append_results(df.to_dict("records"), cfg.paths.experiment_log)

    report: dict = {
        "phase": 2,
        "metro_held_out": args.metro,
        "split": args.on,
        "n_routes": int(df["route_id"].nunique()),
        "erp_available": scorer_available(),
        "arms": {},
    }
    for method, g in df.groupby("method"):
        stats = _describe(g["erp"].to_numpy())
        report["arms"][method] = {
            "erp": stats,
            "seq_dev": _describe(g["seq_dev"].to_numpy()),
            "warnings": _degeneracy_warnings(stats),
        }
    report["verdict"] = (
        "inspect" if any(a["warnings"] for a in report["arms"].values())
        else "plausible"
    )
    return report


def phase3(args, cfg: Config) -> dict:
    from models.behavior.gbm_transition import GBMTransitionModel

    tag = f"ho_{metro_slug(args.metro)}"
    path = Path(cfg.paths.artifacts) / f"gbm_transition_{tag}.pkl"
    if not path.exists():
        raise SystemExit(
            f"missing {path}: run "
            f"`python train_behavior.py --held-out-metro \"{args.metro}\" --model gbm`"
        )
    model = GBMTransitionModel.load(path)
    insts = _instances(args.processed, args.metro, args.on, args.limit)
    if not insts:
        raise SystemExit("no instances: run data.preprocess first")
    fold = tag

    rows = _score_rows(
        insts, lambda inst: _as_solution(model.greedy_order(inst)),
        "gate3_gbm_greedy", fold,
    )
    df = pd.DataFrame(rows)
    append_results(df.to_dict("records"), cfg.paths.experiment_log)

    stats = _describe(df["erp"].to_numpy())
    report: dict = {
        "phase": 3,
        "metro_held_out": args.metro,
        "split": args.on,
        "n_routes": int(len(df)),
        "erp_available": scorer_available(),
        "erp": stats,
        "seq_dev": _describe(df["seq_dev"].to_numpy()),
        "warnings": _degeneracy_warnings(stats),
        "reference": args.reference,
        "reference_source": args.reference_source,
        "tolerance": args.tolerance,
    }
    if args.reference is not None and stats.get("n"):
        gap = stats["mean"] - args.reference
        report["absolute_gap"] = gap
        report["relative_gap"] = gap / args.reference if args.reference else None
        report["within_tolerance"] = bool(abs(gap) <= args.tolerance)
        report["verdict"] = "GATE PASSED" if report["within_tolerance"] else "GATE OPEN"
    else:
        report["verdict"] = (
            "no reference supplied: pass --reference and --reference-source from "
            "the paper being reproduced. The gate cannot be closed without it."
        )
    return report


def _as_solution(order):
    from core.types import Solution

    return Solution(order=order, solver="gbm_greedy", runtime_s=float("nan"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", type=int, choices=[2, 3], required=True)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--metro", default="Austin", choices=list(METROS),
                    help="the held-out metro defining the fold")
    ap.add_argument("--on", default="val", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--reference", type=float, default=None,
                    help="published ERP to reproduce (phase 3)")
    ap.add_argument("--reference-source", default=None,
                    help="citation for --reference, recorded in the report")
    ap.add_argument("--tolerance", type=float, default=0.005)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = Config()
    cfg.paths = Paths().ensure()
    cfg.solver = SolverConfig()

    report = phase2(args, cfg) if args.phase == 2 else phase3(args, cfg)
    out = args.out or Path(cfg.paths.results) / f"phase{args.phase}_gate.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    log.info("report written to %s", out)

    if not scorer_available():
        log.error(
            "the official scorer is not installed, so no ERP was computed and "
            "neither gate can be closed. See eval/erp_official.py."
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

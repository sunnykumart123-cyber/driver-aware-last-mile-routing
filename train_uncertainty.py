"""Train Stage-1b (§4b) and calibrate the Wasserstein radius.

    python train_uncertainty.py --held-out-metro Austin
    python train_uncertainty.py --all-folds
    python train_uncertainty.py --held-out-metro Austin --degrade features

(§8 of the build spec lists ``train_behavior.py`` but no matching script for
Stage-1b; keeping the two stages in separate entry points is a deliberate
deviation, since §1 requires the two learned models to be trained and evaluated
independently.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.config import METROS, Config, Paths, UncertaintyConfig
from core.io import metro_slug
from core.logging_utils import get_logger
from data.splits import model_split
from models.uncertainty.ambiguity_set import (
    calibrate_delta,
    calibrate_delta_from_coverage,
)
from models.uncertainty.quantile_model import (
    QuantileUncertaintyModel,
    fit_historical_stats,
)

log = get_logger(__name__)


def run_fold(metro: str, args, cfg: Config) -> dict:
    tag = f"ho_{metro_slug(metro)}"
    train, val = model_split(args.processed, metro,
                             high_quality_only=not args.all_quality)
    train_i = train.instances(args.limit)
    val_i = val.instances(args.limit_val)
    if not train_i:
        raise SystemExit("no training routes: run `python -m data.preprocess` first")

    stats = fit_historical_stats(train_i)
    model = QuantileUncertaintyModel.fit(train_i, cfg.uncertainty, stats)

    out = Path(cfg.paths.artifacts)
    out.mkdir(parents=True, exist_ok=True)
    path = model.save(out / f"quantile_model_{tag}.pkl")
    (out / f"historical_stats_{tag}.json").write_text(stats.to_json())

    summary = {
        "held_out_metro": metro,
        "n_train": len(train_i),
        "n_val": len(val_i),
        "quantiles": list(model.quantiles),
        "artifact": str(path),
    }

    if args.calibrate_delta and val_i:
        # Primary: calibrate the radius against the measured out-of-sample
        # under-coverage of the Stage-1b quantiles.
        delta, coverage_report = calibrate_delta_from_coverage(
            val_i, model, cfg.uncertainty
        )
        summary["delta"] = delta
        summary["delta_calibration"] = coverage_report
        # Secondary, kept as a diagnostic only: Monte-Carlo stability.
        subset = val_i[: args.calibrate_routes]
        stability_delta, records = calibrate_delta(subset, model, cfg.uncertainty)
        summary["scenario_stability_delta"] = stability_delta
        (out / f"delta_calibration_{tag}.json").write_text(
            json.dumps(
                {"delta": delta, "coverage_calibration": coverage_report,
                 "scenario_stability_delta": stability_delta, "grid": records},
                indent=2,
            )
        )
    (out / f"train_uncertainty_summary_{tag}.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--held-out-metro", default="Austin", choices=list(METROS))
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--scenarios", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--limit-val", type=int, default=None)
    ap.add_argument("--all-quality", action="store_true")
    ap.add_argument("--calibrate-delta", action="store_true")
    ap.add_argument("--calibrate-routes", type=int, default=50)
    ap.add_argument("--degrade", nargs="*", default=[],
                    choices=["features", "quantiles", "fraction"],
                    help="RQ3 degradation hooks (§4b)")
    ap.add_argument("--degrade-fraction", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=20260101)
    args = ap.parse_args(argv)

    cfg = Config()
    cfg.paths = Paths().ensure()
    cfg.uncertainty = UncertaintyConfig(
        eps=args.eps,
        n_scenarios=args.scenarios,
        seed=args.seed,
        degrade_features="features" in args.degrade,
        degrade_quantiles="quantiles" in args.degrade,
        degrade_train_fraction=args.degrade_fraction if "fraction" in args.degrade else 1.0,
    )

    metros = list(METROS) if args.all_folds else [args.held_out_metro]
    for m in metros:
        log.info("=== Stage-1b fold: held out %s ===", m)
        print(json.dumps(run_fold(m, args, cfg), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

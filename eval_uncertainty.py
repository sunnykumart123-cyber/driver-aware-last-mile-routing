"""Evaluate Stage-1b in isolation (§4b, §9 Phase 4).

    python eval_uncertainty.py --held-out-metro Austin
    python eval_uncertainty.py --held-out-metro Austin --plot

Required diagnostics, reported per metro and per split:

* **pinball (quantile) loss** at every fitted level -- the proper scoring rule
  for quantile predictions;
* **reliability diagram** data: nominal level ``q`` against the empirical
  coverage ``P(target <= q-th predicted quantile)``.  A well-calibrated model
  sits on the diagonal; deviations are what the DRO radius has to absorb, which
  is exactly why this is reported before any radius is calibrated.

Calibration is measured on the *unshifted* conditional quantiles, i.e. the
model's own predictive distribution, not the route-anchored version the
optimizer consumes -- otherwise the anchoring would flatter the median.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import METROS, Config, Paths
from core.io import metro_slug
from core.logging_utils import get_logger
from data.splits import held_out_metro_split, model_split
from models.uncertainty.quantile_model import (
    QuantileUncertaintyModel,
    build_training_tables,
)

log = get_logger(__name__)


def pinball_loss(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    d = y - pred
    return float(np.mean(np.maximum(q * d, (q - 1.0) * d)))


def calibration_table(model: QuantileUncertaintyModel, rows: pd.DataFrame,
                      head: str, split: str) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    pred = (model.predict_travel(rows) if head == "travel"
            else model.predict_service(rows))
    y = rows["target"].to_numpy(dtype=float)
    out = []
    for k, q in enumerate(model.quantiles):
        out.append({
            "head": head,
            "split": split,
            "quantile": q,
            "empirical_coverage": float(np.mean(y <= pred[:, k])),
            "pinball_loss": pinball_loss(y, pred[:, k], q),
            "mean_prediction": float(pred[:, k].mean()),
            "mean_target": float(y.mean()),
            "n": int(len(y)),
        })
    df = pd.DataFrame(out)
    df["coverage_error"] = df["empirical_coverage"] - df["quantile"]
    return df


def interval_widths(model: QuantileUncertaintyModel, rows: pd.DataFrame,
                    head: str) -> dict:
    """Sharpness: how wide the predicted intervals are, in minutes."""
    if rows.empty:
        return {}
    pred = (model.predict_travel(rows) if head == "travel"
            else model.predict_service(rows))
    qs = list(model.quantiles)
    return {
        "median_iqr_min": float(np.median(pred[:, -2] - pred[:, 1]))
        if len(qs) >= 4 else float("nan"),
        "median_full_range_min": float(np.median(pred[:, -1] - pred[:, 0])),
        "median_center_min": float(np.median(pred[:, len(qs) // 2])),
    }


def plot_reliability(df: pd.DataFrame, path: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed: skipping the reliability plot")
        return None

    heads = sorted(df["head"].unique())
    fig, axes = plt.subplots(1, len(heads), figsize=(5 * len(heads), 4.2),
                             squeeze=False)
    for ax, head in zip(axes[0], heads):
        sub = df[df["head"] == head]
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        for split, g in sub.groupby("split"):
            g = g.sort_values("quantile")
            ax.plot(g["quantile"], g["empirical_coverage"], "o-", label=split)
        ax.set_title(f"{head} time")
        ax.set_xlabel("nominal quantile")
        ax.set_ylabel("empirical coverage")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--held-out-metro", default="Austin", choices=list(METROS))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--artifact", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = Config()
    cfg.paths = Paths().ensure()
    tag = f"ho_{metro_slug(args.held_out_metro)}"
    path = args.artifact or Path(cfg.paths.artifacts) / f"quantile_model_{tag}.pkl"
    if not path.exists():
        log.error("missing %s: run train_uncertainty.py first", path)
        return 1
    model = QuantileUncertaintyModel.load(path)

    train, val = model_split(args.processed, args.held_out_metro)
    test = held_out_metro_split(args.processed, args.held_out_metro)
    splits = {
        "train": train.instances(args.limit),
        "validation": val.instances(args.limit),
        "held_out_metro": test.instances(args.limit),
    }

    frames, sharp = [], {}
    for name, insts in splits.items():
        if not insts:
            log.warning("split %s is empty", name)
            continue
        tr, sv = build_training_tables(insts, model.stats)
        frames.append(calibration_table(model, tr, "travel", name))
        frames.append(calibration_table(model, sv, "service", name))
        sharp[name] = {
            "travel": interval_widths(model, tr, "travel"),
            "service": interval_widths(model, sv, "service"),
        }
        # Per-metro breakdown inside the multi-metro training split.
        for metro, g in tr.groupby("metro"):
            t = calibration_table(model, g, "travel", f"{name}:{metro}")
            frames.append(t)

    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    out_pq = Path(cfg.paths.results) / f"uncertainty_calibration_{tag}.parquet"
    df.to_parquet(out_pq, index=False)

    main_view = df[~df["split"].str.contains(":")]
    print(main_view.sort_values(["head", "split", "quantile"]).to_string(index=False))

    worst = main_view.loc[main_view["coverage_error"].abs().idxmax()]
    payload = {
        "held_out_metro": args.held_out_metro,
        "artifact": str(path),
        "quantiles": list(model.quantiles),
        "sharpness": sharp,
        "max_abs_coverage_error": float(abs(worst["coverage_error"])),
        "worst_cell": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                       for k, v in worst.to_dict().items()},
        "mean_pinball_by_head": main_view.groupby("head")["pinball_loss"]
        .mean().to_dict(),
    }
    (Path(cfg.paths.results) / f"uncertainty_calibration_{tag}.json").write_text(
        json.dumps(payload, indent=2)
    )
    print(json.dumps(payload["mean_pinball_by_head"], indent=2))

    if args.plot:
        p = plot_reliability(
            main_view, Path(cfg.paths.results) / f"reliability_diagram_{tag}.png"
        )
        if p:
            log.info("wrote %s", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

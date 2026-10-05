"""Descriptive diagnostics on the processed real data.

    python -m scripts.data_diagnostics --limit 2000
    python -m scripts.data_diagnostics --out results/data_diagnostics.json

Reports, per metro and overall, the facts the thesis has to state about the
dataset before any model is discussed:

* route size, zone count, package counts and volumes;
* **time-window density** -- how many stops actually carry a finite window,
  which is what the reliability term operates on and what drives solver cost;
* **zone contiguity of sigma\\*** -- the fraction of real driver sequences that
  visit each sort zone in one unbroken block.  This one is load-bearing: the
  zone-precedence constraint of §3 is only defensible to the extent that real
  drivers respect it, and where they do not, imposing it *excludes the driver's
  own sequence from the feasible set* and puts a floor on the achievable ERP;
* travel-matrix sanity (asymmetry, zero entries, magnitude);
* route-quality label distribution.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from core.io import iter_instances
from core.logging_utils import get_logger
from core.types import zone_contiguous

log = get_logger(__name__)


def route_row(inst) -> dict:
    tt = inst.travel_time
    off = ~np.eye(inst.n, dtype=bool)
    groups = inst.zone_groups()
    seq = inst.actual_sequence
    finite = np.isfinite(inst.tw_end)

    row = {
        "route_id": inst.route_id,
        "metro": inst.metro,
        "station": inst.station,
        "quality": inst.quality,
        "n_stops": inst.n,
        "n_zones": len(groups),
        "zone_size_mean": float(np.mean([len(v) for v in groups.values()])) if groups else 0.0,
        "n_windows": int(finite.sum()),
        "window_frac": float(finite.sum()) / max(inst.n, 1),
        "pkg_total": float(inst.pkg_count.sum()),
        "vol_total_l": float(inst.pkg_volume.sum()),
        "service_total_min": float(inst.service_time.sum()),
        "tt_median_min": float(np.median(tt[off])),
        "tt_max_min": float(tt.max()),
        "tt_asymmetry": float(np.abs(tt - tt.T)[off].mean()),
        "tt_zero_offdiag": int((tt[off] == 0).sum()),
        "capacity_l": None if not np.isfinite(inst.executor_capacity) else float(inst.executor_capacity),
        "departure_minute": inst.meta.get("departure_minute_of_day"),
        "weekday": inst.meta.get("weekday"),
        "has_sigma_star": seq is not None,
    }
    if seq is not None:
        zones_seq = [inst.zones[k] for k in seq[1:]]
        blocks = 1 + sum(1 for a, b in zip(zones_seq, zones_seq[1:]) if a != b)
        row.update({
            "sigma_zone_contiguous": bool(zone_contiguous(seq, inst.zones)),
            "zone_blocks": blocks,
            "extra_zone_blocks": blocks - len(groups),
        })
    return row


def summarise(df: pd.DataFrame) -> dict:
    def stats(col):
        v = pd.to_numeric(df[col], errors="coerce").dropna()
        if v.empty:
            return {}
        return {"mean": float(v.mean()), "median": float(v.median()),
                "min": float(v.min()), "max": float(v.max()),
                "p90": float(v.quantile(0.9))}

    out = {
        "n_routes": int(len(df)),
        "metros": df["metro"].value_counts().to_dict(),
        "quality": df["quality"].value_counts(dropna=False).to_dict(),
        "n_stops": stats("n_stops"),
        "n_zones": stats("n_zones"),
        "n_windows": stats("n_windows"),
        "window_frac": stats("window_frac"),
        "service_total_min": stats("service_total_min"),
        "tt_median_min": stats("tt_median_min"),
        "tt_asymmetry": stats("tt_asymmetry"),
    }
    sub = df[df["has_sigma_star"]]
    if not sub.empty and "sigma_zone_contiguous" in sub:
        contig = sub["sigma_zone_contiguous"].astype(bool)
        out["zone_precedence"] = {
            "n_with_sigma_star": int(len(sub)),
            "sigma_contiguous_count": int(contig.sum()),
            "sigma_contiguous_frac": float(contig.mean()),
            "extra_zone_blocks": stats("extra_zone_blocks"),
            "interpretation": (
                "The zone-precedence constraint of §3 forbids re-entering a "
                "sort zone. Where sigma* is not contiguous, that constraint "
                "excludes the driver's own sequence from the feasible set, so "
                "the attainable ERP has a floor above zero by construction."
            ),
        }
        by_metro = sub.groupby("metro")["sigma_zone_contiguous"].agg(["mean", "count"])
        out["zone_precedence"]["by_metro"] = {
            m: {"contiguous_frac": float(r["mean"]), "n": int(r["count"])}
            for m, r in by_metro.iterrows()
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("results/data_diagnostics.json"))
    ap.add_argument("--rows-out", type=Path,
                    default=Path("results/data_diagnostics_routes.parquet"))
    args = ap.parse_args(argv)

    rows = []
    for k, inst in enumerate(iter_instances(args.processed, limit=args.limit)):
        rows.append(route_row(inst))
        if (k + 1) % 500 == 0:
            log.info("scanned %d routes", k + 1)
    if not rows:
        log.error("no processed routes under %s", args.processed)
        return 1

    df = pd.DataFrame(rows)
    args.rows_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.rows_out, index=False)

    report = summarise(df)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    log.info("wrote %s and %s", args.out, args.rows_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

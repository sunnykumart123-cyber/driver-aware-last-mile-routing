"""Engineering timing probe for the ALNS -- **not an experiment, not data**.

    python -m scripts.benchmark_alns
    python -m scripts.benchmark_alns --sizes 60 150 240 --scenarios 64 --iterations 1000

This script generates arbitrary point sets purely to measure *how long one ALNS
iteration takes* as a function of route size, zone count, window density and
scenario count.  Nothing it produces is a result:

* no objective value, ERP, or reliability number from here may appear in the
  thesis or in ``results/experiment_log.parquet``;
* the §1 "real, open, public data only" constraint is about the data that
  produces *findings*.  A stopwatch reading on a synthetic point set is a
  property of the implementation, like a unit test's runtime.

It exists because the per-iteration cost of this solver is dominated by the
reliability term, which scales with the number of *window-carrying* stops rather
than with the number of stops, and that is worth being able to re-measure after
any change to ``solvers/alns/state.py`` or ``solvers/reliability.py``.

Reference numbers on the development machine (CPU-only laptop, Python 3.13,
numpy 2.4, S = 64, delta = 0.1, lambda = 0.5).  The real data has about 6% of
stops carrying a time window, so that is the operative row.

Zone precedence ON (the §3 constraint, now an ablation):

    stops  windows   ms/iteration
      150        8           13.0
      240       14           49.0

Zone precedence OFF (the default), candidate_cap = 24:

    stops  windows   ms/iteration   final objective vs. zone-constrained
      150        8           25.4   1.02  vs  9.03
      240       14          102.8   1.02  vs 21.17

Zone precedence OFF with candidate_cap = 0 (every position scored exactly) costs
510 ms/iteration at 240 stops for the same objective (1.03), so the shortlist
buys a 5x speed-up for nothing.

Two historical regressions worth guarding: before the vectorised batched scan
the 240-stop cell cost 5391 ms/iteration (~50x worse), and a per-byte comma scan
in data/jsonstream.py halved raw-file indexing throughput.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from core.config import SolverConfig
from core.objective import CombinedObjective
from core.types import Instance, Stop
from models.uncertainty.ambiguity_set import AmbiguitySet
from solvers.alns.alns_solver import solve_alns
from solvers.reliability import ReliabilityEvaluator


def synthetic_probe_instance(n: int, n_zones: int, window_frac: float,
                             seed: int = 0) -> Instance:
    """An arbitrary point set with the *shape* of an LMRRC route. Timing only."""
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * 30.0
    tt = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1) * 2.0
    np.fill_diagonal(tt, 0.0)
    zones = ["DEPOT"] + [f"Z{k % n_zones}" for k in range(n - 1)]
    n_win = int(window_frac * (n - 1))
    windowed = set(rng.choice(np.arange(1, n), size=n_win, replace=False).tolist())
    stops = [
        Stop(
            stop_id=f"S{i}", zone=zones[i],
            lat=34.0 + xy[i, 0] / 100.0, lng=-118.0 + xy[i, 1] / 100.0,
            pkg_count=0 if i == 0 else 2,
            pkg_volume=0.0 if i == 0 else 10.0,
            tw_start=-np.inf,
            tw_end=(200.0 + 400.0 * rng.random()) if i in windowed else np.inf,
            service_time=0.0 if i == 0 else 2.0,
            is_depot=(i == 0),
        )
        for i in range(n)
    ]
    return Instance("probe", "Los Angeles", "DPROBE", stops, tt,
                    meta={"departure_minute_of_day": 480.0})


def run_cell(n: int, n_zones: int, window_frac: float, scenarios: int,
             iterations: int, time_limit: float, lam: float, delta: float,
             seed: int = 0) -> dict:
    inst = synthetic_probe_instance(n, n_zones, window_frac, seed)
    rng = np.random.default_rng(seed + 1)
    tt = inst.travel_time
    travel = (
        tt[None] * (1.0 + 0.2 * rng.standard_normal((scenarios, 1, 1)))
    ).clip(0.0).astype(np.float32)
    service = np.tile(inst.service_time.astype(np.float32), (scenarios, 1))
    service[:, 0] = 0.0
    amb = AmbiguitySet(travel, service, 0.1, delta, "linf", "wasserstein", "probe")
    cost = np.where(np.eye(inst.n, dtype=bool), 0.0, 2.0) + 0.01 * tt
    obj = CombinedObjective(inst, cost, ReliabilityEvaluator(inst, amb), lam=lam)
    cfg = SolverConfig(time_limit_s=time_limit, alns_iterations=iterations,
                       alns_segment=50)

    t0 = time.perf_counter()
    res = solve_alns(inst, obj, cfg)
    elapsed = time.perf_counter() - t0
    return {
        "stops": n,
        "zones": n_zones,
        "windows": int(np.isfinite(inst.tw_end).sum()),
        "scenarios": scenarios,
        "iterations": res.iterations,
        "seconds": round(elapsed, 2),
        "ms_per_iteration": round(1000.0 * elapsed / max(res.iterations, 1), 1),
        "hit_time_limit": res.iterations < iterations,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=[60, 150, 240])
    ap.add_argument("--window-fracs", type=float, nargs="+", default=[0.15, 0.60])
    ap.add_argument("--scenarios", type=int, nargs="+", default=[64])
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--time-limit", type=float, default=45.0)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    logging.disable(logging.INFO)
    rows = []
    for n in args.sizes:
        for wf in args.window_fracs:
            for s in args.scenarios:
                row = run_cell(n, max(n // 8, 2), wf, s, args.iterations,
                               args.time_limit, args.lam, args.delta)
                rows.append(row)
                print(
                    f"stops={row['stops']:4d} zones={row['zones']:3d} "
                    f"windows={row['windows']:4d} S={row['scenarios']:3d} "
                    f"iters={row['iterations']:5d} "
                    f"{row['ms_per_iteration']:8.1f} ms/iteration",
                    flush=True,
                )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Stage-1b: conditional quantile models for travel and service times.

What is being estimated
-----------------------
The LMRRC release does **not** contain realised per-stop arrival times, so the
empirical quantity whose dispersion we model is the observed variation of the
challenge's own time fields across routes:

* ``t_ij`` -- the point-to-point travel time supplied per route.  The same
  (zone-pair, time-of-day, distance) cell recurs across thousands of routes with
  materially different values; that cross-route dispersion is the empirical
  travel-time uncertainty for the cell.
* ``s_i``  -- the planned service time aggregated over the packages at a stop.
  Its dispersion within a (package-count, volume, zone, time-of-day) cell is the
  empirical service-time uncertainty.

Both are therefore fit from real data only -- no distribution is assumed and
nothing is simulated.  See ``docs/DATA_NOTES.md`` for the full argument and its
limitations.

The fitted object exposes conditional quantiles, which :mod:`models.uncertainty.ambiguity_set`
turns into an empirical CDF per arc/stop and hence into the Wasserstein centre
``P_hat`` used by the DRO layer.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from core.config import UncertaintyConfig
from core.logging_utils import get_logger
from core.types import Instance

log = get_logger(__name__)

# Features used for the travel-time head.  ``degrade_features`` (RQ3) keeps only
# the first two, deliberately producing a worse uncertainty model.
TRAVEL_FEATURES = [
    "log_distance",
    "tod_bucket",
    "same_zone",
    "zone_pair_freq",
    "zone_pair_hist_var",
    "arc_hist_median",
    "stops_so_far",
    "stops_remaining",
    "pkg_count_to",
    "pkg_volume_to",
]
TRAVEL_FEATURES_DEGRADED = ["log_distance", "tod_bucket"]

SERVICE_FEATURES = [
    "pkg_count",
    "pkg_volume",
    "tod_bucket",
    "stops_so_far",
    "stops_remaining",
    "zone_size",
    "zone_service_hist_median",
]
SERVICE_FEATURES_DEGRADED = ["pkg_count", "pkg_volume"]

TOD_BUCKET_HOURS = 2.0  # 12 buckets over a day, measured from route departure


def _haversine_min(lat1, lng1, lat2, lng2):
    """Great-circle distance in km (GPS is obfuscated but locally consistent)."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lng2) - np.asarray(lng1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# ------------------------------------------------------------------ statistics
@dataclass
class HistoricalStats:
    """Aggregates computed on the training split only (leakage guard)."""

    zone_pair_freq: dict[tuple[str, str], float] = field(default_factory=dict)
    zone_pair_var: dict[tuple[str, str], float] = field(default_factory=dict)
    zone_pair_median: dict[tuple[str, str], float] = field(default_factory=dict)
    zone_service_median: dict[str, float] = field(default_factory=dict)
    global_travel_median: float = 5.0
    global_service_median: float = 3.0

    def lookup_pair(self, za: str, zb: str) -> tuple[float, float, float]:
        key = (za, zb)
        return (
            self.zone_pair_freq.get(key, 0.0),
            self.zone_pair_var.get(key, 0.0),
            self.zone_pair_median.get(key, self.global_travel_median),
        )

    def lookup_service(self, z: str) -> float:
        return self.zone_service_median.get(z, self.global_service_median)

    def to_json(self) -> str:
        return json.dumps(
            {
                "zone_pair_freq": {f"{a}||{b}": v for (a, b), v in self.zone_pair_freq.items()},
                "zone_pair_var": {f"{a}||{b}": v for (a, b), v in self.zone_pair_var.items()},
                "zone_pair_median": {
                    f"{a}||{b}": v for (a, b), v in self.zone_pair_median.items()
                },
                "zone_service_median": self.zone_service_median,
                "global_travel_median": self.global_travel_median,
                "global_service_median": self.global_service_median,
            }
        )

    @classmethod
    def from_json(cls, text: str) -> "HistoricalStats":
        raw = json.loads(text)

        def unkey(d):
            out = {}
            for k, v in d.items():
                a, b = k.split("||", 1)
                out[(a, b)] = v
            return out

        return cls(
            zone_pair_freq=unkey(raw["zone_pair_freq"]),
            zone_pair_var=unkey(raw["zone_pair_var"]),
            zone_pair_median=unkey(raw["zone_pair_median"]),
            zone_service_median=raw["zone_service_median"],
            global_travel_median=raw["global_travel_median"],
            global_service_median=raw["global_service_median"],
        )


def fit_historical_stats(instances: Iterable[Instance]) -> HistoricalStats:
    """Zone-pair frequency / dispersion tables from the training routes."""
    pair_times: dict[tuple[str, str], list[float]] = {}
    zone_service: dict[str, list[float]] = {}
    all_t: list[float] = []
    all_s: list[float] = []

    for inst in instances:
        zones = inst.zones
        st = inst.service_time
        order = inst.actual_sequence or list(range(inst.n))
        for i, j in zip(order, order[1:]):
            key = (zones[i], zones[j])
            t = float(inst.travel_time[i, j])
            pair_times.setdefault(key, []).append(t)
            all_t.append(t)
        for i in range(inst.n):
            if inst.stops[i].is_depot:
                continue
            zone_service.setdefault(zones[i], []).append(float(st[i]))
            all_s.append(float(st[i]))

    total = max(sum(len(v) for v in pair_times.values()), 1)
    return HistoricalStats(
        zone_pair_freq={k: len(v) / total for k, v in pair_times.items()},
        zone_pair_var={k: float(np.var(v)) for k, v in pair_times.items()},
        zone_pair_median={k: float(np.median(v)) for k, v in pair_times.items()},
        zone_service_median={k: float(np.median(v)) for k, v in zone_service.items()},
        global_travel_median=float(np.median(all_t)) if all_t else 5.0,
        global_service_median=float(np.median(all_s)) if all_s else 3.0,
    )


# -------------------------------------------------------------- feature tables
def travel_rows(inst: Instance, stats: HistoricalStats) -> pd.DataFrame:
    """One row per arc traversed by sigma* (or by index order if absent)."""
    order = inst.actual_sequence or list(range(inst.n))
    zones = inst.zones
    lat = np.array([s.lat for s in inst.stops])
    lng = np.array([s.lng for s in inst.stops])
    pkgc, pkgv = inst.pkg_count, inst.pkg_volume
    dep_min = float(inst.meta.get("departure_minute_of_day", 0.0))

    recs = []
    cum = 0.0
    for p, (i, j) in enumerate(zip(order, order[1:])):
        freq, var, med = stats.lookup_pair(zones[i], zones[j])
        t = float(inst.travel_time[i, j])
        recs.append(
            {
                "route_id": inst.route_id,
                "metro": inst.metro,
                "i": i,
                "j": j,
                "log_distance": float(
                    np.log1p(_haversine_min(lat[i], lng[i], lat[j], lng[j]))
                ),
                "tod_bucket": float(((dep_min + cum) / 60.0) // TOD_BUCKET_HOURS % 12),
                "same_zone": float(zones[i] == zones[j]),
                "zone_pair_freq": freq,
                "zone_pair_hist_var": var,
                "arc_hist_median": med,
                "stops_so_far": float(p),
                "stops_remaining": float(len(order) - p - 1),
                "pkg_count_to": float(pkgc[j]),
                "pkg_volume_to": float(pkgv[j]),
                "target": t,
            }
        )
        cum += float(inst.service_time[i]) + t
    return pd.DataFrame.from_records(recs)


def service_rows(inst: Instance, stats: HistoricalStats) -> pd.DataFrame:
    order = inst.actual_sequence or list(range(inst.n))
    zones = inst.zones
    sizes = {z: len(v) for z, v in inst.zone_groups().items()}
    pkgc, pkgv, st = inst.pkg_count, inst.pkg_volume, inst.service_time
    dep_min = float(inst.meta.get("departure_minute_of_day", 0.0))

    recs = []
    cum = 0.0
    for p, i in enumerate(order):
        if not inst.stops[i].is_depot:
            recs.append(
                {
                    "route_id": inst.route_id,
                    "metro": inst.metro,
                    "i": i,
                    "pkg_count": float(pkgc[i]),
                    "pkg_volume": float(pkgv[i]),
                    "tod_bucket": float(((dep_min + cum) / 60.0) // TOD_BUCKET_HOURS % 12),
                    "stops_so_far": float(p),
                    "stops_remaining": float(len(order) - p - 1),
                    "zone_size": float(sizes.get(zones[i], 1)),
                    "zone_service_hist_median": stats.lookup_service(zones[i]),
                    "target": float(st[i]),
                }
            )
        nxt = order[p + 1] if p + 1 < len(order) else order[0]
        cum += float(st[i]) + float(inst.travel_time[i, nxt])
    return pd.DataFrame.from_records(recs)


def build_training_tables(
    instances: Sequence[Instance], stats: HistoricalStats
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = [travel_rows(i, stats) for i in instances]
    sv = [service_rows(i, stats) for i in instances]
    tr = pd.concat(tr, ignore_index=True) if tr else pd.DataFrame()
    sv = pd.concat(sv, ignore_index=True) if sv else pd.DataFrame()
    return tr, sv


# ------------------------------------------------------------------ the model
class _QuantileHead:
    """One LightGBM model per quantile level, with a robust linear fallback."""

    def __init__(self, quantiles: Sequence[float], features: Sequence[str], **lgb_kw):
        self.quantiles = tuple(quantiles)
        self.features = list(features)
        self.models: dict[float, object] = {}
        self.fallback: dict[float, float] = {}
        self.lgb_kw = dict(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            verbose=-1,
        )
        self.lgb_kw.update(lgb_kw)

    def fit(self, df: pd.DataFrame) -> "_QuantileHead":
        x = df[self.features].to_numpy(dtype=float)
        y = df["target"].to_numpy(dtype=float)
        # Unconditional fallback quantiles, always available.
        self.fallback = {q: float(np.quantile(y, q)) for q in self.quantiles}
        if len(df) < 50:
            log.warning("only %d rows: using unconditional quantiles", len(df))
            return self
        try:
            import lightgbm as lgb
        except ImportError:  # pragma: no cover - documented optional dependency
            log.warning("lightgbm unavailable: using unconditional quantiles")
            return self
        for q in self.quantiles:
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, **self.lgb_kw)
            m.fit(x, y)
            self.models[q] = m
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return an ``(n_rows, n_quantiles)`` array, monotone across columns."""
        n = len(df)
        out = np.empty((n, len(self.quantiles)), dtype=float)
        if self.models:
            x = df[self.features].to_numpy(dtype=float)
            for k, q in enumerate(self.quantiles):
                out[:, k] = self.models[q].predict(x)
        else:
            for k, q in enumerate(self.quantiles):
                out[:, k] = self.fallback[q]
        out = np.maximum.accumulate(out, axis=1)  # enforce monotone quantiles
        return np.maximum(out, 0.0)


@dataclass
class QuantileUncertaintyModel:
    """Stage-1b artefact: conditional quantiles for ``t_ij`` and ``s_i``."""

    cfg: UncertaintyConfig
    stats: HistoricalStats
    travel_head: _QuantileHead
    service_head: _QuantileHead

    @property
    def quantiles(self) -> tuple[float, ...]:
        return self.travel_head.quantiles

    # --------------------------------------------------------------- training
    @classmethod
    def fit(
        cls,
        instances: Sequence[Instance],
        cfg: UncertaintyConfig,
        stats: HistoricalStats | None = None,
    ) -> "QuantileUncertaintyModel":
        rng = np.random.default_rng(cfg.seed)
        instances = list(instances)
        if cfg.degrade_train_fraction < 1.0 and len(instances) > 2:
            k = max(2, int(round(cfg.degrade_train_fraction * len(instances))))
            idx = rng.choice(len(instances), size=k, replace=False)
            instances = [instances[i] for i in sorted(idx)]
            log.info("RQ3 degradation: training on %d routes", len(instances))

        stats = stats or fit_historical_stats(instances)
        tr, sv = build_training_tables(instances, stats)
        qs = cfg.effective_quantiles()
        tf = TRAVEL_FEATURES_DEGRADED if cfg.degrade_features else TRAVEL_FEATURES
        sf = SERVICE_FEATURES_DEGRADED if cfg.degrade_features else SERVICE_FEATURES
        log.info(
            "Stage-1b fit: %d travel rows, %d service rows, quantiles=%s",
            len(tr), len(sv), qs,
        )
        return cls(
            cfg=cfg,
            stats=stats,
            travel_head=_QuantileHead(qs, tf).fit(tr),
            service_head=_QuantileHead(qs, sf).fit(sv),
        )

    # ------------------------------------------------------------- prediction
    def predict_travel(self, rows: pd.DataFrame) -> np.ndarray:
        return self.travel_head.predict(rows)

    def predict_service(self, rows: pd.DataFrame) -> np.ndarray:
        return self.service_head.predict(rows)

    def instance_travel_quantiles(self, inst: Instance) -> np.ndarray:
        """``(n, n, Q)`` conditional travel-time quantiles for every arc."""
        n, q = inst.n, len(self.quantiles)
        rows = _all_arc_rows(inst, self.stats)
        pred = self.predict_travel(rows)
        out = np.zeros((n, n, q), dtype=np.float32)
        out[rows["i"].to_numpy(), rows["j"].to_numpy()] = pred
        # Anchor the conditional distribution on the route's own observed t_ij:
        # shift so the median quantile equals the supplied value.
        mid = len(self.quantiles) // 2
        shift = inst.travel_time.astype(np.float32) - out[:, :, mid]
        out = np.maximum(out + shift[:, :, None], 0.0)
        for k in range(q):
            np.fill_diagonal(out[:, :, k], 0.0)
        return out

    def instance_service_quantiles(self, inst: Instance) -> np.ndarray:
        """``(n, Q)`` conditional service-time quantiles; depot row is zero."""
        rows = _all_stop_rows(inst, self.stats)
        pred = self.predict_service(rows)
        out = np.zeros((inst.n, len(self.quantiles)), dtype=np.float32)
        out[rows["i"].to_numpy()] = pred
        mid = len(self.quantiles) // 2
        planned = inst.service_time.astype(np.float32)
        shift = planned - out[:, mid]
        out = np.maximum(out + shift[:, None], 0.0)
        out[0, :] = 0.0
        return out

    # ------------------------------------------------------------------- io
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load(path: str | Path) -> "QuantileUncertaintyModel":
        with Path(path).open("rb") as fh:
            return pickle.load(fh)


def _all_arc_rows(inst: Instance, stats: HistoricalStats) -> pd.DataFrame:
    """Feature rows for every ordered arc of an instance (prediction time)."""
    n = inst.n
    zones = inst.zones
    lat = np.array([s.lat for s in inst.stops])
    lng = np.array([s.lng for s in inst.stops])
    pkgc, pkgv = inst.pkg_count, inst.pkg_volume
    dep_min = float(inst.meta.get("departure_minute_of_day", 0.0))
    # Position-dependent features are unknown before sequencing; use the route's
    # midpoint, which is what the optimizer sees when it scores any arc.
    mid_pos = (n - 1) / 2.0
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    freq = np.empty(len(ii))
    var = np.empty(len(ii))
    med = np.empty(len(ii))
    for k, (a, b) in enumerate(zip(ii, jj)):
        freq[k], var[k], med[k] = stats.lookup_pair(zones[a], zones[b])
    return pd.DataFrame(
        {
            "i": ii,
            "j": jj,
            "log_distance": np.log1p(
                _haversine_min(lat[ii], lng[ii], lat[jj], lng[jj])
            ),
            "tod_bucket": np.full(len(ii), (dep_min / 60.0) // TOD_BUCKET_HOURS % 12),
            "same_zone": (np.array(zones)[ii] == np.array(zones)[jj]).astype(float),
            "zone_pair_freq": freq,
            "zone_pair_hist_var": var,
            "arc_hist_median": med,
            "stops_so_far": np.full(len(ii), mid_pos),
            "stops_remaining": np.full(len(ii), mid_pos),
            "pkg_count_to": pkgc[jj],
            "pkg_volume_to": pkgv[jj],
        }
    )


def _all_stop_rows(inst: Instance, stats: HistoricalStats) -> pd.DataFrame:
    n = inst.n
    zones = inst.zones
    sizes = {z: len(v) for z, v in inst.zone_groups().items()}
    dep_min = float(inst.meta.get("departure_minute_of_day", 0.0))
    mid_pos = (n - 1) / 2.0
    idx = np.arange(n)
    return pd.DataFrame(
        {
            "i": idx,
            "pkg_count": inst.pkg_count,
            "pkg_volume": inst.pkg_volume,
            "tod_bucket": np.full(n, (dep_min / 60.0) // TOD_BUCKET_HOURS % 12),
            "stops_so_far": np.full(n, mid_pos),
            "stops_remaining": np.full(n, mid_pos),
            "zone_size": np.array([sizes.get(z, 1) for z in zones], dtype=float),
            "zone_service_hist_median": np.array(
                [stats.lookup_service(z) for z in zones], dtype=float
            ),
        }
    )

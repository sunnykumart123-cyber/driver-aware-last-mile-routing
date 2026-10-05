"""Stage-1a secondary model: a gradient-boosted pairwise transition scorer.

This plays three roles required by §4a of the spec:

1. a da-Costa-style ML baseline in its own right (score every candidate arc,
   decode greedily);
2. an interpretability check on the pointer network (LightGBM gain importances
   are reported in ``eval_behavior.py`` next to the pointer's metrics);
3. a fallback source of ``c^beh_ij`` when the pointer network is unavailable,
   which keeps the optimisation stack runnable on its own.

Training is a binary discrimination task: for each position of sigma* the arc the
driver actually took is a positive and ``n_negatives`` other feasible arcs from
the same state are negatives, so the learned score is a calibrated
state-conditional preference rather than an unconditional arc score.
"""
from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from core.logging_utils import get_logger
from core.types import DEPOT, Instance
from models.behavior.dataset import MAX_ZONES, zone_ranks

log = get_logger(__name__)

FEATURES = [
    "travel_time",
    "log_distance",
    "same_zone",
    "zone_rank_delta",
    "zone_rank_transition_freq",
    "zone_pair_freq",
    "tt_rank_among_remaining",
    "tt_ratio_to_nearest",
    "stops_remaining",
    "tod_bucket",
    "pkg_count_to",
    "pkg_volume_to",
    "tw_end_slack",
    "zone_stops_remaining_in_current",
    "returns_to_abandoned_zone",
]


def _hav(lat1, lng1, lat2, lng2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lng2) - np.asarray(lng1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


@dataclass
class TransitionTables:
    """Frequency tables, fit on the training split only."""

    zone_rank_freq: np.ndarray = field(
        default_factory=lambda: np.zeros((MAX_ZONES + 1, MAX_ZONES + 1))
    )
    zone_pair_freq: dict[tuple[str, str], float] = field(default_factory=dict)

    @classmethod
    def fit(cls, instances: Sequence[Instance]) -> "TransitionTables":
        rank_counts = np.zeros((MAX_ZONES + 1, MAX_ZONES + 1))
        pair_counts: dict[tuple[str, str], float] = defaultdict(float)
        total = 0
        for inst in instances:
            if inst.actual_sequence is None:
                continue
            zr, _ = zone_ranks(inst)
            zones = inst.zones
            order = inst.actual_sequence
            for i, j in zip(order, order[1:]):
                rank_counts[zr[i], zr[j]] += 1.0
                pair_counts[(zones[i], zones[j])] += 1.0
                total += 1
        total = max(total, 1)
        row = rank_counts.sum(axis=1, keepdims=True)
        return cls(
            zone_rank_freq=rank_counts / np.where(row > 0, row, 1.0),
            zone_pair_freq={k: v / total for k, v in pair_counts.items()},
        )


def _state_features(
    inst: Instance,
    cur: int,
    candidates: np.ndarray,
    remaining: np.ndarray,
    tables: TransitionTables,
    zr: np.ndarray,
    lat: np.ndarray,
    lng: np.ndarray,
    zone_remaining: dict[str, int],
    visited_zones: set[str],
    elapsed: float,
) -> pd.DataFrame:
    tt = inst.travel_time[cur, candidates]
    tt_rem = inst.travel_time[cur, remaining]
    nearest = float(tt_rem.min()) if len(tt_rem) else 1.0
    order_rank = np.argsort(np.argsort(tt_rem))
    rank_lookup = dict(zip(remaining.tolist(), order_rank.tolist()))
    zones = inst.zones
    dep_min = float(inst.meta.get("departure_minute_of_day", 0.0))
    b = inst.tw_end

    return pd.DataFrame(
        {
            "travel_time": tt,
            "log_distance": np.log1p(
                _hav(lat[cur], lng[cur], lat[candidates], lng[candidates])
            ),
            "same_zone": (np.array(zones)[candidates] == zones[cur]).astype(float),
            "zone_rank_delta": (zr[candidates] - zr[cur]).astype(float),
            "zone_rank_transition_freq": tables.zone_rank_freq[zr[cur], zr[candidates]],
            "zone_pair_freq": [
                tables.zone_pair_freq.get((zones[cur], zones[j]), 0.0) for j in candidates
            ],
            "tt_rank_among_remaining": [
                float(rank_lookup.get(int(j), len(remaining))) for j in candidates
            ],
            # log-ratio with a floor of 6 seconds: raw t/min(t) exploded to
            # ~25,000 on routes with a near-zero arc, which made the feature
            # pure noise.
            "tt_ratio_to_nearest": np.log1p(tt / max(nearest, 0.1)),
            "stops_remaining": np.full(len(candidates), float(len(remaining))),
            "tod_bucket": np.full(len(candidates), ((dep_min + elapsed) / 120.0) % 12),
            "pkg_count_to": inst.pkg_count[candidates],
            "pkg_volume_to": inst.pkg_volume[candidates],
            "tw_end_slack": np.where(
                np.isfinite(b[candidates]),
                b[candidates] - (elapsed + tt),
                1440.0,
            ),
            "zone_stops_remaining_in_current": np.full(
                len(candidates), float(zone_remaining.get(zones[cur], 0))
            ),
            # Re-entering a zone the route already worked and left.  The
            # earlier version of this feature keyed on *finished* zones, which
            # is vacuous: a zone is only finished once all its stops are
            # visited, so no remaining candidate can ever be in one.  It was
            # constant 0 across every training row.  Abandoned-and-returned-to
            # is the behaviour that actually occurs -- about three fifths of
            # real routes do it (docs/DATA_NOTES.md §8.1).
            "returns_to_abandoned_zone": np.array(
                [
                    1.0 if (zones[j] != zones[cur] and zones[j] in visited_zones)
                    else 0.0
                    for j in candidates
                ]
            ),
        }
    )


@dataclass
class GBMTransitionModel:
    tables: TransitionTables
    booster: object | None = None
    n_negatives: int = 24
    seed: int = 20260101
    prob_floor: float = 1e-6
    temperature: float = 1.0
    objective: str = "rank"           # rank | binary
    importances: dict[str, float] = field(default_factory=dict)

    # --------------------------------------------------------------- training
    @classmethod
    def fit(
        cls,
        instances: Sequence[Instance],
        n_negatives: int = 24,
        seed: int = 20260101,
        objective: str = "rank",
        **lgb_kw,
    ) -> "GBMTransitionModel":
        """Fit the pairwise scorer.

        ``objective="rank"`` trains LightGBM's LambdaRank over each decision
        state as a query group, which is what decoding actually does: pick the
        best of all remaining stops.  The earlier binary-classification setup
        learned to separate the driver's choice from ``n_negatives`` *random*
        alternatives and was then asked at inference time to rank ~150 of them
        -- a train/inference mismatch.  ``objective="binary"`` keeps the old
        behaviour for comparison.
        """
        tables = TransitionTables.fit(instances)
        model = cls(tables=tables, n_negatives=n_negatives, seed=seed,
                    objective=objective)
        rows, labels, group_sizes = model._training_rows(instances)
        if rows.empty:
            log.warning("no training rows for the GBM transition model")
            return model
        try:
            import lightgbm as lgb
        except ImportError:  # pragma: no cover
            log.warning("lightgbm unavailable: GBM transition model stays untrained")
            return model
        params = dict(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=50,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            verbose=-1,
            random_state=seed,
        )
        params.update(lgb_kw)
        x = rows[FEATURES].to_numpy(dtype=float)
        if objective == "rank":
            est = lgb.LGBMRanker(objective="lambdarank", **params)
            est.fit(x, labels, group=group_sizes)
        else:
            est = lgb.LGBMClassifier(**params)
            est.fit(x, labels)
        model.booster = est
        gains = getattr(est, "feature_importances_", None)
        if gains is not None:
            total = float(np.sum(gains)) or 1.0
            model.importances = {
                f: float(g) / total for f, g in zip(FEATURES, gains)
            }
        log.info(
            "GBM transition model (%s): %d rows, %d decision states",
            objective, len(rows), len(group_sizes),
        )
        return model

    def _training_rows(self, instances: Sequence[Instance]):
        rng = np.random.default_rng(self.seed)
        frames, labels = [], []
        group_sizes: list[int] = []
        groups = 0
        for inst in instances:
            if inst.actual_sequence is None:
                continue
            zr, _ = zone_ranks(inst)
            lat = np.array([s.lat for s in inst.stops])
            lng = np.array([s.lng for s in inst.stops])
            order = inst.actual_sequence
            zone_remaining = {z: len(v) for z, v in inst.zone_groups().items()}
            visited_zones: set[str] = set()
            elapsed = 0.0
            unvisited = set(range(inst.n)) - {DEPOT}
            for p, cur in enumerate(order[:-1]):
                visited_zones.add(inst.zones[cur])
                nxt = order[p + 1]
                remaining = np.array(sorted(unvisited), dtype=int)
                if len(remaining) == 0:
                    break
                negatives = [j for j in remaining if j != nxt]
                if negatives:
                    k = min(self.n_negatives, len(negatives))
                    negatives = rng.choice(negatives, size=k, replace=False).tolist()
                cands = np.array([nxt] + list(negatives), dtype=int)
                frames.append(
                    _state_features(
                        inst, cur, cands, remaining, self.tables, zr, lat, lng,
                        zone_remaining, visited_zones, elapsed,
                    )
                )
                labels.extend([1] + [0] * len(negatives))
                group_sizes.append(1 + len(negatives))
                groups += 1

                elapsed += float(inst.service_time[cur] + inst.travel_time[cur, nxt])
                unvisited.discard(nxt)
                z = inst.zones[cur]
                zone_remaining[z] = max(zone_remaining.get(z, 1) - 1, 0)
        if not frames:
            return pd.DataFrame(columns=FEATURES), np.array([]), []
        return pd.concat(frames, ignore_index=True), np.array(labels), group_sizes

    # ------------------------------------------------------------- prediction
    def _scores(self, inst: Instance, cur: int, candidates: np.ndarray,
                remaining: np.ndarray, zr, lat, lng, zone_remaining,
                visited_zones, elapsed) -> np.ndarray:
        x = _state_features(
            inst, cur, candidates, remaining, self.tables, zr, lat, lng,
            zone_remaining, visited_zones, elapsed,
        )
        if self.booster is None:
            # Untrained fallback: nearest-neighbour preference, zone-aware.
            tt = x["travel_time"].to_numpy()
            return -tt / max(tt.max(), 1e-6) + 0.5 * x["same_zone"].to_numpy()
        feats = x[FEATURES].to_numpy(dtype=float)
        if self.objective == "rank":
            return np.asarray(self.booster.predict(feats), dtype=float)
        return self.booster.predict_proba(feats)[:, 1]

    def transition_matrix(self, inst: Instance) -> np.ndarray:
        """Dense ``p_ij``: row-wise softmax of the pairwise score."""
        n = inst.n
        zr, _ = zone_ranks(inst)
        lat = np.array([s.lat for s in inst.stops])
        lng = np.array([s.lng for s in inst.stops])
        zone_remaining = {z: len(v) for z, v in inst.zone_groups().items()}
        all_stops = np.arange(n)
        p = np.zeros((n, n), dtype=float)
        # Scored at the route's mid-progress state: the optimizer needs a single
        # static cost per arc, so the state features are held at their midpoint.
        elapsed = 0.5 * float(np.median(inst.travel_time) * (n - 1))
        for i in range(n):
            cands = all_stops[(all_stops != i) & (all_stops != DEPOT)]
            if len(cands) == 0:
                continue
            s = self._scores(
                inst, i, cands, cands, zr, lat, lng, zone_remaining,
                {inst.zones[i]}, elapsed,
            )
            s = np.asarray(s, dtype=float) / max(self.temperature, 1e-6)
            e = np.exp(s - s.max())
            p[i, cands] = e / e.sum()
        np.fill_diagonal(p, 0.0)
        return np.maximum(p, self.prob_floor)

    def behavior_cost_matrix(self, inst: Instance) -> np.ndarray:
        return -np.log(self.transition_matrix(inst))

    def greedy_order(self, inst: Instance) -> list[int]:
        """da-Costa-style decoding: repeatedly take the highest-scoring arc."""
        n = inst.n
        zr, _ = zone_ranks(inst)
        lat = np.array([s.lat for s in inst.stops])
        lng = np.array([s.lng for s in inst.stops])
        zone_remaining = {z: len(v) for z, v in inst.zone_groups().items()}
        visited_zones: set[str] = set()
        unvisited = set(range(n)) - {DEPOT}
        order = [DEPOT]
        cur, elapsed = DEPOT, 0.0
        while unvisited:
            visited_zones.add(inst.zones[cur])
            remaining = np.array(sorted(unvisited), dtype=int)
            s = self._scores(
                inst, cur, remaining, remaining, zr, lat, lng,
                zone_remaining, visited_zones, elapsed,
            )
            nxt = int(remaining[int(np.argmax(s))])
            elapsed += float(inst.service_time[cur] + inst.travel_time[cur, nxt])
            z = inst.zones[cur]
            zone_remaining[z] = max(zone_remaining.get(z, 1) - 1, 0)
            order.append(nxt)
            unvisited.discard(nxt)
            cur = nxt
        return order

    # --------------------------------------------------------------------- io
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load(path: str | Path) -> "GBMTransitionModel":
        with Path(path).open("rb") as fh:
            return pickle.load(fh)

"""Feature construction for Stage-1a (behaviour models).

Zone identity is encoded *structurally*, not by its literal LMRRC sort-zone
string: a zone becomes an integer rank by distance of its centroid from the
station.  Raw zone strings are station-specific, so embedding them directly
cannot transfer to a held-out metro -- and held-out-metro generalisation is the
protocol this thesis is evaluated under (§6, §7).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from core.types import DEPOT, Instance

# Per-stop continuous features, in order.
STOP_FEATURES = [
    "log_pkg_count",
    "log_pkg_volume",
    "tw_start_h",
    "tw_width_h",
    "has_window",
    "dlat_zone",
    "dlng_zone",
    "dlat_route",
    "dlng_route",
    "tt_from_depot_h",
    "tt_to_depot_h",
    "zone_size_frac",
    "is_depot",
]
N_STOP_FEATURES = len(STOP_FEATURES)
MAX_ZONES = 64        # zone-rank embedding vocabulary (plus 0 for the depot)
_BIG_WINDOW_H = 24.0


@dataclass
class EncodedRoute:
    """Tensor-ready encoding of one instance."""

    route_id: str
    metro: str
    features: np.ndarray      # (n, N_STOP_FEATURES) float32
    zone_rank: np.ndarray     # (n,) int64 in [0, MAX_ZONES]
    zone_of: np.ndarray       # (n,) int64 local zone index
    target: np.ndarray | None # (n,) int64 sigma* as stop indices, or None
    n: int
    travel_norm: np.ndarray | None = None   # (n, n) float32, median-normalised

    def __len__(self) -> int:
        return self.n


def zone_ranks(inst: Instance) -> tuple[np.ndarray, np.ndarray]:
    """Rank zones by centroid travel time from the depot.

    Returns ``(zone_rank_per_stop, local_zone_index_per_stop)``.  The depot gets
    rank 0; real zones get 1..MAX_ZONES (clipped).
    """
    groups = inst.zone_groups()
    tt0 = inst.travel_time[DEPOT]
    order = sorted(groups, key=lambda z: float(np.mean(tt0[groups[z]])))
    rank = {z: min(k + 1, MAX_ZONES) for k, z in enumerate(order)}
    local = {z: k for k, z in enumerate(order)}
    zr = np.zeros(inst.n, dtype=np.int64)
    zl = np.zeros(inst.n, dtype=np.int64)
    for i, s in enumerate(inst.stops):
        if s.is_depot:
            continue
        zr[i] = rank[s.zone]
        zl[i] = local[s.zone]
    return zr, zl


def encode_instance(inst: Instance) -> EncodedRoute:
    n = inst.n
    lat = np.array([s.lat for s in inst.stops], dtype=float)
    lng = np.array([s.lng for s in inst.stops], dtype=float)
    groups = inst.zone_groups()
    zr, zl = zone_ranks(inst)

    zone_c_lat = np.zeros(n)
    zone_c_lng = np.zeros(n)
    zone_frac = np.zeros(n)
    for z, idx in groups.items():
        zone_c_lat[idx] = lat[idx].mean()
        zone_c_lng[idx] = lng[idx].mean()
        zone_frac[idx] = len(idx) / max(n - 1, 1)
    mask = ~np.array([s.is_depot for s in inst.stops])
    # The depot belongs to no sort zone: reference it against itself so its
    # offset features are zero instead of the size of its raw coordinates.
    zone_c_lat[~mask] = lat[~mask]
    zone_c_lng[~mask] = lng[~mask]
    if mask.any():
        r_lat, r_lng = lat[mask].mean(), lng[mask].mean()
    else:  # pragma: no cover - a route is never depot-only
        r_lat, r_lng = lat[0], lng[0]

    tw_s = inst.tw_start
    tw_e = inst.tw_end
    finite = np.isfinite(tw_e)
    tw_start_h = np.where(np.isfinite(tw_s), tw_s / 60.0, 0.0)
    width = np.where(finite & np.isfinite(tw_s), (tw_e - tw_s) / 60.0, _BIG_WINDOW_H)

    feats = np.column_stack(
        [
            np.log1p(inst.pkg_count),
            np.log1p(inst.pkg_volume),
            tw_start_h,
            np.clip(width, 0.0, _BIG_WINDOW_H),
            finite.astype(float),
            (lat - zone_c_lat) * 100.0,
            (lng - zone_c_lng) * 100.0,
            (lat - r_lat) * 100.0,
            (lng - r_lng) * 100.0,
            inst.travel_time[DEPOT] / 60.0,
            inst.travel_time[:, DEPOT] / 60.0,
            zone_frac,
            np.array([float(s.is_depot) for s in inst.stops]),
        ]
    ).astype(np.float32)

    target = (
        np.asarray(inst.actual_sequence, dtype=np.int64)
        if inst.actual_sequence is not None
        else None
    )
    # Pairwise travel times, normalised by the route's own median arc so the
    # scale transfers across metros.  The decoder needs these: "which stop comes
    # next" is overwhelmingly a question about proximity to the *current* stop,
    # and per-stop features cannot express that.
    tt = inst.travel_time.astype(np.float32)
    med = float(np.median(tt[~np.eye(n, dtype=bool)])) or 1.0
    travel_norm = (tt / med).astype(np.float32)

    return EncodedRoute(
        route_id=inst.route_id,
        metro=inst.metro,
        features=feats,
        zone_rank=zr,
        zone_of=zl,
        target=target,
        n=n,
        travel_norm=travel_norm,
    )


def encode_many(instances: Sequence[Instance]) -> list[EncodedRoute]:
    return [encode_instance(i) for i in instances]


def collate(batch: Sequence[EncodedRoute]):
    """Pad a batch of routes to a common length.

    Returns a dict of torch tensors; importing torch lazily keeps the
    data-preprocessing path usable without it.
    """
    import torch

    bs = len(batch)
    n_max = max(len(b) for b in batch)
    feats = torch.zeros(bs, n_max, N_STOP_FEATURES)
    zr = torch.zeros(bs, n_max, dtype=torch.long)
    zone_of = torch.full((bs, n_max), -1, dtype=torch.long)
    pad = torch.ones(bs, n_max, dtype=torch.bool)      # True == padding
    tgt = torch.full((bs, n_max), -100, dtype=torch.long)
    travel = torch.zeros(bs, n_max, n_max)

    for k, b in enumerate(batch):
        m = len(b)
        feats[k, :m] = torch.from_numpy(b.features)
        zr[k, :m] = torch.from_numpy(b.zone_rank)
        zone_of[k, :m] = torch.from_numpy(b.zone_of)
        pad[k, :m] = False
        if b.target is not None:
            tgt[k, :m] = torch.from_numpy(b.target)
        if b.travel_norm is not None:
            travel[k, :m, :m] = torch.from_numpy(b.travel_norm)
    return {
        "features": feats,
        "zone_rank": zr,
        "zone_of": zone_of,
        "pad_mask": pad,
        "target": tgt,
        "travel": travel,
        "lengths": torch.tensor([len(b) for b in batch], dtype=torch.long),
        "route_ids": [b.route_id for b in batch],
    }

"""Core data structures shared by every stage of the pipeline.

Time convention
---------------
All times are *minutes since the route's departure epoch* (the
``departure_time_utc`` of the route in the LMRRC data). Travel times come
straight from the challenge's ``travel_times.json`` (seconds) and are converted
to minutes exactly once, in :mod:`data.preprocess`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

DEPOT = 0


@dataclass(frozen=True)
class Stop:
    """One delivery stop (or the depot, when ``is_depot``)."""

    stop_id: str
    zone: str
    lat: float
    lng: float
    pkg_count: int
    pkg_volume: float
    tw_start: float          # minutes from departure; -inf if no window
    tw_end: float            # minutes from departure; +inf if no window
    service_time: float      # planned/mean service time in minutes
    is_depot: bool = False

    @property
    def has_window(self) -> bool:
        return bool(np.isfinite(self.tw_end))


@dataclass
class Instance:
    """A single route instance: the unit of optimization.

    ``travel_time[i, j]`` is the observed point-to-point travel time from stop
    ``i`` to stop ``j`` in minutes. Index 0 is always the depot/station.
    """

    route_id: str
    metro: str
    station: str
    stops: list[Stop]
    travel_time: np.ndarray
    actual_sequence: list[int] | None = None   # sigma*, training routes only
    quality: str | None = None                 # High / Medium / Low
    executor_capacity: float = float("inf")
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ shape
    def __post_init__(self) -> None:
        n = len(self.stops)
        self.travel_time = np.asarray(self.travel_time, dtype=float)
        if self.travel_time.shape != (n, n):
            raise ValueError(
                f"travel_time shape {self.travel_time.shape} != ({n}, {n})"
            )
        if not self.stops[DEPOT].is_depot:
            raise ValueError("stops[0] must be the depot")
        if self.actual_sequence is not None:
            self._check_permutation(self.actual_sequence)

    def _check_permutation(self, order: Sequence[int]) -> None:
        if sorted(order) != list(range(self.n)):
            raise ValueError("actual_sequence is not a permutation of all stops")
        if order[0] != DEPOT:
            raise ValueError("actual_sequence must start at the depot")

    @property
    def n(self) -> int:
        return len(self.stops)

    # ---------------------------------------------------------------- vectors
    @property
    def zones(self) -> list[str]:
        return [s.zone for s in self.stops]

    @property
    def tw_end(self) -> np.ndarray:
        return np.array([s.tw_end for s in self.stops], dtype=float)

    @property
    def tw_start(self) -> np.ndarray:
        return np.array([s.tw_start for s in self.stops], dtype=float)

    @property
    def service_time(self) -> np.ndarray:
        return np.array([s.service_time for s in self.stops], dtype=float)

    @property
    def pkg_count(self) -> np.ndarray:
        return np.array([s.pkg_count for s in self.stops], dtype=float)

    @property
    def pkg_volume(self) -> np.ndarray:
        return np.array([s.pkg_volume for s in self.stops], dtype=float)

    def zone_groups(self) -> dict[str, list[int]]:
        """Stop indices grouped by sort zone, depot excluded."""
        groups: dict[str, list[int]] = {}
        for i, s in enumerate(self.stops):
            if s.is_depot:
                continue
            groups.setdefault(s.zone, []).append(i)
        return groups

    def route_duration(self, order: Sequence[int]) -> float:
        """Deterministic duration of ``order`` (closed tour) with planned service."""
        st = self.service_time
        order = list(order)
        total = 0.0
        for a, b in zip(order, order[1:] + [order[0]]):
            total += st[a] + self.travel_time[a, b]
        return total


@dataclass
class Solution:
    """A route as an ordered list of stop indices, starting at the depot."""

    order: list[int]
    objective: float = float("nan")
    behavior_cost: float = float("nan")
    risk_cost: float = float("nan")
    solver: str = "unknown"
    runtime_s: float = float("nan")
    lower_bound: float = float("nan")
    meta: dict = field(default_factory=dict)

    def as_sequence(self, inst: Instance) -> dict[str, int]:
        """The LMRRC submission format: ``{stop_id: position}``."""
        return {inst.stops[i].stop_id: p for p, i in enumerate(self.order)}

    @property
    def gap(self) -> float:
        if not np.isfinite(self.lower_bound) or not np.isfinite(self.objective):
            return float("nan")
        denom = max(abs(self.objective), 1e-9)
        return (self.objective - self.lower_bound) / denom


def is_valid_order(order: Iterable[int], n: int) -> bool:
    order = list(order)
    return len(order) == n and sorted(order) == list(range(n)) and order[0] == DEPOT


def zone_contiguous(order: Sequence[int], zones: Sequence[str]) -> bool:
    """True if every zone occupies one contiguous block (depot excluded)."""
    seen: list[str] = []
    for i in list(order)[1:]:
        z = zones[i]
        if not seen or seen[-1] != z:
            if z in seen:
                return False
            seen.append(z)
    return True

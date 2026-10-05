"""Mutable route state with incremental, vectorised risk evaluation.

This is the piece that makes the ALNS affordable.  A naive implementation would
re-evaluate the whole DRO layer for every candidate insertion; instead the state
caches the ``(S, n)`` prefix-arrival array and exploits three facts:

1. inserting a stop shifts the arrival time of every *later* position by the
   same scenario-wise delay, and leaves earlier positions untouched;
2. only stops with a finite time window contribute to ``sum eta_i``;
3. under zone precedence a removed stop has only a handful of legal positions
   (inside its own zone block, or at a block boundary if its zone is absent).

All candidate positions for one stop are scored in **one** vectorised call:
the arrival tensor is ``(S, P, W)`` for ``P`` candidate positions and ``W``
window-carrying stops, and the CVaR is a single partition over the scenario
axis.  Nothing inside the search loop solves an optimisation problem, which is
what §5b of the build spec requires.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from core.objective import CombinedObjective
from core.types import DEPOT, Instance


class RouteState:
    """A partial or complete route, with cached arrivals, eta and objective."""

    def __init__(self, obj: CombinedObjective, order: Sequence[int],
                 zone_precedence: bool = True, candidate_cap: int = 0):
        self.obj = obj
        self.inst: Instance = obj.inst
        self.ev = obj.evaluator
        self.amb = self.ev.amb
        self.zone_precedence = zone_precedence
        # 0 == evaluate every legal position exactly (small instances, tests).
        self.candidate_cap = int(candidate_cap)
        self.zones = self.inst.zones
        self.zone_arr = np.asarray(self.zones, dtype=object)
        self.b = self.inst.tw_end
        self.lam = obj.lam
        self.scale = obj.scale
        self.beh = obj.behavior_cost
        self.order: list[int] = list(order)
        self.removed: list[int] = [
            i for i in range(self.inst.n) if i not in set(self.order)
        ]
        self._rebuild()

    # ------------------------------------------------------------------ caches
    def _rebuild(self) -> None:
        o = np.asarray(self.order, dtype=int)
        frm, to = o[:-1], o[1:]
        legs = self.amb.service[:, frm] + self.amb.travel[:, frm, to]
        arr = np.zeros((self.amb.n_scenarios, len(o)), dtype=np.float64)
        if len(o) > 1:
            np.cumsum(legs, axis=1, out=arr[:, 1:])
        self.arrivals = arr

        wpos = np.flatnonzero(np.isfinite(self.b[o]))
        wpos = wpos[wpos > 0]
        self.window_pos = wpos                       # positions, ascending
        self.window_b = self.b[o[wpos]] if wpos.size else np.empty(0)
        if wpos.size:
            self.eta = self.ev.risk_array(
                arr[:, wpos], self.window_b, 2 * wpos - 1
            )
        else:
            self.eta = np.empty(0)
        self.risk_total = float(self.eta.sum())
        self.beh_total = float(self.beh[o[:-1], o[1:]].sum()) if len(o) > 1 else 0.0

    # --------------------------------------------------------------- objective
    @property
    def objective(self) -> float:
        return (
            self.lam * self.beh_total / self.scale
            + (1.0 - self.lam) * self.risk_total / self.scale
        )

    def copy(self) -> "RouteState":
        new = RouteState.__new__(RouteState)
        new.__dict__.update(self.__dict__)
        new.order = list(self.order)
        new.removed = list(self.removed)
        new.arrivals = self.arrivals.copy()
        new.eta = self.eta.copy()
        new.window_pos = self.window_pos.copy()
        new.window_b = self.window_b.copy()
        return new

    # ----------------------------------------------------------------- queries
    def feasible_positions(self, v: int) -> np.ndarray:
        """Insertion indices for ``v`` that keep every zone contiguous.

        If ``v``'s zone is already on the route, ``v`` may only go inside that
        zone's block or immediately after it.  If the zone is absent, ``v``
        starts a new block, which is only legal at a boundary between two
        existing blocks -- dropping it inside another zone's block would split
        that zone.
        """
        n = len(self.order)
        if not self.zone_precedence:
            return np.arange(1, n + 1)
        o = np.asarray(self.order[1:], dtype=int)
        same = np.flatnonzero(self.zone_arr[o] == self.zones[v]) + 1
        if same.size:
            return np.arange(same[0], same[-1] + 2)
        if n <= 1:
            return np.array([1])
        changes = np.flatnonzero(self.zone_arr[o[:-1]] != self.zone_arr[o[1:]]) + 2
        return np.unique(np.concatenate(([1], changes, [n])))

    def _shortlist(self, v: int, positions: np.ndarray, cap: int) -> np.ndarray:
        """Cheap pre-filter: the ``cap`` most promising positions for ``v``.

        Scored with a deterministic surrogate -- behaviour cost plus planned
        travel detour, no scenarios -- so it costs O(P) instead of O(P*W*S).
        Only the survivors get the full DRO evaluation.

        Needed when zone precedence is off, where every one of the ``n``
        positions is legal: the exact scan would be ``O(n*W*S)`` per stop per
        insertion.  It is a search restriction, not an approximation of the
        objective: whatever position is chosen is scored exactly.
        """
        o = np.asarray(self.order, dtype=int)
        n = len(self.order)
        prev = o[positions - 1]
        has_next = positions < n
        nxt = np.where(has_next, o[np.minimum(positions, n - 1)], 0)
        tt = self.inst.travel_time

        surrogate = self.beh[prev, v] + np.where(
            has_next, self.beh[v, nxt] - self.beh[prev, nxt], 0.0
        )
        detour = tt[prev, v] + np.where(
            has_next, tt[v, nxt] - tt[prev, nxt], 0.0
        )
        # Scale the detour onto the behaviour cost's rough magnitude so neither
        # term dominates the shortlist purely by units.
        scale = max(float(np.abs(self.beh).mean()), 1e-9) / max(
            float(np.abs(tt).mean()), 1e-9
        )
        score = self.lam * surrogate + (1.0 - self.lam) * detour * scale
        keep = np.argpartition(score, cap - 1)[:cap]
        return np.sort(positions[keep])

    # ------------------------------------------------- vectorised insertion cost
    def insertion_deltas(self, v: int, positions: np.ndarray | None = None):
        """Objective change for inserting ``v`` at each feasible position.

        Returns ``(deltas, positions)``.  One vectorised pass over all candidate
        positions and all window-carrying stops.
        """
        pos = self.feasible_positions(v) if positions is None else np.asarray(
            positions, dtype=int
        )
        if pos.size == 0:
            return np.empty(0), pos
        o = np.asarray(self.order, dtype=int)
        n = len(self.order)
        prev = o[pos - 1]
        has_next = pos < n
        nxt = np.where(has_next, o[np.minimum(pos, n - 1)], 0)

        d_beh = self.beh[prev, v] + np.where(
            has_next, self.beh[v, nxt] - self.beh[prev, nxt], 0.0
        )

        if self.lam >= 1.0:                       # pure imitation: skip the risk
            return self.lam * d_beh / self.scale, pos

        srv, trv = self.amb.service, self.amb.travel
        # (S, P): arrival at v for each candidate position
        arr_v = self.arrivals[:, pos - 1] + srv[:, prev] + trv[:, prev, v]
        d_risk = self.ev.risk_array(arr_v, self.b[v], 2 * pos - 1)

        if self.window_pos.size:
            # (S, P): scenario-wise delay imposed on everything after pos
            delay = np.where(
                has_next[None, :],
                trv[:, prev, v] + srv[:, v][:, None]
                + trv[:, v, nxt] - trv[:, prev, nxt],
                0.0,
            )
            wq = self.window_pos
            # (S, P, W) shifted arrivals of the window stops
            shifted = self.arrivals[:, None, wq] + delay[:, :, None]
            new_eta = self.ev.risk_array(
                shifted, self.window_b[None, :], (2 * (wq + 1) - 1)[None, :]
            )
            affected = wq[None, :] >= pos[:, None]
            d_risk = d_risk + np.where(
                affected, new_eta - self.eta[None, :], 0.0
            ).sum(axis=1)

        return (
            self.lam * d_beh / self.scale
            + (1.0 - self.lam) * d_risk / self.scale
        ), pos

    def insertion_deltas_many(
        self, stops: Sequence[int], max_cells: int = 8_000_000
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Score *every* (stop, feasible position) pair in one vectorised pass.

        Returns ``(deltas, positions, owners)`` flattened over candidates.  This
        is what keeps the repair operators off an ``O(q^2)`` ladder of small
        numpy calls: one scan of ``q`` removed stops is a handful of array
        operations instead of ``q`` separate evaluations.  ``max_cells`` chunks
        the ``(S, C, W)`` tensor so memory stays bounded on long routes.
        """
        stops = [int(s) for s in stops]
        if not stops:
            empty = np.empty(0)
            return empty, empty.astype(int), empty.astype(int)

        pos_list, owner_list = [], []
        for v in stops:
            p = self.feasible_positions(v)
            if p.size == 0:
                continue
            if self.candidate_cap and p.size > self.candidate_cap:
                p = self._shortlist(v, p, self.candidate_cap)
            pos_list.append(p)
            owner_list.append(np.full(p.size, v, dtype=int))
        if not pos_list:
            empty = np.empty(0)
            return empty, empty.astype(int), empty.astype(int)

        pos = np.concatenate(pos_list)
        owner = np.concatenate(owner_list)
        o = np.asarray(self.order, dtype=int)
        n = len(self.order)
        prev = o[pos - 1]
        has_next = pos < n
        nxt = np.where(has_next, o[np.minimum(pos, n - 1)], 0)

        d_beh = self.beh[prev, owner] + np.where(
            has_next, self.beh[owner, nxt] - self.beh[prev, nxt], 0.0
        )
        if self.lam >= 1.0:
            return self.lam * d_beh / self.scale, pos, owner

        srv, trv = self.amb.service, self.amb.travel
        s_n, w_n = self.amb.n_scenarios, max(self.window_pos.size, 1)
        chunk = max(1, int(max_cells // (s_n * w_n)))
        d_risk = np.empty(pos.size)

        for lo in range(0, pos.size, chunk):
            hi = min(lo + chunk, pos.size)
            sl = slice(lo, hi)
            p_c, v_c = pos[sl], owner[sl]
            prev_c, nxt_c, hn_c = prev[sl], nxt[sl], has_next[sl]

            arr_v = (
                self.arrivals[:, p_c - 1] + srv[:, prev_c] + trv[:, prev_c, v_c]
            )
            risk_v = self.ev.risk_array(arr_v, self.b[v_c], 2 * p_c - 1)

            if self.window_pos.size:
                delay = np.where(
                    hn_c[None, :],
                    trv[:, prev_c, v_c] + srv[:, v_c]
                    + trv[:, v_c, nxt_c] - trv[:, prev_c, nxt_c],
                    0.0,
                )
                wq = self.window_pos
                shifted = self.arrivals[:, None, wq] + delay[:, :, None]
                new_eta = self.ev.risk_array(
                    shifted, self.window_b[None, :], (2 * (wq + 1) - 1)[None, :]
                )
                affected = wq[None, :] >= p_c[:, None]
                risk_v = risk_v + np.where(
                    affected, new_eta - self.eta[None, :], 0.0
                ).sum(axis=1)
            d_risk[sl] = risk_v

        deltas = (
            self.lam * d_beh / self.scale
            + (1.0 - self.lam) * d_risk / self.scale
        )
        return deltas, pos, owner

    def insertion_delta(self, v: int, p: int) -> float:
        if p < 1 or p > len(self.order):
            return float("inf")
        d, _ = self.insertion_deltas(v, np.array([p]))
        return float(d[0])

    def best_insertion(self, v: int) -> tuple[float, int]:
        d, pos = self.insertion_deltas(v)
        if d.size == 0:
            return float("inf"), 1
        k = int(np.argmin(d))
        return float(d[k]), int(pos[k])

    def insertion_costs(self, v: int) -> list[tuple[float, int]]:
        d, pos = self.insertion_deltas(v)
        order = np.argsort(d, kind="stable")
        return [(float(d[k]), int(pos[k])) for k in order]

    # -------------------------------------------------- vectorised removal gain
    def removal_deltas(self) -> np.ndarray:
        """Objective change from deleting the stop at each position ``1..n-1``.

        Negative values are improvements, so the *gain* reported to the
        worst-removal operator is ``-delta``.
        """
        n = len(self.order)
        out = np.zeros(max(n - 1, 0))
        if n <= 1:
            return out
        o = np.asarray(self.order, dtype=int)
        srv, trv = self.amb.service, self.amb.travel
        for k, p in enumerate(range(1, n)):
            v, prev = o[p], o[p - 1]
            if p + 1 < n:
                nxt = o[p + 1]
                d_beh = -self.beh[prev, v] - self.beh[v, nxt] + self.beh[prev, nxt]
            else:
                nxt, d_beh = None, -self.beh[prev, v]
            if self.lam >= 1.0:
                out[k] = self.lam * d_beh / self.scale
                continue

            d_risk = 0.0
            wq = self.window_pos
            here = wq == p
            if here.any():
                d_risk -= float(self.eta[here].sum())
            after = wq > p
            if nxt is not None and after.any():
                delay = (
                    trv[:, prev, v] + srv[:, v] + trv[:, v, nxt] - trv[:, prev, nxt]
                )
                q = wq[after]
                shifted = self.arrivals[:, q] - delay[:, None]
                new_eta = self.ev.risk_array(
                    shifted, self.window_b[after], 2 * (q - 1) - 1
                )
                d_risk += float((new_eta - self.eta[after]).sum())
            out[k] = (
                self.lam * d_beh / self.scale
                + (1.0 - self.lam) * d_risk / self.scale
            )
        return out

    def removal_gain(self, p: int) -> float:
        """Objective *reduction* from deleting the stop at position ``p``."""
        if p < 1 or p >= len(self.order):
            return 0.0
        return -self.removal_deltas()[p - 1]

    def removal_gains(self) -> np.ndarray:
        return -self.removal_deltas()

    # ------------------------------------------------------------------ edits
    def insert(self, v: int, p: int) -> None:
        self.order.insert(p, v)
        if v in self.removed:
            self.removed.remove(v)
        self._rebuild()

    def remove_positions(self, positions: Iterable[int]) -> list[int]:
        pos = sorted({int(p) for p in positions if p >= 1}, reverse=True)
        taken = []
        for p in pos:
            if p < len(self.order):
                taken.append(self.order.pop(p))
        self.removed.extend(taken)
        self._rebuild()
        return taken

    def remove_stops(self, stops: Iterable[int]) -> list[int]:
        stops = set(int(s) for s in stops) - {DEPOT}
        pos = [p for p, i in enumerate(self.order) if i in stops]
        return self.remove_positions(pos)

    @property
    def is_complete(self) -> bool:
        return len(self.order) == self.inst.n and not self.removed

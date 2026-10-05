"""Baseline 1: naive shortest-time TSP (§7).

Nearest neighbour plus 2-opt on the travel-time matrix.  No behaviour term, no
windows, no zones -- the "what would a plain distance minimiser do" reference
point that every other method has to beat on reliability and on ERP.
"""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from core.types import DEPOT, Instance, Solution


def nearest_neighbour(inst: Instance) -> list[int]:
    tt = inst.travel_time
    unvisited = set(range(inst.n)) - {DEPOT}
    order, cur = [DEPOT], DEPOT
    while unvisited:
        nxt = min(unvisited, key=lambda j: tt[cur, j])
        order.append(nxt)
        unvisited.discard(nxt)
        cur = nxt
    return order


def tour_time(inst: Instance, order: Sequence[int]) -> float:
    tt = inst.travel_time
    order = list(order)
    return float(sum(tt[a, b] for a, b in zip(order, order[1:] + [order[0]])))


def two_opt(inst: Instance, order: Sequence[int], max_passes: int = 50) -> list[int]:
    """Standard 2-opt on the closed tour, first-improvement, depot pinned."""
    tt = inst.travel_time
    order = list(order)
    n = len(order)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = order[i - 1], order[i]
                c = order[j]
                d = order[(j + 1) % n]
                delta = (tt[a, c] + tt[b, d]) - (tt[a, b] + tt[c, d])
                if delta < -1e-9:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    improved = True
    return order


def solve_naive_tsp(inst: Instance, use_two_opt: bool = True) -> Solution:
    t0 = time.perf_counter()
    order = nearest_neighbour(inst)
    if use_two_opt:
        order = two_opt(inst, order)
    rt = time.perf_counter() - t0
    return Solution(
        order=order,
        objective=tour_time(inst, order),
        solver="naive_tsp",
        runtime_s=rt,
        meta={"tour_time": tour_time(inst, order)},
    )


def zone_respecting_naive_tsp(inst: Instance) -> Solution:
    """Same idea, but zones stay contiguous -- the fair naive comparator.

    Without this variant the naive baseline also violates the zone-precedence
    constraint, which would confound "no behaviour model" with "no zone
    structure" when reading the ablation table.
    """
    t0 = time.perf_counter()
    tt = inst.travel_time
    groups = inst.zone_groups()
    remaining = dict(groups)
    order, cur = [DEPOT], DEPOT
    while remaining:
        z = min(remaining, key=lambda zz: float(np.min(tt[cur, remaining[zz]])))
        members = list(remaining.pop(z))
        while members:
            nxt = min(members, key=lambda j: tt[cur, j])
            members.remove(nxt)
            order.append(nxt)
            cur = nxt
    rt = time.perf_counter() - t0
    return Solution(order=order, objective=tour_time(inst, order),
                    solver="naive_tsp_zoned", runtime_s=rt)

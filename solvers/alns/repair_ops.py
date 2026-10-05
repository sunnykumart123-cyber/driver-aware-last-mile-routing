"""Repair operators (§5b: greedy and regret-k).

Both score candidate positions with the combined
``lambda * c^beh + (1 - lambda) * delta eta`` cost through
:meth:`RouteState.insertion_deltas_many` -- the incremental evaluator, never a
re-solve of the DRO model.

### Why the scan is batched

The reliability term couples globally: inserting a stop shifts the arrival time
of every later stop, so in principle each insertion invalidates the cost of
every other candidate. A textbook greedy repair therefore rescans all ``q``
removed stops after each insertion, which is ``O(q^2)`` risk evaluations and is
the single most expensive thing in the search at 200+ stops.

Instead each scan scores *all* (stop, position) pairs in one vectorised pass and
then applies up to ``refresh_every`` insertions before rescanning, adjusting
positions for the insertions already applied and re-checking zone feasibility
each time. The costs of the later insertions in a batch are therefore slightly
stale.

That is a deliberate heuristic choice and it is safe in the way that matters:
the *reported* objective is never the incremental estimate -- ``RouteState``
recomputes the exact objective after every insertion, and the ALNS asserts
agreement with a from-scratch evaluation in
``tests/test_alns_vs_exact.py``. Staleness can only cost search quality, never
correctness of a reported number. Set ``refresh_every=1`` for the exact
textbook behaviour (and the tests cover both settings).
"""
from __future__ import annotations

import numpy as np

from solvers.alns.state import RouteState


def _apply_batch(
    state: RouteState,
    deltas: np.ndarray,
    positions: np.ndarray,
    owners: np.ndarray,
    rank: np.ndarray,
    limit: int,
) -> int:
    """Apply up to ``limit`` insertions from one scored scan.

    ``rank`` is the order in which candidates are considered.  Positions are
    shifted for insertions already applied in this batch, and every candidate is
    re-validated against the current zone blocks before use.
    """
    applied = 0
    done: set[int] = set()
    shifts: list[int] = []
    for k in rank:
        v, p = int(owners[k]), int(positions[k])
        if v in done:
            continue
        for q in shifts:                      # in application order
            if q <= p:
                p += 1
        if p not in set(state.feasible_positions(v).tolist()):
            continue
        state.insert(v, p)
        shifts.append(p)
        done.add(v)
        applied += 1
        if applied >= limit:
            break
    return applied


def _force_one(state: RouteState) -> None:
    """Last resort: park a stop at the end of its own zone block."""
    v = state.removed[0]
    pos = state.feasible_positions(v)
    state.insert(v, int(pos[-1]) if pos.size else len(state.order))


def greedy_insertion(state: RouteState, rng: np.random.Generator,
                     noise: float = 0.0, refresh_every: int = 8) -> RouteState:
    """Repeatedly perform the cheapest (stop, position) insertions."""
    while state.removed:
        d, pos, owner = state.insertion_deltas_many(state.removed)
        if d.size == 0:
            _force_one(state)
            continue
        if noise:
            d = d * (1.0 + noise * (rng.random(d.shape) - 0.5))
        rank = np.argsort(d, kind="stable")
        if _apply_batch(state, d, pos, owner, rank, refresh_every) == 0:
            _force_one(state)
    return state


def regret_k_insertion(state: RouteState, rng: np.random.Generator,
                       k: int = 3, refresh_every: int = 8) -> RouteState:
    """Insert the stops with the largest regret between their best ``k`` slots.

    Regret is ``sum_{m=2..k} (cost_m - cost_1)``: a stop whose second-best option
    is much worse than its best goes in first, because delaying it risks losing
    its one good slot.
    """
    while state.removed:
        d, pos, owner = state.insertion_deltas_many(state.removed)
        if d.size == 0:
            _force_one(state)
            continue

        best_cost: dict[int, float] = {}
        best_pos: dict[int, int] = {}
        regret: dict[int, float] = {}
        for v in np.unique(owner):
            sel = np.flatnonzero(owner == v)
            costs = np.sort(d[sel])
            best_cost[int(v)] = float(costs[0])
            best_pos[int(v)] = int(pos[sel[int(np.argmin(d[sel]))]])
            regret[int(v)] = float(np.sum(costs[1:k] - costs[0])) if costs.size > 1 \
                else float(costs.size)

        stops = sorted(regret, key=lambda v: (-regret[v], best_cost[v]))
        cand_pos = np.array([best_pos[v] for v in stops], dtype=int)
        cand_owner = np.array(stops, dtype=int)
        cand_d = np.array([best_cost[v] for v in stops], dtype=float)
        rank = np.arange(len(stops))
        if _apply_batch(state, cand_d, cand_pos, cand_owner, rank,
                        refresh_every) == 0:
            _force_one(state)
    return state


def make_regret(k: int, refresh_every: int = 8):
    def _op(state: RouteState, rng: np.random.Generator) -> RouteState:
        return regret_k_insertion(state, rng, k=k, refresh_every=refresh_every)

    _op.__name__ = f"regret_{k}_insertion"
    return _op


def make_greedy(refresh_every: int = 8, noise: float = 0.0):
    def _op(state: RouteState, rng: np.random.Generator) -> RouteState:
        return greedy_insertion(state, rng, noise=noise,
                                refresh_every=refresh_every)

    _op.__name__ = "greedy_insertion"
    return _op


REPAIR_OPERATORS = {
    "greedy_insertion": greedy_insertion,
    "regret_k_insertion": regret_k_insertion,
}

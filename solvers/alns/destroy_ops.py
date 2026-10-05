"""Destroy operators (§5b: four of them).

Each operator takes a complete :class:`~solvers.alns.state.RouteState`, removes
``q`` stops in place and returns the removed stop indices.  The signature is
uniform so :class:`~solvers.alns.alns_solver.ALNS` can select among them by
adaptive weight.
"""
from __future__ import annotations

import numpy as np

from solvers.alns.state import RouteState


def random_removal(state: RouteState, q: int, rng: np.random.Generator) -> list[int]:
    """Uniformly random stops: the diversification workhorse."""
    positions = np.arange(1, len(state.order))
    if len(positions) == 0:
        return []
    q = int(min(q, len(positions)))
    chosen = rng.choice(positions, size=q, replace=False)
    return state.remove_positions(chosen.tolist())


def worst_removal(state: RouteState, q: int, rng: np.random.Generator,
                  determinism: float = 3.0) -> list[int]:
    """Remove the stops whose deletion reduces the *combined* objective most.

    The gain is measured on the lambda-weighted objective, so at low lambda this
    targets the stops that drive ``eta`` and at high lambda the ones sitting on
    behaviourally implausible arcs.  Randomised with the usual Ropke-Pisinger
    ``y^determinism`` rule so the operator is not fully greedy.
    """
    if len(state.order) <= 1:
        return []
    gains = state.removal_gains()               # one vectorised pass
    pool = [int(p) for p in (np.argsort(-gains) + 1)]
    picked: list[int] = []
    q = int(min(q, len(pool)))
    for _ in range(q):
        y = rng.random() ** determinism
        k = int(y * len(pool))
        picked.append(pool.pop(min(k, len(pool) - 1)))
    return state.remove_stops([state.order[p] for p in sorted(picked)])


def zone_cluster_removal(state: RouteState, q: int, rng: np.random.Generator) -> list[int]:
    """Remove whole sort zones.

    Zone blocks are the unit the drivers themselves work in, so this operator is
    the one that can reorder the zone sequence -- with zone precedence switched
    on, no other operator can.
    """
    zones = [state.zones[i] for i in state.order[1:]]
    if not zones:
        return []
    uniq = list(dict.fromkeys(zones))
    uniq = [uniq[k] for k in rng.permutation(len(uniq))]
    target: list[int] = []
    for z in uniq:
        members = [i for i in state.order[1:] if state.zones[i] == z]
        target.extend(members)
        if len(target) >= q:
            break
    return state.remove_stops(target)


def shaw_removal(state: RouteState, q: int, rng: np.random.Generator,
                 w_travel: float = 1.0, w_zone: float = 0.6,
                 w_window: float = 0.3, determinism: float = 4.0) -> list[int]:
    """Shaw / related removal: a seed stop plus its most related neighbours.

    Relatedness mixes normalised travel time, zone identity and time-window
    similarity, i.e. exactly the dimensions along which the behaviour term and
    the reliability term disagree.
    """
    interior = state.order[1:]
    if not interior:
        return []
    tt = state.inst.travel_time
    tmax = float(tt.max()) or 1.0
    b = state.b
    fin = np.isfinite(b)
    bmax = float(b[fin].max()) if fin.any() else 1.0

    seed = int(rng.choice(interior))
    picked = [seed]
    q = int(min(q, len(interior)))
    while len(picked) < q:
        ref = picked[int(rng.integers(len(picked)))]
        pool = [i for i in interior if i not in picked]
        if not pool:
            break

        def relatedness(j: int) -> float:
            r = w_travel * tt[ref, j] / tmax
            r += w_zone * (0.0 if state.zones[j] == state.zones[ref] else 1.0)
            if np.isfinite(b[ref]) and np.isfinite(b[j]):
                r += w_window * abs(b[ref] - b[j]) / max(bmax, 1.0)
            return r

        pool.sort(key=relatedness)
        y = rng.random() ** determinism
        picked.append(pool[min(int(y * len(pool)), len(pool) - 1)])
    return state.remove_stops(picked)


DESTROY_OPERATORS = {
    "random_removal": random_removal,
    "worst_removal": worst_removal,
    "zone_cluster_removal": zone_cluster_removal,
    "shaw_removal": shaw_removal,
}

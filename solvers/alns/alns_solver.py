"""The production solver: adaptive large neighbourhood search (§5b).

Loop
----
1. pick a destroy and a repair operator by adaptive weight (Ropke & Pisinger);
2. remove ``q`` stops, reinsert them all;
3. accept or reject by simulated annealing;
4. every ``segment`` iterations, update operator weights from the scores earned.

Scores follow the standard convention: ``sigma1`` for a new global best,
``sigma2`` for an improvement on the incumbent, ``sigma3`` for an accepted
non-improving move.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from core.config import SolverConfig
from core.logging_utils import get_logger
from core.objective import CombinedObjective
from core.types import DEPOT, Instance, Solution
from solvers.alns.acceptance import AcceptanceCriterion, build_acceptance
from solvers.alns.destroy_ops import (
    random_removal,
    shaw_removal,
    worst_removal,
    zone_cluster_removal,
)
from solvers.alns.repair_ops import make_greedy, make_regret
from solvers.alns.state import RouteState

log = get_logger(__name__)

SIGMA1, SIGMA2, SIGMA3 = 33.0, 9.0, 13.0
REACTION = 0.3          # weight decay r


@dataclass
class _OperatorBank:
    names: list[str]
    ops: list[Callable]
    weights: np.ndarray = field(init=False)
    scores: np.ndarray = field(init=False)
    counts: np.ndarray = field(init=False)
    usage: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        k = len(self.ops)
        self.weights = np.ones(k)
        self.scores = np.zeros(k)
        self.counts = np.zeros(k)
        self.usage = np.zeros(k, dtype=int)

    def pick(self, rng: np.random.Generator) -> int:
        p = self.weights / self.weights.sum()
        k = int(rng.choice(len(self.ops), p=p))
        self.counts[k] += 1
        self.usage[k] += 1
        return k

    def reward(self, k: int, score: float) -> None:
        self.scores[k] += score

    def update(self) -> None:
        used = self.counts > 0
        self.weights[used] = (
            (1 - REACTION) * self.weights[used]
            + REACTION * self.scores[used] / self.counts[used]
        )
        self.weights = np.clip(self.weights, 1e-3, None)
        self.scores[:] = 0.0
        self.counts[:] = 0.0

    def report(self) -> dict[str, dict[str, float]]:
        return {
            n: {"weight": float(w), "uses": int(u)}
            for n, w, u in zip(self.names, self.weights, self.usage)
        }


@dataclass
class ALNSResult:
    solution: Solution
    history: list[float]
    operator_stats: dict
    iterations: int
    runtime_s: float


# ------------------------------------------------------------ initial solution
def zone_nearest_neighbour(inst: Instance, obj: CombinedObjective | None = None,
                           zone_precedence: bool = True) -> list[int]:
    """Cheap feasible start: nearest-neighbour, zone-block respecting.

    Zones are entered in order of their nearest stop to the current position and
    completed before moving on, which is both what the drivers do and what the
    zone-precedence constraint requires.
    """
    tt = inst.travel_time
    groups = inst.zone_groups()
    remaining_zones = dict(groups)
    order = [DEPOT]
    cur = DEPOT
    while remaining_zones:
        z = min(
            remaining_zones,
            key=lambda zz: float(np.min(tt[cur, remaining_zones[zz]])),
        )
        members = list(remaining_zones.pop(z))
        while members:
            nxt = min(members, key=lambda j: tt[cur, j])
            members.remove(nxt)
            order.append(nxt)
            cur = nxt
    if not zone_precedence:
        return order
    return order


def solve_alns(
    inst: Instance,
    obj: CombinedObjective,
    cfg: SolverConfig | None = None,
    initial: Sequence[int] | None = None,
    acceptance: AcceptanceCriterion | None = None,
    seed: int | None = None,
) -> ALNSResult:
    cfg = cfg or SolverConfig()
    rng = np.random.default_rng(cfg.seed if seed is None else seed)

    start_order = list(initial) if initial is not None else zone_nearest_neighbour(
        inst, obj, cfg.zone_precedence
    )
    state = RouteState(obj, start_order, cfg.zone_precedence,
                       candidate_cap=cfg.candidate_cap)
    best = state.copy()
    best_val = cur_val = state.objective
    accept = acceptance or build_acceptance(cur_val, cfg)

    destroy = _OperatorBank(
        names=["random_removal", "worst_removal", "zone_cluster_removal", "shaw_removal"],
        ops=[random_removal, worst_removal, zone_cluster_removal, shaw_removal],
    )
    repair = _OperatorBank(
        names=["greedy_insertion", f"regret_{cfg.regret_k}_insertion"],
        ops=[
            make_greedy(cfg.repair_refresh_every),
            make_regret(cfg.regret_k, cfg.repair_refresh_every),
        ],
    )

    n_removable = max(inst.n - 1, 1)
    q_cap = min(cfg.destroy_max_stops, n_removable)
    q_lo = max(1, min(int(cfg.destroy_min_frac * n_removable), q_cap))
    q_hi = max(q_lo + 1, min(int(cfg.destroy_max_frac * n_removable), q_cap))

    history = [cur_val]
    t0 = time.perf_counter()
    it = 0
    while it < cfg.alns_iterations:
        if time.perf_counter() - t0 > cfg.time_limit_s:
            log.debug("ALNS stopped on the time limit after %d iterations", it)
            break
        it += 1
        d_idx = destroy.pick(rng)
        r_idx = repair.pick(rng)
        q = int(rng.integers(q_lo, q_hi + 1))

        cand = state.copy()
        destroy.ops[d_idx](cand, q, rng)
        repair.ops[r_idx](cand, rng)
        cand_val = cand.objective

        score = 0.0
        if cand_val < best_val - 1e-12:
            best, best_val = cand.copy(), cand_val
            score = SIGMA1
        elif cand_val < cur_val - 1e-12:
            score = SIGMA2

        if accept.accept(cur_val, cand_val, rng):
            if score == 0.0:
                score = SIGMA3
            state, cur_val = cand, cand_val
        accept.step()

        destroy.reward(d_idx, score)
        repair.reward(r_idx, score)
        if it % cfg.alns_segment == 0:
            destroy.update()
            repair.update()
        history.append(cur_val)

    rt = time.perf_counter() - t0
    # to_solution re-scores the route from scratch through the reliability
    # evaluator; it must agree with the incremental bookkeeping (tests assert it).
    sol = obj.to_solution(
        best.order, "alns", rt,
        iterations=it,
        initial_objective=history[0],
        incremental_objective=best_val,
    )
    stats = {"destroy": destroy.report(), "repair": repair.report()}
    log.info(
        "ALNS %s: %.4f -> %.4f in %d iterations (%.1fs)",
        inst.route_id, history[0], best_val, it, rt,
    )
    return ALNSResult(sol, history, stats, it, rt)

"""Logic-based Benders decomposition -- the secondary validation solver (§5c).

Decomposition
-------------
**Master** (MILP): the sequencing problem -- assignment, MTZ, zone precedence --
with the behaviour cost in full and the whole reliability term collapsed into one
continuous variable ``zeta >= 0``.  The master therefore starts as a pure
behaviour problem and is progressively informed about risk.

**Subproblem** (no optimisation): given the master's sequence, the DRO layer is
evaluated exactly by :class:`~solvers.reliability.ReliabilityEvaluator`.

**Cut.**  Let ``p`` be the earliest position whose surplus ``eta_p`` is positive
and let ``A_p`` be the arcs of the depot-rooted prefix ending at ``p``.  The
arrival distribution at ``p`` is a function of exactly those arcs, so *any*
feasible solution containing all of ``A_p`` has surplus at least ``eta_p`` at
that stop.  That gives the valid cut

    zeta >= w * eta_p * ( 1 - sum_{(i,j) in A_p} (1 - x_ij) ),   w = (1-lambda)/scale

which is far stronger than a no-good cut on the whole tour because ``|A_p|`` is
usually a small prefix.  When the incumbent is already reliable
(``eta == 0`` everywhere) the cut is generated from the total risk of the tour
instead, which closes off that sequence.

Termination: the master objective is a valid lower bound at every iteration and
the best evaluated sequence a valid upper bound, so the loop reports a certified
gap whenever it stops early.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from core.config import SolverConfig
from core.logging_utils import get_logger
from core.objective import CombinedObjective
from core.types import DEPOT, Instance, Solution
from solvers.exact_milp import _order_from_x

log = get_logger(__name__)


@dataclass
class BendersResult:
    solution: Solution
    lower_bound: float
    upper_bound: float
    iterations: int
    n_cuts: int
    runtime_s: float
    status: str
    history: list[dict] = field(default_factory=list)

    @property
    def gap(self) -> float:
        if not np.isfinite(self.upper_bound):
            return float("nan")
        denom = max(abs(self.upper_bound), 1e-9)
        return (self.upper_bound - self.lower_bound) / denom


def _prefix_arcs(order: Sequence[int], position: int) -> list[tuple[int, int]]:
    return [(order[k], order[k + 1]) for k in range(position)]


def solve_benders(
    inst: Instance,
    obj: CombinedObjective,
    cfg: SolverConfig | None = None,
    max_iterations: int = 200,
    initial: Sequence[int] | None = None,
    tol: float = 1e-6,
) -> BendersResult:
    import pulp

    cfg = cfg or SolverConfig()
    n = inst.n
    ev = obj.evaluator
    w = (1.0 - obj.lam) / obj.scale
    arcs = [(i, j) for i in range(n) for j in range(n) if i != j]

    master = pulp.LpProblem("benders_master", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", arcs, cat="Binary")
    u = pulp.LpVariable.dicts("u", range(n), lowBound=0, upBound=n - 1, cat="Integer")
    zeta = pulp.LpVariable("zeta", lowBound=0)

    for i in range(n):
        master += pulp.lpSum(x[(i, j)] for j in range(n) if j != i) == 1
        master += pulp.lpSum(x[(j, i)] for j in range(n) if j != i) == 1
    master += u[DEPOT] == 0
    for i in range(1, n):
        master += u[i] >= 1
    for i, j in arcs:
        if i != DEPOT and j != DEPOT:
            master += u[i] - u[j] + n * x[(i, j)] <= n - 1

    if cfg.zone_precedence:
        groups = inst.zone_groups()
        zs = list(groups)
        y = pulp.LpVariable.dicts(
            "y", [(a, c) for a in zs for c in zs if a != c], cat="Binary"
        )
        for a in zs:
            for c in zs:
                if a < c:
                    master += y[(a, c)] + y[(c, a)] == 1
        for a in zs:
            for c in zs:
                for e in zs:
                    if len({a, c, e}) == 3:
                        master += y[(a, c)] + y[(c, e)] - y[(a, e)] <= 1
        for a in zs:
            off = pulp.lpSum(len(groups[c]) * y[(c, a)] for c in zs if c != a)
            for i in groups[a]:
                master += u[i] >= off + 1
                master += u[i] <= off + len(groups[a])

    beh = obj.behavior_cost
    master += (
        (obj.lam / obj.scale)
        * pulp.lpSum(beh[i, j] * x[(i, j)] for i, j in arcs if j != DEPOT)
        + zeta
    )

    best_order = list(initial) if initial is not None else None
    ub = obj(best_order) if best_order is not None else float("inf")
    lb = -float("inf")
    n_cuts = 0
    history: list[dict] = []
    status = "IterationLimit"
    t0 = time.perf_counter()

    for it in range(1, max_iterations + 1):
        remaining = cfg.time_limit_s - (time.perf_counter() - t0)
        if remaining <= 0:
            status = "TimeLimit"
            break
        res = master.solve(
            pulp.PULP_CBC_CMD(msg=0, timeLimit=int(max(remaining, 1)))
        )
        name = pulp.LpStatus[res]
        if name != "Optimal":
            status = f"Master{name}"
            break

        lb = max(lb, float(pulp.value(master.objective)))
        order = _order_from_x(x, n)
        terms = obj.terms(order, full_report=True)
        if terms.total < ub - 1e-12:
            ub, best_order = terms.total, list(order)

        zeta_val = float(pulp.value(zeta))
        true_risk = w * terms.report.eta.sum()
        history.append(
            {"iteration": it, "lb": lb, "ub": ub, "zeta": zeta_val,
             "true_risk": true_risk}
        )
        if true_risk <= zeta_val + tol:
            status = "Optimal"
            break

        eta = terms.report.eta
        pos = [p for p in range(len(order)) if eta[p] > tol]
        if pos:
            p = pos[0]
            rhs_value = w * float(eta[p])
            cut_arcs = _prefix_arcs(order, p)
        else:  # pragma: no cover - only if zeta is under-set with zero risk
            rhs_value = true_risk
            cut_arcs = _prefix_arcs(order, len(order) - 1)
        master += zeta >= rhs_value * (
            1 - pulp.lpSum(1 - x[(i, j)] for i, j in cut_arcs)
        )
        n_cuts += 1

    rt = time.perf_counter() - t0
    if best_order is None:
        best_order = list(range(n))
    sol = obj.to_solution(best_order, "benders", rt, lower_bound=lb,
                          n_cuts=n_cuts, benders_status=status)
    log.info(
        "Benders %s: lb=%.4f ub=%.4f cuts=%d status=%s (%.1fs)",
        inst.route_id, lb, ub, n_cuts, status, rt,
    )
    return BendersResult(sol, lb, ub, len(history), n_cuts, rt, status, history)

"""Exact formulations -- validation only (§5a).

Three backends behind one entry point:

``gurobi``      the MILP of §3 through ``gurobipy`` (preferred when licensed);
``pulp``        the same MILP through PuLP/CBC (open-source, always available);
``bruteforce``  complete enumeration of feasible sequences, scored by the *same*
                :class:`~core.objective.CombinedObjective` the heuristics use.

The brute-force backend is what makes the unit tests meaningful: it is
independent of the MILP encoding, so ``tests/test_milp_toy_cases.py`` compares
two independent computations of the optimum rather than a model against itself.

Scenario-based reformulation of the DRO constraint
--------------------------------------------------
With ``P_hat`` supported on ``S`` scenarios, the worst-case CVaR constraint of
§3 is exactly

    eta_i >= tau_i + (1/(eps*S)) * sum_s nu_i^s + (delta/eps) * ||w_i||_*
    nu_i^s >= T_i^s - b_i - tau_i,    nu_i^s >= 0,    eta_i >= 0

with ``||w_i||_1 = 2*u_i - 1`` for the ``linf`` ground norm, which is linear in
the MTZ position variable.  The ``l2`` ground norm makes the inflation
``sqrt(2u_i - 1)`` and is therefore not exactly MILP-representable; it is
supported by the heuristic solvers and rejected here.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from core.config import SolverConfig
from core.logging_utils import get_logger
from core.objective import CombinedObjective
from core.types import DEPOT, Instance, Solution

log = get_logger(__name__)


@dataclass
class MilpResult:
    solution: Solution
    status: str
    objective: float
    lower_bound: float
    runtime_s: float
    backend: str
    n_vars: int = 0
    n_cons: int = 0
    log: dict = field(default_factory=dict)


# --------------------------------------------------------------------- helpers
def _scenario_arrays(obj: CombinedObjective, max_scenarios: int | None):
    amb = obj.evaluator.amb
    travel, service = amb.travel, amb.service
    s = travel.shape[0]
    if max_scenarios is not None and s > max_scenarios:
        idx = np.linspace(0, s - 1, max_scenarios).round().astype(int)
        travel, service = travel[idx], service[idx]
        log.info("exact MILP: sub-sampling P_hat from %d to %d scenarios", s, len(idx))
    return np.asarray(travel, dtype=float), np.asarray(service, dtype=float)


def _big_m(travel: np.ndarray, service: np.ndarray) -> float:
    return float(travel.max() * travel.shape[1] + service.max() * service.shape[1] + 1.0)


def _dual_weight_linear(amb, u_expr, one):
    """``||w||_*`` as a linear expression in the MTZ position variable."""
    if amb.ground_norm == "linf":
        return 2.0 * u_expr - one
    if amb.ground_norm == "l1":
        return one
    raise ValueError(
        f"ground_norm {amb.ground_norm!r} is not MILP-representable; "
        "use 'linf' or 'l1' for the exact solver"
    )


def _feasible_orders(inst: Instance, zone_precedence: bool):
    """All sequences the model allows, as an iterator over tuples."""
    stops = [i for i in range(inst.n) if i != DEPOT]
    if not zone_precedence:
        for perm in itertools.permutations(stops):
            yield (DEPOT,) + perm
        return
    groups = inst.zone_groups()
    zone_names = list(groups)
    for zorder in itertools.permutations(zone_names):
        within = [itertools.permutations(groups[z]) for z in zorder]
        for combo in itertools.product(*within):
            yield (DEPOT,) + tuple(i for block in combo for i in block)


def count_feasible_orders(inst: Instance, zone_precedence: bool) -> int:
    import math

    if not zone_precedence:
        return math.factorial(inst.n - 1)
    groups = inst.zone_groups()
    total = math.factorial(len(groups))
    for v in groups.values():
        total *= math.factorial(len(v))
    return total


# ---------------------------------------------------------------- brute force
def solve_bruteforce(
    inst: Instance,
    obj: CombinedObjective,
    cfg: SolverConfig | None = None,
    max_orders: int = 400_000,
) -> MilpResult:
    cfg = cfg or SolverConfig()
    n_orders = count_feasible_orders(inst, cfg.zone_precedence)
    if n_orders > max_orders:
        raise ValueError(
            f"{n_orders} feasible sequences exceeds max_orders={max_orders}; "
            "brute force is for toy validation only"
        )
    t0 = time.perf_counter()
    best_order, best_val = None, float("inf")
    for order in _feasible_orders(inst, cfg.zone_precedence):
        val = obj(order)
        if val < best_val - 1e-12:
            best_order, best_val = list(order), val
    rt = time.perf_counter() - t0
    sol = obj.to_solution(best_order, "bruteforce", rt, lower_bound=best_val,
                          n_orders=n_orders)
    return MilpResult(sol, "Optimal", best_val, best_val, rt, "bruteforce",
                      log={"n_orders": n_orders})


# ---------------------------------------------------------------------- PuLP
def _solve_pulp(
    inst: Instance,
    obj: CombinedObjective,
    cfg: SolverConfig,
    max_scenarios: int | None,
) -> MilpResult:
    import pulp

    amb = obj.evaluator.amb
    travel, service = _scenario_arrays(obj, max_scenarios)
    s_n = travel.shape[0]
    n = inst.n
    b = inst.tw_end
    eps = amb.eps
    scale = obj._scale
    m = _big_m(travel, service)
    arcs = [(i, j) for i in range(n) for j in range(n) if i != j]

    prob = pulp.LpProblem("mtp_dro_tsptw", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", arcs, cat="Binary")
    u = pulp.LpVariable.dicts("u", range(n), lowBound=0, upBound=n - 1, cat="Integer")
    t = pulp.LpVariable.dicts(
        "T", [(i, s) for i in range(n) for s in range(s_n)], lowBound=0
    )
    with_win = [i for i in range(n) if np.isfinite(b[i])]
    tau = pulp.LpVariable.dicts("tau", with_win, cat="Continuous")
    nu = pulp.LpVariable.dicts(
        "nu", [(i, s) for i in with_win for s in range(s_n)], lowBound=0
    )
    eta = pulp.LpVariable.dicts("eta", with_win, lowBound=0)

    # assignment
    for i in range(n):
        prob += pulp.lpSum(x[(i, j)] for j in range(n) if j != i) == 1
        prob += pulp.lpSum(x[(j, i)] for j in range(n) if j != i) == 1

    # MTZ subtour elimination
    prob += u[DEPOT] == 0
    for i in range(1, n):
        prob += u[i] >= 1
    for i, j in arcs:
        if i != DEPOT and j != DEPOT:
            prob += u[i] - u[j] + n * x[(i, j)] <= n - 1

    # zone precedence: every zone occupies one contiguous block of positions
    if cfg.zone_precedence:
        groups = inst.zone_groups()
        zs = list(groups)
        y = pulp.LpVariable.dicts(
            "y", [(a, c) for a in zs for c in zs if a != c], cat="Binary"
        )
        for a in zs:
            for c in zs:
                if a >= c:
                    continue
                prob += y[(a, c)] + y[(c, a)] == 1
        for a in zs:
            for c in zs:
                for e in zs:
                    if len({a, c, e}) == 3:
                        prob += y[(a, c)] + y[(c, e)] - y[(a, e)] <= 1
        for a in zs:
            off = pulp.lpSum(len(groups[c]) * y[(c, a)] for c in zs if c != a)
            for i in groups[a]:
                prob += u[i] >= off + 1
                prob += u[i] <= off + len(groups[a])

    # arrival-time propagation, one copy per scenario (depot excluded as a head)
    for s in range(s_n):
        prob += t[(DEPOT, s)] == 0
        for i, j in arcs:
            if j == DEPOT:
                continue
            prob += t[(j, s)] >= (
                t[(i, s)] + service[s, i] + travel[s, i, j] - m * (1 - x[(i, j)])
            )

    # CVaR / DRO reliability block
    for i in with_win:
        for s in range(s_n):
            prob += nu[(i, s)] >= t[(i, s)] - float(b[i]) - tau[i]
        infl = _dual_weight_linear(amb, u[i], 1.0)
        prob += eta[i] >= (
            tau[i]
            + (1.0 / (eps * s_n)) * pulp.lpSum(nu[(i, s)] for s in range(s_n))
            + (amb.delta / eps) * infl
        )

    beh = obj.behavior_cost
    prob += (
        (obj.lam / scale)
        * pulp.lpSum(beh[i, j] * x[(i, j)] for i, j in arcs if j != DEPOT)
        + ((1.0 - obj.lam) / scale) * pulp.lpSum(eta[i] for i in with_win)
    )

    t0 = time.perf_counter()
    solver = pulp.PULP_CBC_CMD(
        msg=0, timeLimit=int(max(cfg.time_limit_s, 1)), gapRel=cfg.mip_gap
    )
    status = prob.solve(solver)
    rt = time.perf_counter() - t0
    status_name = pulp.LpStatus[status]

    if status_name not in ("Optimal",):
        log.warning("PuLP status %s on %s", status_name, inst.route_id)
        empty = Solution(order=list(range(n)), solver="exact_pulp", runtime_s=rt)
        return MilpResult(empty, status_name, float("nan"), float("nan"), rt, "pulp")

    order = _order_from_x(x, n)
    mip_obj = float(pulp.value(prob.objective))
    sol = obj.to_solution(order, "exact_pulp", rt, lower_bound=mip_obj,
                          milp_objective=mip_obj, n_scenarios=s_n)
    return MilpResult(
        sol, status_name, mip_obj, mip_obj, rt, "pulp",
        n_vars=len(prob.variables()), n_cons=len(prob.constraints),
    )


def _order_from_x(x, n: int) -> list[int]:
    import pulp

    nxt = {}
    for (i, j), var in x.items():
        if pulp.value(var) is not None and pulp.value(var) > 0.5:
            nxt[i] = j
    order, cur = [DEPOT], DEPOT
    for _ in range(n - 1):
        cur = nxt[cur]
        order.append(cur)
    if sorted(order) != list(range(n)):
        raise RuntimeError("MILP solution does not form a single Hamiltonian tour")
    return order


# -------------------------------------------------------------------- Gurobi
def _solve_gurobi(
    inst: Instance,
    obj: CombinedObjective,
    cfg: SolverConfig,
    max_scenarios: int | None,
) -> MilpResult:
    import gurobipy as gp
    from gurobipy import GRB

    amb = obj.evaluator.amb
    travel, service = _scenario_arrays(obj, max_scenarios)
    s_n = travel.shape[0]
    n = inst.n
    b = inst.tw_end
    eps = amb.eps
    scale = obj._scale
    m = _big_m(travel, service)
    arcs = [(i, j) for i in range(n) for j in range(n) if i != j]
    with_win = [i for i in range(n) if np.isfinite(b[i])]

    mdl = gp.Model("mtp_dro_tsptw")
    mdl.Params.OutputFlag = 0
    mdl.Params.TimeLimit = cfg.time_limit_s
    mdl.Params.MIPGap = cfg.mip_gap

    x = mdl.addVars(arcs, vtype=GRB.BINARY, name="x")
    u = mdl.addVars(range(n), lb=0, ub=n - 1, vtype=GRB.INTEGER, name="u")
    t = mdl.addVars(range(n), range(s_n), lb=0.0, name="T")
    tau = mdl.addVars(with_win, lb=-GRB.INFINITY, name="tau")
    nu = mdl.addVars(with_win, range(s_n), lb=0.0, name="nu")
    eta = mdl.addVars(with_win, lb=0.0, name="eta")

    mdl.addConstrs((x.sum(i, "*") == 1 for i in range(n)), name="out")
    mdl.addConstrs((x.sum("*", j) == 1 for j in range(n)), name="in")
    mdl.addConstr(u[DEPOT] == 0)
    mdl.addConstrs((u[i] >= 1 for i in range(1, n)))
    mdl.addConstrs(
        (u[i] - u[j] + n * x[i, j] <= n - 1
         for i, j in arcs if i != DEPOT and j != DEPOT),
        name="mtz",
    )

    if cfg.zone_precedence:
        groups = inst.zone_groups()
        zs = list(groups)
        pairs = [(a, c) for a in zs for c in zs if a != c]
        y = mdl.addVars(pairs, vtype=GRB.BINARY, name="y")
        for a in zs:
            for c in zs:
                if a < c:
                    mdl.addConstr(y[a, c] + y[c, a] == 1)
        for a in zs:
            for c in zs:
                for e in zs:
                    if len({a, c, e}) == 3:
                        mdl.addConstr(y[a, c] + y[c, e] - y[a, e] <= 1)
        for a in zs:
            off = gp.quicksum(len(groups[c]) * y[c, a] for c in zs if c != a)
            for i in groups[a]:
                mdl.addConstr(u[i] >= off + 1)
                mdl.addConstr(u[i] <= off + len(groups[a]))

    for s in range(s_n):
        mdl.addConstr(t[DEPOT, s] == 0)
        for i, j in arcs:
            if j == DEPOT:
                continue
            mdl.addConstr(
                t[j, s] >= t[i, s] + service[s, i] + travel[s, i, j] - m * (1 - x[i, j])
            )

    for i in with_win:
        for s in range(s_n):
            mdl.addConstr(nu[i, s] >= t[i, s] - float(b[i]) - tau[i])
        infl = _dual_weight_linear(amb, u[i], 1.0)
        mdl.addConstr(
            eta[i] >= tau[i]
            + (1.0 / (eps * s_n)) * gp.quicksum(nu[i, s] for s in range(s_n))
            + (amb.delta / eps) * infl
        )

    beh = obj.behavior_cost
    mdl.setObjective(
        (obj.lam / scale) * gp.quicksum(beh[i, j] * x[i, j] for i, j in arcs if j != DEPOT)
        + ((1.0 - obj.lam) / scale) * gp.quicksum(eta[i] for i in with_win),
        GRB.MINIMIZE,
    )

    t0 = time.perf_counter()
    mdl.optimize()
    rt = time.perf_counter() - t0
    if mdl.SolCount == 0:
        empty = Solution(order=list(range(n)), solver="exact_gurobi", runtime_s=rt)
        return MilpResult(empty, str(mdl.Status), float("nan"), float("nan"), rt, "gurobi")

    nxt = {i: j for i, j in arcs if x[i, j].X > 0.5}
    order, cur = [DEPOT], DEPOT
    for _ in range(n - 1):
        cur = nxt[cur]
        order.append(cur)
    sol = obj.to_solution(order, "exact_gurobi", rt, lower_bound=mdl.ObjBound,
                          milp_objective=mdl.ObjVal, n_scenarios=s_n)
    status = "Optimal" if mdl.Status == GRB.OPTIMAL else str(mdl.Status)
    return MilpResult(sol, status, mdl.ObjVal, mdl.ObjBound, rt, "gurobi",
                      n_vars=mdl.NumVars, n_cons=mdl.NumConstrs)


# ------------------------------------------------------------------ base MILP
def solve_base_milp(
    inst: Instance,
    theta: float = 1.0,
    cfg: SolverConfig | None = None,
) -> MilpResult:
    """The *base* MILP of §3: deterministic times, soft-lateness objective.

    Built and tested before the reliability layer is switched on (Phase 5,
    step 1).  ``c_ij`` is the planned travel time.
    """
    import pulp

    cfg = cfg or SolverConfig()
    n = inst.n
    tt, st, b = inst.travel_time, inst.service_time, inst.tw_end
    arcs = [(i, j) for i in range(n) for j in range(n) if i != j]
    with_win = [i for i in range(n) if np.isfinite(b[i])]
    m = float(tt.max() * n + st.sum() + 1.0)

    prob = pulp.LpProblem("mtp_base_tsptw", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", arcs, cat="Binary")
    u = pulp.LpVariable.dicts("u", range(n), lowBound=0, upBound=n - 1, cat="Integer")
    t = pulp.LpVariable.dicts("T", range(n), lowBound=0)
    ell = pulp.LpVariable.dicts("l", with_win, lowBound=0)

    for i in range(n):
        prob += pulp.lpSum(x[(i, j)] for j in range(n) if j != i) == 1
        prob += pulp.lpSum(x[(j, i)] for j in range(n) if j != i) == 1
    prob += u[DEPOT] == 0
    for i in range(1, n):
        prob += u[i] >= 1
    for i, j in arcs:
        if i != DEPOT and j != DEPOT:
            prob += u[i] - u[j] + n * x[(i, j)] <= n - 1
    prob += t[DEPOT] == 0
    for i, j in arcs:
        if j == DEPOT:
            continue
        prob += t[j] >= t[i] + st[i] + tt[i, j] - m * (1 - x[(i, j)])
    for i in with_win:
        prob += ell[i] >= t[i] - float(b[i])

    prob += (
        pulp.lpSum(tt[i, j] * x[(i, j)] for i, j in arcs)
        + theta * pulp.lpSum(ell[i] for i in with_win)
    )

    t0 = time.perf_counter()
    status = prob.solve(
        pulp.PULP_CBC_CMD(msg=0, timeLimit=int(max(cfg.time_limit_s, 1)))
    )
    rt = time.perf_counter() - t0
    name = pulp.LpStatus[status]
    if name != "Optimal":
        empty = Solution(order=list(range(n)), solver="base_milp", runtime_s=rt)
        return MilpResult(empty, name, float("nan"), float("nan"), rt, "pulp")
    order = _order_from_x(x, n)
    val = float(pulp.value(prob.objective))
    sol = Solution(order=order, objective=val, solver="base_milp", runtime_s=rt,
                   lower_bound=val)
    return MilpResult(sol, name, val, val, rt, "pulp",
                      n_vars=len(prob.variables()), n_cons=len(prob.constraints))


def base_milp_objective(inst: Instance, order: Sequence[int], theta: float = 1.0) -> float:
    """Closed-form value of the base MILP objective for a given sequence."""
    from core.objective import deterministic_lateness

    order = list(order)
    tt = inst.travel_time
    travel = sum(tt[a, c] for a, c in zip(order, order[1:] + [order[0]]))
    return float(travel + deterministic_lateness(inst, order, theta))


# ------------------------------------------------------------------ dispatcher
def solve_exact(
    inst: Instance,
    obj: CombinedObjective,
    cfg: SolverConfig | None = None,
    max_scenarios: int | None = 16,
) -> MilpResult:
    """Solve §3's full model with the best available backend."""
    cfg = cfg or SolverConfig()
    backend = cfg.exact_backend
    if backend == "auto":
        backend = "gurobi" if _has_gurobi() else "pulp"
    if backend == "gurobi":
        return _solve_gurobi(inst, obj, cfg, max_scenarios)
    if backend == "pulp":
        return _solve_pulp(inst, obj, cfg, max_scenarios)
    if backend == "bruteforce":
        return solve_bruteforce(inst, obj, cfg)
    raise ValueError(f"unknown exact backend {backend!r}")


def _has_gurobi() -> bool:
    try:
        import gurobipy  # noqa: F401
    except Exception:
        return False
    try:  # a licence check, not just an import check
        import gurobipy as gp

        env = gp.Env(params={"OutputFlag": 0})
        env.dispose()
        return True
    except Exception:  # pragma: no cover - depends on the local licence
        return False

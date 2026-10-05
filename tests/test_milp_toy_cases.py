"""Phase-5 gate (§9): the exact MILP must match hand-computed optima.

Three things are checked, in increasing strength:

1. **hand-computed values** -- three cases whose optimum was worked out on paper
   from the fixture's numbers (see the docstrings for the arithmetic);
2. **MILP vs. exhaustive enumeration** -- five parametrised cases where the PuLP
   encoding is compared against an independent brute-force optimum;
3. **structural properties** -- zone precedence honoured, a single Hamiltonian
   tour, and the base MILP agreeing with its own closed-form objective.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from core.types import zone_contiguous
from solvers.exact_milp import (
    base_milp_objective,
    count_feasible_orders,
    solve_base_milp,
    solve_bruteforce,
    solve_exact,
)
from tests.conftest import make_objective


# ------------------------------------------------------- 1. hand-computed cases
def test_hand_computed_pure_imitation(toy4, toy4_cost, solver_cfg):
    """lambda = 1.

    Behaviour costs are 1 on arcs (0,1),(1,2),(2,3) and 5 elsewhere, and the
    objective divides by n-1 = 3.  Sequence 0-1-2-3 costs (1+1+1)/3 = 1.0;
    0-1-3-2 costs (1+5+5)/3 = 3.667; every other sequence costs 15/3 = 5.0.
    """
    obj = make_objective(toy4, toy4_cost, lam=1.0)
    res = solve_exact(toy4, obj, solver_cfg, max_scenarios=1)
    assert res.status == "Optimal"
    assert res.objective == pytest.approx(1.0, abs=1e-6)
    assert res.solution.order == [0, 1, 2, 3]


def test_hand_computed_combined_objective(toy4, toy4_cost, solver_cfg):
    """lambda = 0.5, b_3 = 35.

    0-1-2-3 arrives at stop 3 at 10+5+10+5+10 = 40, so eta_3 = 5 and the
    objective is 0.5*(3/3) + 0.5*(5/3) = 4/3.  The best alternative, 0-1-3-2,
    scores 0.5*(11/3) + 0 = 1.833; the remaining four sequences all score 2.5
    or worse.
    """
    obj = make_objective(toy4, toy4_cost, lam=0.5)
    res = solve_exact(toy4, obj, solver_cfg, max_scenarios=1)
    assert res.status == "Optimal"
    assert res.objective == pytest.approx(4.0 / 3.0, abs=1e-6)
    assert res.solution.order == [0, 1, 2, 3]


def test_hand_computed_pure_reliability(toy4, toy4_cost, solver_cfg):
    """lambda = 0: the optimum is exactly zero risk, reachable several ways."""
    obj = make_objective(toy4, toy4_cost, lam=0.0)
    res = solve_exact(toy4, obj, solver_cfg, max_scenarios=1)
    assert res.status == "Optimal"
    assert res.objective == pytest.approx(0.0, abs=1e-6)
    assert obj.evaluator.total_eta(res.solution.order) == pytest.approx(0.0, abs=1e-6)


def test_hand_computed_loose_window_is_free(toy4_loose, toy4_cost, solver_cfg):
    """With b_3 = 40 the behaviourally best route is also perfectly reliable."""
    obj = make_objective(toy4_loose, toy4_cost, lam=0.5)
    res = solve_exact(toy4_loose, obj, solver_cfg, max_scenarios=1)
    assert res.objective == pytest.approx(0.5 * 1.0, abs=1e-6)
    assert res.solution.order == [0, 1, 2, 3]


def test_hand_computed_wasserstein_inflation(toy4_loose, toy4_cost, solver_cfg):
    """delta > 0 inflates eta by (delta/eps)*(2u_i - 1) at the windowed stop.

    Stop 3 sits at position 3, so ||w||_1 = 5 and with delta = 0.1, eps = 0.1
    the surplus is 5.0; the objective becomes 0.5*1.0 + 0.5*(5/3) = 4/3.
    """
    obj = make_objective(toy4_loose, toy4_cost, lam=0.5, eps=0.1, delta=0.1)
    res = solve_exact(toy4_loose, obj, solver_cfg, max_scenarios=1)
    assert res.status == "Optimal"
    assert res.solution.order == [0, 1, 2, 3]
    assert res.objective == pytest.approx(0.5 + 0.5 * 5.0 / 3.0, abs=1e-6)


# --------------------------------------------- 2. MILP vs. exhaustive enumeration
@pytest.mark.parametrize(
    "lam,eps,delta,fixture",
    [
        (1.0, 0.10, 0.0, "toy4"),
        (0.75, 0.10, 0.0, "toy4"),
        (0.50, 0.10, 0.0, "toy4"),
        (0.25, 0.10, 0.1, "toy4"),
        (0.00, 0.10, 0.0, "toy4"),
        (0.50, 0.20, 0.0, "toy4_loose"),
        (0.50, 0.10, 0.0, "toy_zones"),
        (0.25, 0.10, 0.2, "toy_zones"),
    ],
)
def test_milp_matches_bruteforce(lam, eps, delta, fixture, toy4_cost, solver_cfg,
                                request):
    inst = request.getfixturevalue(fixture)
    cost = (
        toy4_cost if inst.n == 4
        else np.where(np.eye(inst.n, dtype=bool), 0.0, 3.0) + 0.1 * inst.travel_time
    )
    obj = make_objective(inst, cost, lam=lam, eps=eps, delta=delta)

    brute = solve_bruteforce(inst, obj, solver_cfg)
    milp = solve_exact(inst, obj, solver_cfg, max_scenarios=1)

    assert milp.status == "Optimal"
    assert milp.objective == pytest.approx(brute.objective, abs=1e-5)
    # The MILP's own objective value must equal the evaluator's re-scoring.
    assert obj(milp.solution.order) == pytest.approx(milp.objective, abs=1e-5)


# ----------------------------------------------------- 3. structural properties
def test_milp_respects_zone_precedence(toy_zones, solver_cfg):
    cost = np.where(np.eye(toy_zones.n, dtype=bool), 0.0, 2.0)
    obj = make_objective(toy_zones, cost, lam=0.5)
    res = solve_exact(toy_zones, obj, solver_cfg, max_scenarios=1)
    assert res.status == "Optimal"
    assert zone_contiguous(res.solution.order, toy_zones.zones)


def test_zone_precedence_can_be_switched_off(toy_zones, solver_cfg):
    from dataclasses import replace

    cfg = replace(solver_cfg, zone_precedence=False)
    cost = np.where(np.eye(toy_zones.n, dtype=bool), 0.0, 2.0)
    obj = make_objective(toy_zones, cost, lam=0.5)
    free = solve_bruteforce(toy_zones, obj, cfg)
    zoned = solve_bruteforce(toy_zones, obj, solver_cfg)
    # Relaxing a constraint cannot make the optimum worse.
    assert free.objective <= zoned.objective + 1e-9


def test_milp_returns_a_single_tour(toy_zones, solver_cfg):
    cost = np.where(np.eye(toy_zones.n, dtype=bool), 0.0, 2.0)
    obj = make_objective(toy_zones, cost, lam=1.0)
    res = solve_exact(toy_zones, obj, solver_cfg, max_scenarios=1)
    order = res.solution.order
    assert sorted(order) == list(range(toy_zones.n))
    assert order[0] == 0


def test_base_milp_matches_its_closed_form(toy4, solver_cfg):
    """The deterministic base MILP of §3, before the reliability layer."""
    res = solve_base_milp(toy4, theta=1.0, cfg=solver_cfg)
    assert res.status == "Optimal"
    assert res.objective == pytest.approx(
        base_milp_objective(toy4, res.solution.order, theta=1.0), abs=1e-5
    )
    best = min(
        (base_milp_objective(toy4, [0, *p], 1.0)
         for p in itertools.permutations([1, 2, 3])),
    )
    assert res.objective == pytest.approx(best, abs=1e-5)


def test_feasible_order_counting(toy_zones):
    # 2 zones of 2 stops: 2! zone orders x 2! x 2! within-zone orders = 8
    assert count_feasible_orders(toy_zones, zone_precedence=True) == 8
    assert count_feasible_orders(toy_zones, zone_precedence=False) == 24


def test_l2_ground_norm_is_rejected_by_the_exact_solver(toy4, toy4_cost, solver_cfg):
    obj = make_objective(toy4, toy4_cost, lam=0.5, delta=0.1)
    obj.evaluator.amb.ground_norm = "l2"
    with pytest.raises(ValueError, match="not MILP-representable"):
        solve_exact(toy4, obj, solver_cfg, max_scenarios=1)


# ------------------------------------------ multi-scenario exact reformulation
@pytest.mark.parametrize("lam,delta", [(1.0, 0.0), (0.5, 0.0), (0.5, 0.2), (0.0, 0.0)])
def test_milp_matches_bruteforce_multi_scenario(stochastic_toy_zones, solver_cfg,
                                                lam, delta):
    """The scenario-based CVaR block, with the full five-scenario P_hat."""
    from core.objective import CombinedObjective
    from solvers.reliability import ReliabilityEvaluator

    inst, amb = stochastic_toy_zones
    amb = amb.with_radius(delta)
    cost = np.where(np.eye(inst.n, dtype=bool), 0.0, 3.0) + 0.05 * inst.travel_time
    obj = CombinedObjective(inst=inst, behavior_cost=cost,
                            evaluator=ReliabilityEvaluator(inst, amb), lam=lam)
    brute = solve_bruteforce(inst, obj, solver_cfg)
    milp = solve_exact(inst, obj, solver_cfg, max_scenarios=None)
    assert milp.status == "Optimal"
    assert milp.objective == pytest.approx(brute.objective, abs=1e-5)
    assert obj(milp.solution.order) == pytest.approx(milp.objective, abs=1e-5)

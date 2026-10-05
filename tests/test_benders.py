"""Logic-based Benders tests.

The decomposition must (a) terminate at the true optimum on enumerable
instances, (b) report a valid lower bound at all times, and (c) actually use its
prefix cuts rather than degenerating into no-good cuts on whole tours.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.types import zone_contiguous
from solvers.benders.benders_solver import solve_benders
from solvers.exact_milp import solve_bruteforce
from tests.conftest import make_objective


@pytest.mark.parametrize("lam", [1.0, 0.5, 0.25])
def test_benders_matches_bruteforce_on_toy4(toy4, toy4_cost, solver_cfg, lam):
    obj = make_objective(toy4, toy4_cost, lam=lam)
    brute = solve_bruteforce(toy4, obj, solver_cfg)
    res = solve_benders(toy4, obj, solver_cfg)
    assert res.upper_bound == pytest.approx(brute.objective, abs=1e-6)
    assert res.lower_bound <= brute.objective + 1e-6


@pytest.mark.parametrize("delta", [0.0, 0.2])
def test_benders_matches_bruteforce_with_zones(toy_zones, solver_cfg, delta):
    cost = np.where(np.eye(toy_zones.n, dtype=bool), 0.0, 3.0) + 0.1 * toy_zones.travel_time
    obj = make_objective(toy_zones, cost, lam=0.5, delta=delta)
    brute = solve_bruteforce(toy_zones, obj, solver_cfg)
    res = solve_benders(toy_zones, obj, solver_cfg)
    assert res.upper_bound == pytest.approx(brute.objective, abs=1e-6)
    assert zone_contiguous(res.solution.order, toy_zones.zones)


def test_lower_bound_never_exceeds_the_optimum(toy4, toy4_cost, solver_cfg):
    obj = make_objective(toy4, toy4_cost, lam=0.5)
    brute = solve_bruteforce(toy4, obj, solver_cfg)
    res = solve_benders(toy4, obj, solver_cfg)
    for h in res.history:
        assert h["lb"] <= brute.objective + 1e-6
    assert 0.0 <= res.gap < 1e-4 or res.status != "Optimal"


def test_cuts_are_generated_only_when_risk_is_underestimated(toy4_loose, toy4_cost,
                                                             solver_cfg):
    """With a loose window the behaviour optimum is already reliable: no cut."""
    obj = make_objective(toy4_loose, toy4_cost, lam=0.5)
    res = solve_benders(toy4_loose, obj, solver_cfg)
    assert res.status == "Optimal"
    assert res.n_cuts == 0
    assert res.solution.order == [0, 1, 2, 3]


def test_cuts_are_generated_when_the_window_bites(toy4, toy4_cost, solver_cfg):
    obj = make_objective(toy4, toy4_cost, lam=0.1)
    res = solve_benders(toy4, obj, solver_cfg)
    assert res.n_cuts >= 1
    assert res.status in ("Optimal", "IterationLimit", "TimeLimit")


def test_warm_start_is_used_as_an_upper_bound(toy4, toy4_cost, solver_cfg):
    obj = make_objective(toy4, toy4_cost, lam=0.5)
    warm = [0, 3, 2, 1]
    res = solve_benders(toy4, obj, solver_cfg, initial=warm)
    assert res.upper_bound <= obj(warm) + 1e-9


def test_iteration_limit_reports_a_gap(toy_zones, solver_cfg):
    cost = np.where(np.eye(toy_zones.n, dtype=bool), 0.0, 3.0)
    obj = make_objective(toy_zones, cost, lam=0.1, delta=0.5)
    res = solve_benders(toy_zones, obj, solver_cfg, max_iterations=1)
    assert res.iterations <= 1
    assert np.isfinite(res.gap)

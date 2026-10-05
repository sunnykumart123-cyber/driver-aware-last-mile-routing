"""The ALNS must reach the exact optimum on instances small enough to enumerate.

Also checks the two properties the search depends on for correctness:

* the **incremental** objective bookkeeping in :class:`RouteState` agrees with a
  from-scratch evaluation (if it did not, the search would be optimising a
  different function than the one reported);
* every destroy/repair pair preserves a valid, zone-contiguous permutation.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from core.types import is_valid_order, zone_contiguous
from solvers.alns.alns_solver import solve_alns, zone_nearest_neighbour
from solvers.alns.destroy_ops import DESTROY_OPERATORS
from solvers.alns.repair_ops import greedy_insertion, regret_k_insertion
from solvers.alns.state import RouteState
from solvers.exact_milp import solve_bruteforce
from tests.conftest import make_objective


@pytest.mark.parametrize("lam", [1.0, 0.75, 0.5, 0.25, 0.0])
def test_alns_finds_the_exact_optimum_on_toy4(toy4, toy4_cost, solver_cfg, lam):
    obj = make_objective(toy4, toy4_cost, lam=lam)
    brute = solve_bruteforce(toy4, obj, solver_cfg)
    alns = solve_alns(toy4, obj, solver_cfg)
    assert alns.solution.objective == pytest.approx(brute.objective, abs=1e-6)


@pytest.mark.parametrize("lam,delta", [(0.5, 0.0), (0.5, 0.2), (0.25, 0.1)])
def test_alns_finds_the_exact_optimum_with_zones(toy_zones, solver_cfg, lam, delta):
    cost = np.where(np.eye(toy_zones.n, dtype=bool), 0.0, 3.0) + 0.1 * toy_zones.travel_time
    obj = make_objective(toy_zones, cost, lam=lam, delta=delta)
    brute = solve_bruteforce(toy_zones, obj, solver_cfg)
    alns = solve_alns(toy_zones, obj, solver_cfg)
    assert alns.solution.objective == pytest.approx(brute.objective, abs=1e-6)
    assert zone_contiguous(alns.solution.order, toy_zones.zones)


def test_incremental_and_full_objectives_agree(toy_zones, solver_cfg):
    """The ALNS's cached objective must equal a fresh evaluation of the route."""
    cost = np.where(np.eye(toy_zones.n, dtype=bool), 0.0, 3.0) + 0.1 * toy_zones.travel_time
    obj = make_objective(toy_zones, cost, lam=0.5, delta=0.1)
    res = solve_alns(toy_zones, obj, solver_cfg)
    incremental = res.solution.meta["incremental_objective"]
    assert res.solution.objective == pytest.approx(incremental, abs=1e-9)
    assert obj(res.solution.order) == pytest.approx(incremental, abs=1e-9)


def test_insertion_delta_matches_a_recomputed_state(toy_zones):
    """The O(W*S) insertion formula must equal an O(S*n) rebuild."""
    cost = np.where(np.eye(toy_zones.n, dtype=bool), 0.0, 3.0)
    obj = make_objective(toy_zones, cost, lam=0.4, delta=0.15)
    full = RouteState(obj, [0, 1, 3, 2, 4], zone_precedence=False)

    for removed in (1, 2, 3, 4):
        partial = RouteState(
            obj, [i for i in full.order if i != removed], zone_precedence=False
        )
        before = partial.objective
        for p in range(1, len(partial.order) + 1):
            delta = partial.insertion_delta(removed, p)
            probe = RouteState(
                obj, partial.order[:p] + [removed] + partial.order[p:],
                zone_precedence=False,
            )
            assert delta == pytest.approx(probe.objective - before, abs=1e-9)


@pytest.mark.parametrize("op_name", sorted(DESTROY_OPERATORS))
@pytest.mark.parametrize("repair", [greedy_insertion, regret_k_insertion])
def test_operators_preserve_feasibility(toy_zones, solver_cfg, op_name, repair):
    cost = np.where(np.eye(toy_zones.n, dtype=bool), 0.0, 3.0)
    obj = make_objective(toy_zones, cost, lam=0.5)
    rng = np.random.default_rng(3)
    for _ in range(20):
        state = RouteState(obj, zone_nearest_neighbour(toy_zones), True)
        DESTROY_OPERATORS[op_name](state, 2, rng)
        assert len(state.order) + len(state.removed) == toy_zones.n
        repair(state, rng)
        assert state.is_complete
        assert is_valid_order(state.order, toy_zones.n)
        assert zone_contiguous(state.order, toy_zones.zones)


def test_alns_improves_on_its_starting_point(toy_zones, solver_cfg):
    cost = np.where(np.eye(toy_zones.n, dtype=bool), 0.0, 3.0)
    cost[0, 4] = cost[4, 2] = cost[2, 3] = cost[3, 1] = 0.5   # a hidden good path
    obj = make_objective(toy_zones, cost, lam=1.0)
    res = solve_alns(toy_zones, obj, solver_cfg)
    assert res.solution.objective <= res.history[0] + 1e-9
    assert res.iterations > 0
    assert set(res.operator_stats) == {"destroy", "repair"}


def test_alns_is_reproducible(toy_zones, solver_cfg):
    cost = np.where(np.eye(toy_zones.n, dtype=bool), 0.0, 3.0)
    obj = make_objective(toy_zones, cost, lam=0.5)
    a = solve_alns(toy_zones, obj, solver_cfg, seed=11)
    b = solve_alns(toy_zones, obj, solver_cfg, seed=11)
    assert a.solution.order == b.solution.order
    assert a.solution.objective == pytest.approx(b.solution.objective)


def test_nearest_neighbour_start_is_zone_contiguous(toy_zones):
    order = zone_nearest_neighbour(toy_zones)
    assert is_valid_order(order, toy_zones.n)
    assert zone_contiguous(order, toy_zones.zones)


def test_time_limit_is_respected(toy_zones, solver_cfg):
    """The search must stop on the clock rather than on the iteration count.

    Asserted on the *reason* it stopped, not on a wall-clock bound: a tight
    bound here is a flake waiting to happen on a loaded machine (this test
    failed at `runtime_s < 6.0` with a 1 s limit while training jobs were
    running).  The generous ceiling below only catches a limit that is ignored
    outright.
    """
    cfg = replace(solver_cfg, alns_iterations=10_000_000, time_limit_s=1.0)
    cost = np.where(np.eye(toy_zones.n, dtype=bool), 0.0, 3.0)
    obj = make_objective(toy_zones, cost, lam=0.5)
    res = solve_alns(toy_zones, obj, cfg)
    assert res.iterations < 10_000_000, "stopped on iterations, not on time"
    assert res.iterations > 0
    assert res.runtime_s < 120.0, "the time limit was ignored entirely"


# ---------------------------------------------------- multi-scenario ambiguity
@pytest.mark.parametrize("lam,delta,geometry", [
    (0.5, 0.0, "wasserstein"),
    (0.5, 0.3, "wasserstein"),
    (0.25, 0.1, "wasserstein"),
    (0.5, 0.0, "box"),
    (0.5, 0.0, "ellipsoidal"),
    (0.0, 0.2, "wasserstein"),
])
def test_alns_matches_bruteforce_multi_scenario(stochastic_toy_zones, solver_cfg,
                                                lam, delta, geometry):
    """The case a single-scenario fixture cannot exercise: S > 1 broadcasting."""
    from core.objective import CombinedObjective
    from models.uncertainty.ambiguity_set import AmbiguitySet
    from solvers.reliability import ReliabilityEvaluator

    inst, amb = stochastic_toy_zones
    amb = AmbiguitySet(amb.travel, amb.service, amb.eps, delta,
                       amb.ground_norm, geometry, amb.route_id)
    cost = np.where(np.eye(inst.n, dtype=bool), 0.0, 3.0) + 0.05 * inst.travel_time
    obj = CombinedObjective(inst=inst, behavior_cost=cost,
                            evaluator=ReliabilityEvaluator(inst, amb), lam=lam)
    brute = solve_bruteforce(inst, obj, solver_cfg)
    alns = solve_alns(inst, obj, solver_cfg)
    assert alns.solution.objective == pytest.approx(brute.objective, abs=1e-6)
    assert zone_contiguous(alns.solution.order, inst.zones)


def test_insertion_and_removal_deltas_multi_scenario(stochastic_toy_zones):
    """Incremental insert/remove algebra must be exact with S > 1 too."""
    from core.objective import CombinedObjective
    from solvers.reliability import ReliabilityEvaluator

    inst, amb = stochastic_toy_zones
    cost = np.where(np.eye(inst.n, dtype=bool), 0.0, 3.0)
    obj = CombinedObjective(inst=inst, behavior_cost=cost,
                            evaluator=ReliabilityEvaluator(inst, amb.with_radius(0.2)),
                            lam=0.4)
    full = RouteState(obj, [0, 1, 3, 2, 4], zone_precedence=False)

    # removal
    removal = full.removal_deltas()
    for p in range(1, len(full.order)):
        probe = RouteState(obj, [i for k, i in enumerate(full.order) if k != p],
                           zone_precedence=False)
        assert removal[p - 1] == pytest.approx(probe.objective - full.objective,
                                               abs=1e-9)
    # insertion
    for removed in (1, 2, 3, 4):
        partial = RouteState(obj, [i for i in full.order if i != removed],
                             zone_precedence=False)
        before = partial.objective
        deltas, positions = partial.insertion_deltas(removed)
        for d, p in zip(deltas, positions):
            probe = RouteState(obj, partial.order[:p] + [removed] + partial.order[p:],
                               zone_precedence=False)
            assert d == pytest.approx(probe.objective - before, abs=1e-9)


# ------------------------------------------------- unconstrained zone ordering
@pytest.mark.parametrize("lam", [1.0, 0.5, 0.0])
def test_alns_matches_bruteforce_without_zone_precedence(stochastic_toy_zones,
                                                          solver_cfg, lam):
    """The new default: zones may be re-entered, as real drivers do."""
    from core.objective import CombinedObjective
    from solvers.reliability import ReliabilityEvaluator

    inst, amb = stochastic_toy_zones
    cfg = replace(solver_cfg, zone_precedence=False, candidate_cap=0)
    cost = np.where(np.eye(inst.n, dtype=bool), 0.0, 3.0) + 0.05 * inst.travel_time
    obj = CombinedObjective(inst=inst, behavior_cost=cost,
                            evaluator=ReliabilityEvaluator(inst, amb), lam=lam)
    brute = solve_bruteforce(inst, obj, cfg)
    alns = solve_alns(inst, obj, cfg)
    assert alns.solution.objective == pytest.approx(brute.objective, abs=1e-6)


def test_relaxing_zone_precedence_cannot_worsen_the_optimum(stochastic_toy_zones,
                                                             solver_cfg):
    from core.objective import CombinedObjective
    from solvers.reliability import ReliabilityEvaluator

    inst, amb = stochastic_toy_zones
    cost = np.where(np.eye(inst.n, dtype=bool), 0.0, 3.0)
    obj = CombinedObjective(inst=inst, behavior_cost=cost,
                            evaluator=ReliabilityEvaluator(inst, amb), lam=0.5)
    zoned = solve_bruteforce(inst, obj, replace(solver_cfg, zone_precedence=True))
    free = solve_bruteforce(inst, obj, replace(solver_cfg, zone_precedence=False))
    assert free.objective <= zoned.objective + 1e-9


def test_candidate_cap_still_yields_a_valid_complete_route(stochastic_toy_zones,
                                                            solver_cfg):
    """The shortlist restricts the search, never the validity of the result."""
    from core.objective import CombinedObjective
    from core.types import is_valid_order
    from solvers.reliability import ReliabilityEvaluator

    inst, amb = stochastic_toy_zones
    cfg = replace(solver_cfg, zone_precedence=False, candidate_cap=2)
    cost = np.where(np.eye(inst.n, dtype=bool), 0.0, 3.0) + 0.05 * inst.travel_time
    obj = CombinedObjective(inst=inst, behavior_cost=cost,
                            evaluator=ReliabilityEvaluator(inst, amb), lam=0.5)
    res = solve_alns(inst, obj, cfg)
    assert is_valid_order(res.solution.order, inst.n)
    # Whatever position the shortlist picked, the reported objective is exact.
    assert obj(res.solution.order) == pytest.approx(res.solution.objective, abs=1e-9)


def test_shortlist_keeps_the_exact_best_when_cap_is_large(stochastic_toy_zones):
    from core.objective import CombinedObjective
    from solvers.reliability import ReliabilityEvaluator

    inst, amb = stochastic_toy_zones
    cost = np.where(np.eye(inst.n, dtype=bool), 0.0, 3.0)
    obj = CombinedObjective(inst=inst, behavior_cost=cost,
                            evaluator=ReliabilityEvaluator(inst, amb), lam=0.5)
    uncapped = RouteState(obj, [0, 1, 3, 2], zone_precedence=False, candidate_cap=0)
    capped = RouteState(obj, [0, 1, 3, 2], zone_precedence=False, candidate_cap=99)
    assert uncapped.best_insertion(4) == capped.best_insertion(4)

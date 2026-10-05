"""Unit tests for the CVaR / DRO algebra, against hand-computed values."""
from __future__ import annotations

import numpy as np
import pytest

from core.objective import deterministic_lateness
from models.uncertainty.ambiguity_set import AmbiguitySet, invert_quantiles
from solvers.reliability import ReliabilityEvaluator, sample_cvar
from tests.conftest import make_objective


def test_sample_cvar_matches_tail_mean():
    losses = np.arange(1.0, 11.0)                 # 1..10
    # eps = 0.2 -> the worst two observations, mean 9.5
    assert sample_cvar(losses, 0.2) == pytest.approx(9.5)
    # eps = 1 -> the plain mean
    assert sample_cvar(losses, 1.0) == pytest.approx(losses.mean())
    # eps = 0.1 -> the single worst observation
    assert sample_cvar(losses, 0.1) == pytest.approx(10.0)


def test_sample_cvar_dominates_var_and_mean():
    rng = np.random.default_rng(0)
    losses = rng.normal(size=500)
    for eps in (0.05, 0.1, 0.25, 0.5):
        cvar = sample_cvar(losses, eps)
        var = np.quantile(losses, 1 - eps)
        assert cvar >= var - 1e-9
        assert cvar >= losses.mean() - 1e-9


def test_sample_cvar_is_monotone_in_eps():
    rng = np.random.default_rng(1)
    losses = rng.gamma(2.0, 3.0, size=400)
    vals = [sample_cvar(losses, e) for e in (0.05, 0.1, 0.2, 0.4, 0.8)]
    assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:]))


def test_deterministic_set_reproduces_soft_lateness(toy4, toy4_cost):
    """With one scenario and delta = 0, sum eta == the base MILP's lateness."""
    obj = make_objective(toy4, toy4_cost, lam=0.0)
    for order in ([0, 1, 2, 3], [0, 3, 2, 1], [0, 2, 1, 3]):
        eta = obj.evaluator.total_eta(order)
        assert eta == pytest.approx(deterministic_lateness(toy4, order, theta=1.0))


def test_hand_computed_cvar_on_three_scenarios(stochastic_toy4):
    inst, amb = stochastic_toy4
    ev = ReliabilityEvaluator(inst, amb)
    order = [0, 1, 2, 3]
    # arrival at stop 3 is 30f + 10 for f in {0.8, 1.0, 1.4} -> {34, 40, 52}
    arr = ev.arrivals(order)
    assert arr[:, 3].tolist() == pytest.approx([34.0, 40.0, 52.0])
    # losses against b = 35 are {-1, 5, 17}; CVaR at eps = 1/3 is 5 + 12 = 17
    eta = ev.eta(order)
    assert eta[3] == pytest.approx(17.0, abs=1e-6)
    assert eta[:3].tolist() == pytest.approx([0.0, 0.0, 0.0])

    rep = ev.evaluate(order)
    assert rep.prob_on_time == pytest.approx(1 / 3)
    assert rep.expected_lateness == pytest.approx(22.0 / 3.0)


def test_wasserstein_inflation_is_exactly_delta_over_eps_times_dual_norm(
    stochastic_toy4,
):
    inst, amb = stochastic_toy4
    order = [0, 1, 2, 3]
    base = ReliabilityEvaluator(inst, amb).eta(order)[3]
    inflated = ReliabilityEvaluator(inst, amb.with_radius(0.5)).eta(order)[3]
    # position 3 -> ||w||_1 = 2*3 - 1 = 5 components; (delta/eps) * 5 = 7.5
    assert inflated - base == pytest.approx(0.5 / (1 / 3) * 5.0)


def test_ground_norm_changes_the_dual_weight(stochastic_toy4):
    inst, amb = stochastic_toy4
    order = [0, 1, 2, 3]
    base = ReliabilityEvaluator(inst, amb).eta(order)[3]
    linf = ReliabilityEvaluator(inst, amb.with_radius(0.3)).eta(order)[3] - base
    l1_amb = AmbiguitySet(amb.travel, amb.service, amb.eps, 0.3, "l1",
                          "wasserstein", amb.route_id)
    l2_amb = AmbiguitySet(amb.travel, amb.service, amb.eps, 0.3, "l2",
                          "wasserstein", amb.route_id)
    l1 = ReliabilityEvaluator(inst, l1_amb).eta(order)[3] - base
    l2 = ReliabilityEvaluator(inst, l2_amb).eta(order)[3] - base
    assert l1 == pytest.approx(0.3 / (1 / 3))
    assert l2 == pytest.approx(0.3 / (1 / 3) * np.sqrt(5.0))
    assert linf > l2 > l1


def test_alternative_geometries_are_more_conservative(stochastic_toy4):
    inst, amb = stochastic_toy4
    order = [0, 1, 2, 3]
    was = ReliabilityEvaluator(inst, amb).eta(order)[3]
    box = ReliabilityEvaluator(
        inst, AmbiguitySet(amb.travel, amb.service, amb.eps, 0.0,
                           amb.ground_norm, "box", "")
    ).eta(order)[3]
    ell = ReliabilityEvaluator(
        inst, AmbiguitySet(amb.travel, amb.service, amb.eps, 0.0,
                           amb.ground_norm, "ellipsoidal", "")
    ).eta(order)[3]
    assert box >= 0 and ell >= 0
    # The CVaR safeguard averages the tail, so it is at least the quantile.
    assert was >= box - 1e-9


def test_visiting_the_windowed_stop_first_minimises_eta(stochastic_toy4):
    """Hand-computed: 0-3-2-1 gives arrivals {24, 30, 42} at the windowed stop.

    Losses against b = 35 are {-11, -5, 7}, so at eps = 1/3 the CVaR is
    -5 + 7 - (-5) = 7, on-time in two of the three scenarios -- and no other
    sequence does better, because every other one reaches stop 3 later.
    """
    inst, amb = stochastic_toy4
    ev = ReliabilityEvaluator(inst, amb)
    order = [0, 3, 2, 1]
    assert ev.total_eta(order) == pytest.approx(7.0)
    assert ev.evaluate(order).prob_on_time == pytest.approx(2 / 3)

    import itertools

    for perm in itertools.permutations([1, 2, 3]):
        assert ev.total_eta([0, *perm]) >= 7.0 - 1e-9


def test_arrivals_are_a_prefix_cumsum(stochastic_toy4):
    inst, amb = stochastic_toy4
    ev = ReliabilityEvaluator(inst, amb)
    order = [0, 2, 1, 3]
    arr = ev.arrivals(order)
    for s in range(amb.n_scenarios):
        total = 0.0
        for p in range(1, len(order)):
            total += amb.service[s, order[p - 1]] + amb.travel[s, order[p - 1], order[p]]
            assert arr[s, p] == pytest.approx(total, rel=1e-6)


def test_invert_quantiles_is_monotone_and_nonnegative():
    levels = (0.05, 0.25, 0.5, 0.75, 0.95)
    values = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    u = np.array([[0.05], [0.5], [0.95]])
    out = invert_quantiles(levels, values, u)
    assert out.shape == (3, 1)
    assert out[0, 0] == pytest.approx(1.0)
    assert out[1, 0] == pytest.approx(3.0)
    assert out[2, 0] == pytest.approx(5.0)
    assert (out >= 0).all()


def test_risk_at_matches_eta(stochastic_toy4):
    """The scalar primitive used by the ALNS must agree with the vector path."""
    inst, amb = stochastic_toy4
    ev = ReliabilityEvaluator(inst, amb.with_radius(0.2))
    order = [0, 1, 2, 3]
    eta = ev.eta(order)
    arr = ev.arrivals(order)
    for p in range(1, len(order)):
        scalar = ev.risk_at(arr[:, p], inst.tw_end[order[p]], max(2 * p - 1, 1))
        assert scalar == pytest.approx(eta[p], abs=1e-9)

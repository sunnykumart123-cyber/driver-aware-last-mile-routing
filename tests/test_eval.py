"""Metric, logging and significance-testing tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import Config, UncertaintyConfig
from core.io import append_results, load_results
from eval.erp_official import kendall_tau, scorer_available, sequence_deviation
from eval.metrics import EVAL_SEED_OFFSET, EvalContext, evaluate_solution, price_of_robustness, summarise
from eval.significance_tests import cliffs_delta, holm, lambda_trend_test, paired_compare
from solvers.alns.alns_solver import solve_alns
from tests.conftest import make_objective


# ------------------------------------------------------------------- similarity
def test_sequence_deviation_is_zero_for_the_actual_sequence(toy4):
    assert sequence_deviation(toy4, toy4.actual_sequence) == pytest.approx(0.0)
    assert kendall_tau(toy4, toy4.actual_sequence) == pytest.approx(1.0)


def test_sequence_deviation_grows_with_disorder(toy4):
    near = [0, 1, 3, 2]
    far = [0, 3, 2, 1]
    assert 0 < sequence_deviation(toy4, near) < sequence_deviation(toy4, far)
    assert kendall_tau(toy4, near) > kendall_tau(toy4, far)


def test_erp_is_not_silently_substituted():
    """If the official scorer is absent, nothing may pretend to be ERP."""
    from eval.erp_official import ScorerUnavailable, load_official_scorer

    if scorer_available():
        pytest.skip("the official scorer is installed in this environment")
    with pytest.raises(ScorerUnavailable):
        load_official_scorer()


# ------------------------------------------------------------------ result rows
def test_evaluate_solution_row_schema(toy4, toy4_cost, solver_cfg):
    cfg = Config()
    cfg.uncertainty = UncertaintyConfig(n_scenarios=1)
    obj = make_objective(toy4, toy4_cost, lam=0.5)
    sol = solve_alns(toy4, obj, solver_cfg).solution
    ctx = EvalContext(cfg, None, n_eval_scenarios=1)
    row = evaluate_solution(toy4, sol, obj, ctx, "unit_test", "fold_x")

    for key in ("method", "lam", "metro", "route_id", "n_stops", "reliability_index",
                "expected_lateness", "objective", "behavior_cost", "risk_cost",
                "solver", "runtime_s", "delta", "eps", "geometry"):
        assert key in row
    assert row["method"] == "unit_test"
    assert row["fold"] == "fold_x"
    assert 0.0 <= row["reliability_index"] <= 1.0
    assert row["expected_lateness"] >= 0.0
    assert row["erp_available"] == scorer_available()


def test_evaluation_uses_an_independent_scenario_stream(toy4):
    """The measurement seed must differ from the optimizer's."""
    cfg = Config()
    ctx = EvalContext(cfg, None)
    assert EVAL_SEED_OFFSET != 0
    ev = ctx.independent_evaluator(toy4)
    # delta is stripped for measurement: the yardstick is the empirical law.
    assert ev.amb.delta == 0.0


def test_append_and_load_results_round_trip(tmp_path):
    path = tmp_path / "experiment_log.parquet"
    append_results([{"method": "a", "objective": 1.0}], path)
    append_results([{"method": "b", "objective": 2.0}], path)
    df = load_results(path)
    assert len(df) == 2
    assert set(df["method"]) == {"a", "b"}
    assert load_results(tmp_path / "missing.parquet").empty


# --------------------------------------------------------------- aggregate views
def _fake_sweep() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for route in range(30):
        base = rng.normal(0.4, 0.05)
        for lam in (1.0, 0.5, 0.0):
            rows.append({
                "method": "mtp", "lam": lam, "metro": "Austin", "fold": "ho_austin",
                "route_id": f"r{route}",
                # reliability improves as lambda falls, similarity degrades
                "reliability_index": min(1.0, base + 0.4 * (1 - lam)),
                "erp": base + 0.3 * (1 - lam) + rng.normal(0, 0.01),
                "seq_dev": base,
                "expected_lateness": 10 * lam,
                "objective": 1.0, "runtime_s": 0.1,
            })
    return pd.DataFrame(rows)


def test_price_of_robustness_is_paired_and_zero_at_the_reference():
    df = _fake_sweep()
    por = price_of_robustness(df, metric="erp")
    at_one = por[np.isclose(por["lam"], 1.0)]
    assert at_one["d_erp"].abs().max() < 1e-9
    assert at_one["d_reliability"].abs().max() < 1e-9
    at_zero = por[np.isclose(por["lam"], 0.0)]
    assert at_zero["d_erp"].mean() > 0            # similarity gets worse
    assert at_zero["d_reliability"].mean() > 0    # reliability gets better


def test_summarise_produces_one_row_per_group():
    out = summarise(_fake_sweep(), by=("lam",))
    assert len(out) == 3
    assert "reliability_index_mean" in out.columns
    assert (out["reliability_index_count"] == 30).all()


# ------------------------------------------------------------------ significance
def test_paired_compare_detects_a_real_difference():
    df = _fake_sweep()
    df["method"] = "lam_" + df["lam"].astype(str)
    out = paired_compare(df, "reliability_index")
    assert len(out) == 3                     # three pairs among three lambdas
    assert (out["n_pairs"] == 30).all()
    sig = out[out["p_holm"] < 0.05]
    assert len(sig) >= 1
    assert sig["rank_biserial"].abs().max() > 0.5


def test_paired_compare_reports_identical_methods_without_crashing():
    df = _fake_sweep()
    dup = df[np.isclose(df["lam"], 1.0)].copy()
    dup["method"] = "copy"
    base = df[np.isclose(df["lam"], 1.0)].copy()
    base["method"] = "orig"
    out = paired_compare(pd.concat([base, dup]), "reliability_index")
    assert out["note"].iloc[0] == "identical on every route"
    assert out["p_value"].iloc[0] == 1.0


def test_holm_is_monotone_and_bounded():
    df = pd.DataFrame({"p_value": [0.001, 0.02, 0.04, 0.5]})
    out = holm(df)
    assert (out["p_holm"] <= 1.0).all()
    assert (out["p_holm"] >= out["p_value"] - 1e-12).all()
    assert out["p_holm"].is_monotonic_increasing


def test_cliffs_delta_bounds():
    a = np.array([1.0, 2.0, 3.0])
    assert cliffs_delta(a, a + 10) == pytest.approx(-1.0)
    assert cliffs_delta(a + 10, a) == pytest.approx(1.0)
    assert abs(cliffs_delta(a, a)) < 1e-9


def test_lambda_trend_detects_monotonicity():
    out = lambda_trend_test(_fake_sweep(), "reliability_index")
    assert out["rho"] < -0.9          # reliability rises as lambda falls
    assert out["p_value"] < 1e-6

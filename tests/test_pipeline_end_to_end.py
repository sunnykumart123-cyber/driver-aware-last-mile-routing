"""End-to-end integration: processed routes -> Stage-1 -> solver -> logged rows.

This is the Phase-6 definition of done in miniature (§9): the whole chain runs on
a small set of routes written to a temporary ``data/processed`` tree, both
solvers agree on the small instances, and the result rows land in a parquet log
with the schema §7 specifies.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.config import BehaviorConfig, Config, Paths, SolverConfig, UncertaintyConfig
from core.io import load_results, write_instance
from data.splits import held_out_metro_split, model_split
from models.uncertainty.quantile_model import QuantileUncertaintyModel, fit_historical_stats
from pipeline.run_pipeline import PipelineRunner
from tests.conftest import TOY4_SERVICE, TOY4_TT, make_instance

TRAIN_METROS = ("Chicago", "Seattle", "Boston", "Los Angeles")
HELD_OUT = "Austin"


def _route(k: int, metro: str):
    rng = np.random.default_rng(100 + k)
    tt = TOY4_TT * (1.0 + 0.15 * rng.standard_normal())
    np.fill_diagonal(tt, 0.0)
    seq = [0, 1, 2, 3] if k % 3 else [0, 1, 3, 2]
    return make_instance(
        tt, TOY4_SERVICE,
        zones=["DEPOT", "Z1", "Z1", "Z2"],
        tw_end=[np.inf, np.inf, np.inf, 36.0 + (k % 5)],
        route_id=f"{metro[:2].lower()}_{k:03d}",
        metro=metro,
        actual_sequence=seq,
    )


@pytest.fixture
def processed_tree(tmp_path):
    root = tmp_path / "processed"
    for m in TRAIN_METROS:
        for k in range(6):
            write_instance(_route(k, m), root)
    for k in range(4):
        write_instance(_route(50 + k, HELD_OUT), root)
    return root


@pytest.fixture
def artifacts(tmp_path, processed_tree):
    """Train miniature Stage-1 artefacts for the held-out-Austin fold."""
    import torch

    from models.behavior.dataset import collate, encode_many
    from models.behavior.pointer_net import PointerNet

    art = tmp_path / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    train, _ = model_split(processed_tree, HELD_OUT)
    instances = train.instances()
    assert instances, "the split fixture produced no training routes"

    torch.manual_seed(0)
    cfg = BehaviorConfig(embed_dim=16, hidden_dim=16, n_layers=1, lr=5e-3)
    model = PointerNet(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    batch = collate(encode_many(instances))
    for _ in range(40):
        opt.zero_grad()
        loss = model.loss(batch)
        loss.backward()
        opt.step()
    model.save(art / "pointer_net_ho_austin.pt")

    unc_cfg = UncertaintyConfig(n_scenarios=16, eps=0.2)
    stats = fit_historical_stats(instances)
    QuantileUncertaintyModel.fit(instances, unc_cfg, stats).save(
        art / "quantile_model_ho_austin.pkl"
    )
    return art


def _config(tmp_path, artifacts) -> Config:
    cfg = Config()
    cfg.held_out_metro = HELD_OUT
    cfg.behavior.model = "pointer"
    cfg.uncertainty = UncertaintyConfig(n_scenarios=16, eps=0.2)
    cfg.solver = SolverConfig(time_limit_s=10.0, alns_iterations=300,
                              alns_segment=50, exact_backend="pulp", seed=3)
    cfg.paths = Paths(root=tmp_path, raw=tmp_path / "raw",
                      processed=tmp_path / "processed", artifacts=artifacts,
                      results=tmp_path / "results").ensure()
    cfg.tag = "integration"
    return cfg


def test_stage1_artifacts_load_and_produce_costs(tmp_path, processed_tree, artifacts):
    cfg = _config(tmp_path, artifacts)
    runner = PipelineRunner.from_artifacts(cfg)
    assert runner.uncertainty_model is not None
    inst = held_out_metro_split(processed_tree, HELD_OUT).instances()[0]
    obj = runner.build_objective(inst, lam=0.5)
    assert np.isfinite(obj.behavior_cost).all()
    assert obj.evaluator.amb.n_scenarios == 16
    assert np.isfinite(obj(inst.actual_sequence))


def test_pipeline_runs_and_logs_rows(tmp_path, processed_tree, artifacts):
    cfg = _config(tmp_path, artifacts)
    runner = PipelineRunner.from_artifacts(cfg)
    test = held_out_metro_split(processed_tree, HELD_OUT)
    df = runner.run_split(
        test.iter_instances(), method="alns", lams=[1.0, 0.5, 0.0],
        fold="ho_austin", log_path=cfg.paths.experiment_log,
    )
    assert len(df) == len(test) * 3
    assert "error" not in df.columns
    assert df["objective"].notna().all()
    assert df["reliability_index"].between(0.0, 1.0).all()

    logged = load_results(cfg.paths.experiment_log)
    assert len(logged) == len(df)
    assert set(logged["lam"]) == {1.0, 0.5, 0.0}


def test_solver_choice_by_instance_size(tmp_path, processed_tree, artifacts):
    cfg = _config(tmp_path, artifacts)
    runner = PipelineRunner.from_artifacts(cfg)
    inst = held_out_metro_split(processed_tree, HELD_OUT).instances()[0]
    assert runner.choose_solver(inst, "auto") == "exact"      # n = 4
    assert runner.choose_solver(inst, "alns") == "alns"


def test_alns_and_exact_agree_on_the_small_instances(tmp_path, processed_tree,
                                                     artifacts):
    cfg = _config(tmp_path, artifacts)
    runner = PipelineRunner.from_artifacts(cfg)
    for inst in held_out_metro_split(processed_tree, HELD_OUT).instances():
        obj = runner.build_objective(inst, lam=0.5)
        exact = runner.solve(inst, obj, "exact")
        alns = runner.solve(inst, obj, "alns")
        assert alns.objective == pytest.approx(exact.objective, abs=1e-5)


def test_lower_lambda_never_worsens_reliability_on_average(tmp_path, processed_tree,
                                                           artifacts):
    """The qualitative claim RQ2 tests: less imitation weight buys reliability."""
    cfg = _config(tmp_path, artifacts)
    runner = PipelineRunner.from_artifacts(cfg)
    insts = held_out_metro_split(processed_tree, HELD_OUT).instances()
    means = {}
    for lam in (1.0, 0.0):
        vals = []
        for inst in insts:
            obj = runner.build_objective(inst, lam=lam)
            sol = runner.solve(inst, obj, "exact")
            vals.append(runner.ctx.independent_evaluator(inst)
                        .evaluate(sol.order).prob_on_time)
        means[lam] = float(np.mean(vals))
    assert means[0.0] >= means[1.0] - 1e-9


def test_official_eval_access_is_recorded(tmp_path, processed_tree):
    from data.splits import official_eval_split

    paths = Paths(results=tmp_path / "results").ensure()
    official_eval_split(processed_tree, paths, reason="unit test")
    record = (tmp_path / "results" / "official_eval_uses.jsonl").read_text()
    assert "unit test" in record
    assert record.count("\n") == 1

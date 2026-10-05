"""Shared plumbing for the experiment scripts.

Keeps one definition of "how a fold is set up" and "how a baseline becomes a
result row", so ``ablations.py``, ``lambda_sweep_rq2.py`` and
``sensitivity_rq3.py`` cannot drift apart in how they log.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from core.config import Config, Paths, UncertaintyConfig
from core.io import append_results, metro_slug
from core.logging_utils import get_logger
from core.objective import CombinedObjective
from core.types import Instance
from data.splits import held_out_metro_split, model_split
from eval.metrics import evaluate_solution
from models.behavior.cost import UniformBehavior, build_behavior_cost
from models.uncertainty.ambiguity_set import AmbiguitySet
from pipeline.run_pipeline import PipelineRunner
from solvers.reliability import ReliabilityEvaluator

log = get_logger(__name__)


def base_config(
    held_out_metro: str,
    tag: str,
    behavior_model: str = "pointer",
    eps: float = 0.10,
    delta: float = 0.0,
    geometry: str = "wasserstein",
    scenarios: int = 64,
    time_limit: float = 30.0,
    iterations: int = 2000,
) -> Config:
    cfg = Config()
    cfg.paths = Paths().ensure()
    cfg.held_out_metro = held_out_metro
    cfg.tag = tag
    cfg.behavior.model = behavior_model
    cfg.uncertainty = UncertaintyConfig(
        eps=eps, delta=delta, geometry=geometry, n_scenarios=scenarios
    )
    cfg.solver.time_limit_s = time_limit
    cfg.solver.alns_iterations = iterations
    return cfg


def fold_instances(processed: str | Path, metro: str, on: str, limit: int | None,
                   max_stops: int | None = None) -> list[Instance]:
    train, val = model_split(processed, metro)
    test = held_out_metro_split(processed, metro)
    split = {"train": train, "val": val, "test": test}[on]
    insts = split.instances(limit)
    if max_stops is not None:
        insts = [i for i in insts if i.n <= max_stops]
    return insts


def make_runner(cfg: Config) -> PipelineRunner:
    return PipelineRunner.from_artifacts(cfg, tag=f"ho_{metro_slug(cfg.held_out_metro)}")


def objective_for(
    runner: PipelineRunner,
    inst: Instance,
    lam: float,
    behavior: str | None = None,
    uncertainty_override: UncertaintyConfig | None = None,
) -> CombinedObjective:
    """Build an objective with optional ablation overrides."""
    unc = uncertainty_override or runner.cfg.uncertainty
    if behavior == "uniform":
        cost = build_behavior_cost(UniformBehavior(), inst)
    else:
        cost = build_behavior_cost(runner.behavior_model, inst)
    if runner.uncertainty_model is None:
        amb = AmbiguitySet.deterministic(inst, unc)
    else:
        amb = AmbiguitySet.from_model(
            inst, runner.uncertainty_model, unc, np.random.default_rng(unc.seed)
        )
    return CombinedObjective(inst=inst, behavior_cost=cost,
                             evaluator=ReliabilityEvaluator(inst, amb), lam=lam)


def log_rows(rows: Sequence[dict], cfg: Config) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if not df.empty:
        append_results(df.to_dict("records"), cfg.paths.experiment_log)
        log.info("appended %d rows to %s", len(df), cfg.paths.experiment_log)
    return df


def evaluate_and_row(runner: PipelineRunner, inst: Instance, sol, obj,
                     method: str, fold: str, extra: dict | None = None) -> dict:
    return evaluate_solution(inst, sol, obj, runner.ctx, method, fold, extra)

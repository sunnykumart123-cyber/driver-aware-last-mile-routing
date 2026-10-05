"""Metrics (§7) and the canonical result-row schema.

Every experiment in this repository reports through :func:`evaluate_solution`,
so ``results/experiment_log.parquet`` has one schema and the thesis tables can
be built by grouping it.

The reliability index is measured on an **independent** scenario sample: the
optimizer sees ``P_hat`` (``uncertainty.seed``), the evaluator draws a fresh set
(``seed + EVAL_SEED_OFFSET``).  Scoring a route on the very scenarios it was
optimised against would report the in-sample optimism instead of the guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from core.config import Config, UncertaintyConfig
from core.logging_utils import get_logger
from core.objective import CombinedObjective, deterministic_lateness
from core.types import Instance, Solution
from eval.erp_official import (
    ScorerUnavailable,
    erp_score,
    kendall_tau,
    scorer_available,
    sequence_deviation,
)
from models.uncertainty.ambiguity_set import AmbiguitySet
from solvers.reliability import ReliabilityEvaluator

log = get_logger(__name__)

EVAL_SEED_OFFSET = 104_729          # a prime, so the two streams cannot coincide
EVAL_SCENARIOS = 512

RESULT_COLUMNS = [
    "method", "lam", "metro", "fold", "route_id", "n_stops", "n_zones",
    "n_windows", "erp", "erp_available", "seq_dev", "kendall_tau",
    "reliability_index", "expected_lateness", "expected_duration",
    "total_eta", "max_eta", "behavior_cost", "risk_cost", "objective",
    "route_travel_time", "deterministic_lateness", "solver", "runtime_s",
    "lower_bound", "optimality_gap", "delta", "eps", "geometry",
    "behavior_model", "degraded", "tag",
]


@dataclass
class EvalContext:
    """Held-fixed pieces of an evaluation: the independent scenario draw."""

    cfg: Config
    model_uncertainty: object | None = None
    n_eval_scenarios: int = EVAL_SCENARIOS

    def independent_evaluator(self, inst: Instance) -> ReliabilityEvaluator:
        u = self.cfg.uncertainty
        eval_cfg = UncertaintyConfig(
            **{**u.__dict__,
               "seed": u.seed + EVAL_SEED_OFFSET,
               "n_scenarios": self.n_eval_scenarios}
        )
        if self.model_uncertainty is None:
            amb = AmbiguitySet.deterministic(inst, eval_cfg)
        else:
            amb = AmbiguitySet.from_model(
                inst, self.model_uncertainty, eval_cfg,
                np.random.default_rng(eval_cfg.seed),
            )
        # The *measurement* uses the empirical distribution itself: delta only
        # belongs in the optimizer's ambiguity set, never in the yardstick.
        return ReliabilityEvaluator(inst, amb.with_radius(0.0))


def reliability_index(inst: Instance, order: Sequence[int],
                      ctx: EvalContext) -> tuple[float, float, float]:
    """``(Pr[no violation], expected lateness, expected duration)`` out of sample."""
    rep = ctx.independent_evaluator(inst).evaluate(order)
    return rep.prob_on_time, rep.expected_lateness, rep.expected_duration


def evaluate_solution(
    inst: Instance,
    sol: Solution,
    obj: CombinedObjective,
    ctx: EvalContext,
    method: str,
    fold: str = "",
    extra: dict | None = None,
) -> dict:
    """One result row, ready for ``results/experiment_log.parquet``."""
    order = sol.order
    p_on, lateness, duration = reliability_index(inst, order, ctx)

    erp, erp_ok = float("nan"), False
    if inst.actual_sequence is not None and scorer_available():
        try:
            erp, erp_ok = erp_score(inst, order), True
        except (ScorerUnavailable, ValueError) as exc:  # pragma: no cover
            log.debug("ERP unavailable for %s: %s", inst.route_id, exc)

    terms = obj.terms(order, full_report=True)
    tt = inst.travel_time
    travel = float(sum(tt[a, b] for a, b in zip(order, order[1:] + [order[0]])))
    u = ctx.cfg.uncertainty

    row = {
        "method": method,
        "lam": float(obj.lam),
        "metro": inst.metro,
        "fold": fold,
        "route_id": inst.route_id,
        "n_stops": inst.n,
        "n_zones": int(inst.meta.get("n_zones", len(inst.zone_groups()))),
        "n_windows": int(np.isfinite(inst.tw_end).sum()),
        "erp": erp,
        "erp_available": erp_ok,
        "seq_dev": sequence_deviation(inst, order),
        "kendall_tau": kendall_tau(inst, order),
        "reliability_index": p_on,
        "expected_lateness": lateness,
        "expected_duration": duration,
        "total_eta": float(terms.report.total_eta) if terms.report else float("nan"),
        "max_eta": float(terms.report.max_eta) if terms.report else float("nan"),
        "behavior_cost": terms.behavior,
        "risk_cost": terms.risk,
        "objective": terms.total,
        "route_travel_time": travel,
        "deterministic_lateness": deterministic_lateness(inst, order, ctx.cfg.theta),
        "solver": sol.solver,
        "runtime_s": sol.runtime_s,
        "lower_bound": sol.lower_bound,
        "optimality_gap": sol.gap,
        "delta": u.delta,
        "eps": u.eps,
        "geometry": u.geometry,
        "behavior_model": ctx.cfg.behavior.model,
        "degraded": bool(
            u.degrade_features or u.degrade_quantiles or u.degrade_train_fraction < 1.0
        ),
        "tag": ctx.cfg.tag,
    }
    if extra:
        row.update(extra)
    return row


# ------------------------------------------------------------- aggregate views
def price_of_robustness(df, metric: str = "erp", reference_lam: float = 1.0,
                        group: Sequence[str] = ("metro", "fold", "route_id")) -> "object":
    """``metric(lambda) - metric(lambda = 1)`` per route, the §7 definition.

    Returns a dataframe with one row per (group, lambda) carrying the paired
    difference, so the reliability gain and the ERP degradation can be plotted
    against each other.
    """
    import pandas as pd

    df = pd.DataFrame(df)
    ref = df[np.isclose(df["lam"], reference_lam)]
    keys = list(group)
    ref = ref.groupby(keys, as_index=False)[[metric, "reliability_index"]].mean()
    ref = ref.rename(columns={metric: f"{metric}_ref",
                              "reliability_index": "reliability_ref"})
    merged = df.merge(ref, on=keys, how="inner")
    merged[f"d_{metric}"] = merged[metric] - merged[f"{metric}_ref"]
    merged["d_reliability"] = merged["reliability_index"] - merged["reliability_ref"]
    return merged


def summarise(df, by: Sequence[str] = ("method", "lam"),
              metrics: Sequence[str] = ("erp", "seq_dev", "reliability_index",
                                        "expected_lateness", "objective",
                                        "runtime_s")) -> "object":
    import pandas as pd

    df = pd.DataFrame(df)
    present = [m for m in metrics if m in df.columns]
    g = df.groupby(list(by))[present]
    out = g.agg(["mean", "std", "count"])
    out.columns = ["_".join(c) for c in out.columns]
    return out.reset_index()

"""The combined objective every solver optimises.

    min_x  lambda * sum_{(i,j)} c^beh_ij x_ij  +  (1 - lambda) * sum_i eta_i

``c^beh_ij = -log p_ij`` comes from Stage-1a; ``eta_i`` from the DRO reliability
layer.  Both terms are divided by the number of stops so that a single
``lambda`` grid is comparable across routes of different length -- a positive
rescaling of each term, which leaves the argmin at ``lambda in {0, 1}``
unchanged and makes the interior sweep interpretable (see
``docs/FORMULATION.md``, "Scaling remark").

The closing arc back to the station is excluded from the behaviour term (the
Stage-1a decoder does not emit a depot-return probability) but is included in
route duration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from core.types import Instance, Solution
from solvers.reliability import ReliabilityEvaluator, RiskReport


@dataclass
class ObjectiveTerms:
    behavior: float
    risk: float
    total: float
    report: RiskReport | None = None


@dataclass
class CombinedObjective:
    """Callable objective bound to one instance."""

    inst: Instance
    behavior_cost: np.ndarray            # (n, n), -log p_ij
    evaluator: ReliabilityEvaluator
    lam: float = 0.5
    normalize: bool = True
    _scale: float = field(init=False, default=1.0)

    def __post_init__(self) -> None:
        self.behavior_cost = np.asarray(self.behavior_cost, dtype=float)
        if self.behavior_cost.shape != (self.inst.n, self.inst.n):
            raise ValueError("behavior_cost has the wrong shape")
        if not 0.0 <= self.lam <= 1.0:
            raise ValueError("lam must lie in [0, 1]")
        self._scale = float(max(self.inst.n - 1, 1)) if self.normalize else 1.0

    @property
    def scale(self) -> float:
        """Positive rescaling applied to both terms (see module docstring)."""
        return self._scale

    # -------------------------------------------------------------------- eval
    def behavior_term(self, order: Sequence[int]) -> float:
        order = np.asarray(order, dtype=int)
        return float(self.behavior_cost[order[:-1], order[1:]].sum()) / self._scale

    def risk_term(self, order: Sequence[int]) -> float:
        return self.evaluator.total_eta(order) / self._scale

    def terms(self, order: Sequence[int], full_report: bool = False) -> ObjectiveTerms:
        beh = self.behavior_term(order)
        if full_report:
            rep = self.evaluator.evaluate(order)
            risk = float(rep.eta.sum()) / self._scale
        else:
            rep, risk = None, self.risk_term(order)
        total = self.lam * beh + (1.0 - self.lam) * risk
        return ObjectiveTerms(behavior=beh, risk=risk, total=total, report=rep)

    def __call__(self, order: Sequence[int]) -> float:
        # Skip the risk evaluation entirely at lambda == 1 (pure imitation):
        # this is what makes the imitation baseline cheap.
        if self.lam >= 1.0:
            return self.behavior_term(order)
        if self.lam <= 0.0:
            return self.risk_term(order)
        return self.lam * self.behavior_term(order) + (1.0 - self.lam) * self.risk_term(order)

    # ------------------------------------------------------------------ helper
    def to_solution(self, order: Sequence[int], solver: str, runtime_s: float = float("nan"),
                    lower_bound: float = float("nan"), **meta) -> Solution:
        t = self.terms(order, full_report=True)
        return Solution(
            order=list(order),
            objective=t.total,
            behavior_cost=t.behavior,
            risk_cost=t.risk,
            solver=solver,
            runtime_s=runtime_s,
            lower_bound=lower_bound,
            meta={**(t.report.as_dict() if t.report else {}), **meta},
        )


def deterministic_lateness(inst: Instance, order: Sequence[int], theta: float = 1.0) -> float:
    """Soft-lateness objective of the *base* MILP (§3), planned times only."""
    order = list(order)
    st, tt, b = inst.service_time, inst.travel_time, inst.tw_end
    t = 0.0
    total = 0.0
    for a, nxt in zip(order, order[1:]):
        t += st[a] + tt[a, nxt]
        if np.isfinite(b[nxt]):
            total += theta * max(0.0, t - b[nxt])
    return total

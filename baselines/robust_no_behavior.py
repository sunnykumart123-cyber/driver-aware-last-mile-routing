"""Baseline 3: uncertainty-aware robust routing with no behaviour term (§7).

This is the Zhang et al. (2021) side of the literature gap: the full DRO
reliability layer, ``lambda = 0``, and a uniform behaviour cost so that nothing
about the observed driver sequences enters the objective.  A travel-time
tie-break is applied *inside* the reliability-indifferent region, because with
no behaviour term and no window pressure the objective is flat over a large set
of sequences and an arbitrary pick would make the baseline look worse than it is.
"""
from __future__ import annotations

from core.config import SolverConfig
from core.objective import CombinedObjective
from core.types import Instance, Solution
from models.behavior.cost import UniformBehavior, build_behavior_cost
from solvers.alns.alns_solver import solve_alns
from solvers.reliability import ReliabilityEvaluator


def build_robust_objective(inst: Instance, evaluator: ReliabilityEvaluator,
                           travel_tiebreak: float = 1e-3) -> CombinedObjective:
    """``lambda = 0`` with a uniform behaviour matrix.

    ``travel_tiebreak`` scales the travel-time matrix into the behaviour slot at
    a negligible weight; it breaks ties without competing with the risk term.
    """
    cost = build_behavior_cost(UniformBehavior(), inst)
    cost = cost + travel_tiebreak * inst.travel_time
    return CombinedObjective(inst=inst, behavior_cost=cost,
                             evaluator=evaluator, lam=travel_tiebreak)


def solve_robust_no_behavior(inst: Instance, evaluator: ReliabilityEvaluator,
                             cfg: SolverConfig | None = None) -> tuple[Solution, CombinedObjective]:
    obj = build_robust_objective(inst, evaluator)
    res = solve_alns(inst, obj, cfg or SolverConfig())
    res.solution.solver = "robust_no_behavior"
    return res.solution, obj

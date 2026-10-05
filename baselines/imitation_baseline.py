"""Baseline 2: pure imitation, ``lambda = 1`` (§7).

Three flavours, because the literature this thesis positions against uses
different decoders:

``pointer_greedy``  greedy rollout of the Stage-1a pointer network -- the
                    Mo et al. (2023) style baseline;
``gbm_greedy``      greedy decoding of the pairwise transition scorer -- the
                    da Costa et al. (2023) style baseline;
``alns_lam1``       the thesis's own solver run at ``lambda = 1``, which
                    isolates the effect of the reliability term by holding the
                    search procedure fixed.

The Phase-3 gate of §9 is checked against ``pointer_greedy`` / ``gbm_greedy``:
their ERP must land within a defensible tolerance of the published numbers
before any later phase is built on top.
"""
from __future__ import annotations

import time

from core.config import SolverConfig
from core.logging_utils import get_logger
from core.objective import CombinedObjective
from core.types import Instance, Solution
from models.behavior.dataset import encode_instance
from solvers.alns.alns_solver import solve_alns

log = get_logger(__name__)


def pointer_greedy(inst: Instance, model) -> Solution:
    t0 = time.perf_counter()
    order = model.greedy_order(encode_instance(inst))
    return Solution(order=order, solver="pointer_greedy",
                    runtime_s=time.perf_counter() - t0)


def gbm_greedy(inst: Instance, model) -> Solution:
    t0 = time.perf_counter()
    order = model.greedy_order(inst)
    return Solution(order=order, solver="gbm_greedy",
                    runtime_s=time.perf_counter() - t0)


def alns_imitation(inst: Instance, obj: CombinedObjective,
                   cfg: SolverConfig | None = None) -> Solution:
    """Search the behaviour objective only; ``obj.lam`` is forced to 1."""
    if obj.lam != 1.0:
        obj = CombinedObjective(
            inst=obj.inst, behavior_cost=obj.behavior_cost,
            evaluator=obj.evaluator, lam=1.0, normalize=obj.normalize,
        )
    res = solve_alns(inst, obj, cfg or SolverConfig())
    res.solution.solver = "alns_lam1"
    return res.solution


def solve_imitation(inst: Instance, model, obj: CombinedObjective | None = None,
                    flavour: str = "pointer_greedy",
                    cfg: SolverConfig | None = None) -> Solution:
    if flavour == "pointer_greedy":
        return pointer_greedy(inst, model)
    if flavour == "gbm_greedy":
        return gbm_greedy(inst, model)
    if flavour == "alns_lam1":
        if obj is None:
            raise ValueError("alns_lam1 needs a CombinedObjective")
        return alns_imitation(inst, obj, cfg)
    raise ValueError(f"unknown imitation flavour {flavour!r}")

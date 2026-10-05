"""One interface for turning any Stage-1a model into ``c^beh_ij``.

``build_behavior_cost`` is the only place the rest of the codebase learns about
behaviour-model types, which is what lets ``experiments/ablations.py`` swap in
the uniform cost (the "no-behaviour" ablation) without touching the solvers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from core.config import BehaviorConfig
from core.types import DEPOT, Instance


class BehaviorModel(Protocol):
    def transition_matrix(self, inst: Instance) -> np.ndarray: ...
    def behavior_cost_matrix(self, inst: Instance) -> np.ndarray: ...


class UniformBehavior:
    """The no-behaviour ablation: every feasible arc is equally driver-like.

    ``c^beh_ij`` is then a constant on all arcs, so the behaviour term becomes
    route-length-invariant and lambda controls nothing but a constant offset --
    exactly the counterfactual §7 asks for.
    """

    def __init__(self, prob_floor: float = 1e-6):
        self.prob_floor = prob_floor

    def transition_matrix(self, inst: Instance) -> np.ndarray:
        n = inst.n
        p = np.full((n, n), 1.0 / max(n - 2, 1), dtype=float)
        np.fill_diagonal(p, 0.0)
        p[:, DEPOT] = 0.0
        return np.maximum(p, self.prob_floor)

    def behavior_cost_matrix(self, inst: Instance) -> np.ndarray:
        return -np.log(self.transition_matrix(inst))


def build_behavior_cost(model: BehaviorModel, inst: Instance,
                        clip: float | None = 30.0) -> np.ndarray:
    """``-log p_ij`` with the diagonal and depot column made non-attractive.

    ``clip`` bounds the cost of a very unlikely arc so that a single
    near-zero probability cannot dominate the whole objective (the MILP needs
    finite coefficients; the pointer's ``prob_floor`` already guarantees
    finiteness, this guards the scale).
    """
    c = np.asarray(model.behavior_cost_matrix(inst), dtype=float)
    if clip is not None:
        c = np.minimum(c, clip)
    np.fill_diagonal(c, 0.0)
    return c


def load_behavior_model(cfg: BehaviorConfig, artifacts: str | Path,
                        tag: str = "") -> BehaviorModel:
    """Load the Stage-1a artefact named by ``cfg.model``."""
    artifacts = Path(artifacts)
    suffix = f"_{tag}" if tag else ""
    if cfg.model == "uniform":
        return UniformBehavior(cfg.prob_floor)
    if cfg.model == "pointer":
        from models.behavior.pointer_net import PointerNet

        return PointerNet.load(artifacts / f"pointer_net{suffix}.pt")
    if cfg.model == "gbm":
        from models.behavior.gbm_transition import GBMTransitionModel

        return GBMTransitionModel.load(artifacts / f"gbm_transition{suffix}.pkl")
    raise ValueError(f"unknown behaviour model {cfg.model!r}")

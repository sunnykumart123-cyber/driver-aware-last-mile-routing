"""Acceptance criteria for the ALNS.

Simulated annealing is the default (§5b).  The starting temperature is set as a
fraction of the incumbent objective so the schedule transfers across routes of
very different scale, and the cooling rate is tuned on a validation subset by
:func:`tune_cooling`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class AcceptanceCriterion:
    def accept(self, current: float, candidate: float,
               rng: np.random.Generator) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def step(self) -> None:
        pass


@dataclass
class HillClimb(AcceptanceCriterion):
    def accept(self, current: float, candidate: float, rng) -> bool:
        return candidate < current - 1e-12


@dataclass
class SimulatedAnnealing(AcceptanceCriterion):
    start_temp: float
    cooling: float = 0.9995
    min_temp: float = 1e-8
    temp: float = float("nan")

    def __post_init__(self) -> None:
        self.temp = max(float(self.start_temp), self.min_temp)

    def accept(self, current: float, candidate: float, rng) -> bool:
        if candidate <= current:
            return True
        delta = candidate - current
        return bool(rng.random() < math.exp(-delta / max(self.temp, self.min_temp)))

    def step(self) -> None:
        self.temp = max(self.temp * self.cooling, self.min_temp)


@dataclass
class RecordToRecordTravel(AcceptanceCriterion):
    """Accept anything within ``threshold`` of the best found so far."""

    best: float
    threshold: float

    def accept(self, current: float, candidate: float, rng) -> bool:
        if candidate < self.best:
            self.best = candidate
        return candidate <= self.best + self.threshold


def build_acceptance(initial_objective: float, cfg) -> AcceptanceCriterion:
    start = max(abs(initial_objective) * cfg.sa_start_temp_frac, 1e-6)
    return SimulatedAnnealing(start_temp=start, cooling=cfg.sa_cooling)


def tune_cooling(run_fn, candidates=(0.999, 0.9995, 0.9999), **kw) -> tuple[float, dict]:
    """Pick the cooling rate that gives the best mean objective.

    ``run_fn(cooling) -> float`` is expected to solve a validation subset and
    return the mean objective; the caller supplies it so this module stays free
    of pipeline imports.
    """
    scores = {c: float(run_fn(c, **kw)) for c in candidates}
    best = min(scores, key=scores.get)
    return best, scores

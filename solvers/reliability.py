"""The reliability layer: worst-case CVaR of time-window violation.

Formulation implemented here (derivation in ``docs/FORMULATION.md``)
--------------------------------------------------------------------
For a fixed sequence, the arrival time at the stop in position ``p`` is a linear
functional of the primitive uncertainties::

    T_p(xi) = <w_p, xi>,      w_p in {0,1}^d,   ||w_p||_0 = 2p - 1

(``p`` travel arcs and ``p-1`` service times on the prefix).  The spec's
chance constraint is enforced through its CVaR safeguard,

    CVaR_{1-eps}[ T_p - b ] <= eta_p,    eta_p >= 0,

which implies ``Pr[T_p <= b] >= 1 - eps`` whenever ``eta_p = 0`` because
``VaR <= CVaR``.  Under a Wasserstein-1 ball of radius ``delta`` around the
empirical ``P_hat``, the worst-case CVaR of a Lipschitz loss has the closed form
(Mohajerin Esfahani & Kuhn 2018, Thm. 6.3; the reformulation used by Zhang et
al. 2021):

    sup_{P in B_delta(P_hat)} CVaR_{1-eps}[<w,xi> - b]
        = CVaR^{P_hat}_{1-eps}[<w,xi> - b] + (delta / eps) * ||w||_*

The first term is the sample CVaR, computed exactly from the scenario set by
sorting; the second is a deterministic inflation that grows with route depth,
which is what penalises long fragile prefixes.  ``ground_norm`` selects the dual
norm: ``linf`` -> ``||w||_1 = 2p-1``, ``l1`` -> ``1``, ``l2`` -> ``sqrt(2p-1)``.

Alternative geometries (``box``, ``ellipsoidal``) are implemented for the §7
ablation on ambiguity-set geometry.

Everything is vectorised over scenarios so that one route evaluation is a single
``cumsum`` plus a sort -- this is the fast per-insertion evaluation the ALNS
needs, and it never re-solves an optimisation problem.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import norm

from core.types import Instance
from models.uncertainty.ambiguity_set import AmbiguitySet


def sample_cvar(losses: np.ndarray, eps: float, axis: int = 0) -> np.ndarray:
    """Exact sample CVaR at level ``1-eps`` (mean of the worst ``eps`` mass).

    Computes ``min_tau tau + (1/(eps*S)) * sum (L_s - tau)_+`` in closed form:
    the minimiser is the empirical VaR, so

        CVaR = VaR + (1/(eps*S)) * sum_s (L_s - VaR)_+ .
    """
    losses = np.asarray(losses, dtype=float)
    s = losses.shape[axis]
    if not 0.0 < eps <= 1.0:
        raise ValueError("eps must lie in (0, 1]")
    if eps >= 1.0:
        return losses.mean(axis=axis)
    k = int(np.ceil((1.0 - eps) * s)) - 1        # 0-based index of the VaR
    k = min(max(k, 0), s - 1)
    var = np.partition(losses, k, axis=axis).take(k, axis=axis)
    excess = np.clip(losses - np.expand_dims(var, axis), 0.0, None).sum(axis=axis)
    return var + excess / (eps * s)


@dataclass
class RiskReport:
    """Per-route output of the reliability layer."""

    eta: np.ndarray            # (n,) worst-case CVaR surplus per position
    total_eta: float           # sum of eta -- the (1-lambda) term of the objective
    max_eta: float
    prob_on_time: float        # Monte-Carlo Pr[no window violation on the route]
    expected_lateness: float   # mean total lateness over scenarios, minutes
    expected_duration: float   # mean route duration, minutes
    n_windows: int             # stops that actually carry a finite window

    def as_dict(self) -> dict:
        return {
            "total_eta": self.total_eta,
            "max_eta": self.max_eta,
            "prob_on_time": self.prob_on_time,
            "expected_lateness": self.expected_lateness,
            "expected_duration": self.expected_duration,
            "n_windows": self.n_windows,
        }


class ReliabilityEvaluator:
    """Evaluates the DRO reliability layer for arbitrary sequences of one route.

    The instance's scenario arrays are held once; every ``evaluate`` call is
    O(S * n) work plus one sort per position with a finite window.
    """

    def __init__(self, inst: Instance, amb: AmbiguitySet):
        if amb.n != inst.n:
            raise ValueError("ambiguity set and instance disagree on n")
        self.inst = inst
        self.amb = amb
        self.b = inst.tw_end                       # (n,) may contain +inf
        self.has_window = np.isfinite(self.b)
        self.n_windows = int(self.has_window.sum())

    # ---------------------------------------------------------------- arrivals
    def arrivals(self, order: Sequence[int]) -> np.ndarray:
        """``(S, n)`` arrival time at each *position* of ``order``.

        Position 0 is the depot with arrival 0; position ``p`` accumulates the
        service times of positions ``0..p-1`` and the travel times along the
        prefix arcs.
        """
        order = np.asarray(order, dtype=int)
        frm, to = order[:-1], order[1:]
        legs = self.amb.service[:, frm] + self.amb.travel[:, frm, to]   # (S, n-1)
        out = np.zeros((self.amb.n_scenarios, len(order)), dtype=np.float64)
        np.cumsum(legs, axis=1, out=out[:, 1:])
        return out

    # -------------------------------------------------------------------- risk
    def dual_weights(self, n_components: np.ndarray) -> np.ndarray:
        """``||w||_*`` for an array of prefix-component counts."""
        n_comp = np.maximum(np.asarray(n_components, dtype=float), 1.0)
        gn = self.amb.ground_norm
        if gn == "linf":
            return n_comp
        if gn == "l1":
            return np.ones_like(n_comp)
        if gn == "l2":
            return np.sqrt(n_comp)
        raise ValueError(f"unknown ground_norm {gn!r}")

    def risk_array(
        self,
        arrival: np.ndarray,
        b_vals: np.ndarray,
        n_components: np.ndarray,
    ) -> np.ndarray:
        """Vectorised worst-case CVaR surplus.

        ``arrival`` has the scenario axis first and any trailing shape;
        ``b_vals`` and ``n_components`` broadcast against that trailing shape.
        This is the single implementation of the geometry -- :meth:`eta`,
        :meth:`risk_at` and the ALNS's incremental evaluation all go through it,
        so they cannot disagree.
        """
        arrival = np.asarray(arrival, dtype=float)
        b_vals = np.asarray(b_vals, dtype=float)
        n_comp = np.asarray(n_components, dtype=float)
        geom = self.amb.geometry
        eps, delta = self.amb.eps, self.amb.delta

        # Stops without a window carry b = +inf; substitute a finite sentinel so
        # the arithmetic stays defined, then zero those entries at the end.
        finite = np.isfinite(b_vals)
        b_safe = np.where(finite, b_vals, 0.0)

        if geom == "wasserstein":
            core = sample_cvar(arrival - b_safe, eps, axis=0)
            out = core + (delta / eps) * self.dual_weights(n_comp)
        elif geom == "box":
            # Independent per-component upper quantiles: the most conservative
            # of the three geometries, and the cheapest.
            out = np.quantile(arrival, 1.0 - eps, axis=0) - b_safe + delta * n_comp
        elif geom == "ellipsoidal":
            mu = arrival.mean(axis=0)
            sd = arrival.std(axis=0, ddof=1) if arrival.shape[0] > 1 else 0.0
            out = (
                mu + float(norm.ppf(1.0 - eps)) * sd - b_safe
                + delta * np.sqrt(np.maximum(n_comp, 1.0))
            )
        else:
            raise ValueError(f"unknown geometry {geom!r}")

        return np.where(finite, np.maximum(out, 0.0), 0.0)

    def eta(self, order: Sequence[int]) -> np.ndarray:
        """Worst-case CVaR surplus ``eta_p`` for every position of ``order``."""
        order = np.asarray(order, dtype=int)
        arr = self.arrivals(order)
        b = self.b[order]
        mask = np.isfinite(b)
        eta = np.zeros(len(order), dtype=float)
        if not mask.any():
            return eta
        pos = np.arange(len(order))
        eta[mask] = self.risk_array(
            arr[:, mask], b[mask], np.maximum(2 * pos[mask] - 1, 1)
        )
        return eta

    def risk_at(self, arrival: np.ndarray, b_val: float, n_components: int) -> float:
        """Worst-case CVaR surplus for one stop, given its arrival samples.

        This is the scalar form of :meth:`eta` and is the primitive the ALNS
        uses for incremental insertion evaluation -- same geometry, same radius,
        no optimisation problem solved.
        """
        if not np.isfinite(b_val):
            return 0.0
        return float(
            self.risk_array(np.asarray(arrival)[:, None],
                            np.array([b_val]),
                            np.array([max(int(n_components), 1)]))[0]
        )

    # ------------------------------------------------------------------ report
    def evaluate(self, order: Sequence[int]) -> RiskReport:
        order = list(order)
        eta = self.eta(order)
        arr = self.arrivals(order)
        b = self.b[np.asarray(order, dtype=int)]
        mask = np.isfinite(b)

        if mask.any():
            late = np.clip(arr[:, mask] - b[mask], 0.0, None)
            prob_on_time = float((late.max(axis=1) <= 1e-9).mean())
            expected_lateness = float(late.sum(axis=1).mean())
        else:
            prob_on_time, expected_lateness = 1.0, 0.0

        closing = (
            self.amb.service[:, order[-1]]
            + self.amb.travel[:, order[-1], order[0]]
        )
        duration = float((arr[:, -1] + closing).mean())

        return RiskReport(
            eta=eta,
            total_eta=float(eta.sum()),
            max_eta=float(eta.max(initial=0.0)),
            prob_on_time=prob_on_time,
            expected_lateness=expected_lateness,
            expected_duration=duration,
            n_windows=self.n_windows,
        )

    # -------------------------------------------------- incremental evaluation
    def total_eta(self, order: Sequence[int]) -> float:
        return float(self.eta(order).sum())

    def prefix_feasible(self, order: Sequence[int]) -> bool:
        """True when the DRO chance constraint holds at every stop."""
        return bool(self.total_eta(order) <= 1e-9)

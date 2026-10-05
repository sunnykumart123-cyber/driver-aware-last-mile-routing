"""Ambiguity set construction: from Stage-1b quantiles to ``P_hat`` and delta.

For one instance the Stage-1b model gives, for every arc ``(i,j)`` and stop
``i``, a conditional quantile curve.  Inverting that curve at a fixed matrix of
uniforms (common random numbers, seeded) yields ``S`` empirical scenarios -- the
discrete centre ``P_hat`` of the Wasserstein ball used in §3 of the spec.

Two things matter for correctness downstream:

1. **Common random numbers.**  The same uniforms are reused for every candidate
   route, so differences in risk between two routes are not sampling noise.
2. **Radius calibration.**  ``delta`` is *not* a free constant: ``calibrate_delta``
   picks the smallest radius on a grid such that the nominal ``1-eps`` guarantee
   holds out-of-sample on held-out routes (§4b of the spec).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from core.config import UncertaintyConfig
from core.logging_utils import get_logger
from core.types import Instance

log = get_logger(__name__)


def invert_quantiles(
    q_levels: Sequence[float], q_values: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """Inverse-CDF sampling from a piecewise-linear quantile curve.

    ``q_values`` has shape ``(..., Q)`` and ``u`` shape ``(S, ...)``; the result
    has shape ``(S, ...)``.  Outside ``[q_levels[0], q_levels[-1]]`` the curve is
    extended with the slope of the outermost segment and clipped at zero, so the
    tails stay data-driven rather than assumed.
    """
    q = np.asarray(q_levels, dtype=float)
    v = np.asarray(q_values, dtype=float)
    flat_v = v.reshape(-1, v.shape[-1])                  # (M, Q)
    flat_u = np.asarray(u, dtype=float).reshape(u.shape[0], -1)   # (S, M)

    idx = np.clip(np.searchsorted(q, flat_u, side="right") - 1, 0, len(q) - 2)
    q_lo, q_hi = q[idx], q[idx + 1]
    m_idx = np.arange(flat_v.shape[0])[None, :]
    v_lo = flat_v[m_idx, idx]
    v_hi = flat_v[m_idx, idx + 1]
    w = (flat_u - q_lo) / np.maximum(q_hi - q_lo, 1e-12)
    out = v_lo + w * (v_hi - v_lo)
    return np.maximum(out, 0.0).reshape((u.shape[0],) + v.shape[:-1])


@dataclass
class AmbiguitySet:
    """Empirical centre ``P_hat`` for one instance plus the ball's parameters."""

    travel: np.ndarray        # (S, n, n) float32 scenario travel times, minutes
    service: np.ndarray       # (S, n)    float32 scenario service times, minutes
    eps: float
    delta: float
    ground_norm: str = "linf"
    geometry: str = "wasserstein"
    route_id: str = ""

    @property
    def n_scenarios(self) -> int:
        return self.travel.shape[0]

    @property
    def n(self) -> int:
        return self.travel.shape[1]

    def with_radius(self, delta: float) -> "AmbiguitySet":
        return AmbiguitySet(
            self.travel, self.service, self.eps, float(delta),
            self.ground_norm, self.geometry, self.route_id,
        )

    def dual_norm_weight(self, n_components: int) -> float:
        """``||w||_*`` for the 0/1 prefix indicator with ``n_components`` ones."""
        if self.ground_norm == "linf":     # dual is l1
            return float(n_components)
        if self.ground_norm == "l1":       # dual is l_inf
            return 1.0
        if self.ground_norm == "l2":
            return float(np.sqrt(n_components))
        raise ValueError(f"unknown ground_norm {self.ground_norm!r}")

    # ------------------------------------------------------------- factories
    @classmethod
    def from_model(
        cls,
        inst: Instance,
        model,                       # QuantileUncertaintyModel
        cfg: UncertaintyConfig,
        rng: np.random.Generator | None = None,
    ) -> "AmbiguitySet":
        rng = rng or np.random.default_rng(cfg.seed)
        tq = model.instance_travel_quantiles(inst)        # (n, n, Q)
        sq = model.instance_service_quantiles(inst)       # (n, Q)
        levels = model.quantiles
        s = cfg.n_scenarios
        u_t = rng.random((s, inst.n, inst.n))
        u_s = rng.random((s, inst.n))
        travel = invert_quantiles(levels, tq, u_t).astype(np.float32)
        service = invert_quantiles(levels, sq, u_s).astype(np.float32)
        for k in range(s):
            np.fill_diagonal(travel[k], 0.0)
        service[:, 0] = 0.0
        return cls(
            travel=travel,
            service=service,
            eps=cfg.eps,
            delta=cfg.delta,
            ground_norm=cfg.ground_norm,
            geometry=cfg.geometry,
            route_id=inst.route_id,
        )

    @classmethod
    def deterministic(cls, inst: Instance, cfg: UncertaintyConfig) -> "AmbiguitySet":
        """Degenerate set: a single scenario at the planned times.

        Used by the deterministic baselines and by the toy unit tests, where the
        DRO objective must collapse to the soft-lateness objective.
        """
        travel = inst.travel_time.astype(np.float32)[None, :, :].copy()
        service = inst.service_time.astype(np.float32)[None, :].copy()
        service[:, 0] = 0.0
        return cls(travel, service, cfg.eps, cfg.delta, cfg.ground_norm,
                   cfg.geometry, inst.route_id)


def build_ambiguity_sets(
    instances: Iterable[Instance],
    model,
    cfg: UncertaintyConfig,
) -> dict[str, AmbiguitySet]:
    rng = np.random.default_rng(cfg.seed)
    return {i.route_id: AmbiguitySet.from_model(i, model, cfg, rng) for i in instances}


# ----------------------------------------------------------- radius calibration
def calibrate_delta_from_coverage(
    instances: Sequence[Instance],
    model,
    cfg: UncertaintyConfig,
) -> tuple[float, dict]:
    """Calibrate the radius against the *measured* miscalibration of Stage-1b.

    Why not :func:`calibrate_delta`
    -------------------------------
    The obvious protocol -- declare routes reliable, then check them on a fresh
    scenario draw -- cannot detect a misspecified model here, for two reasons:

    * the fresh draw comes from the *same fitted model*, so it measures
      Monte-Carlo stability, not whether the model is right;
    * the predictive quantiles are anchored so their median equals the route's
      own supplied ``t_ij`` (see ``models/uncertainty/quantile_model.py``), so
      the one realisation available per route sits at the centre of its own
      predictive distribution by construction.  Per-route coverage is therefore
      not identifiable, and any such check returns ``delta = 0``.

    What *is* identifiable is the **spread**: across routes, does the fitted
    conditional quantile at level ``1-eps`` actually cover a ``1-eps`` fraction
    of observed times?  ``eval_uncertainty.py`` measures exactly that, and on
    this data the travel head is badly overconfident out of sample (nominal 95%
    covering ~75% on a held-out metro).

    This function turns that shortfall into a radius.  The DRO inflation adds
    ``(delta/eps) * ||w||_*`` to the CVaR, i.e. ``delta/eps`` per prefix
    component under the ``linf`` ground norm.  Setting ``delta = eps * m``, with
    ``m`` the per-component margin by which the nominal upper quantile
    *under-covers* out of sample, makes the worst-case bound carry exactly the
    measured deficit.
    """
    from models.uncertainty.quantile_model import build_training_tables

    tr, sv = build_training_tables(list(instances), model.stats)
    levels = list(model.quantiles)
    target_level = 1.0 - cfg.eps
    # The fitted level closest to 1 - eps is the one the bound leans on.
    k = int(np.argmin([abs(q - target_level) for q in levels]))

    report: dict = {"eps": cfg.eps, "level_used": levels[k], "heads": {}}
    margins = []
    for name, rows, predict in (
        ("travel", tr, model.predict_travel),
        ("service", sv, model.predict_service),
    ):
        if rows.empty:
            continue
        pred = predict(rows)[:, k]
        y = rows["target"].to_numpy(dtype=float)
        coverage = float(np.mean(y <= pred))
        # Per-component margin: how much higher the quantile would have to sit
        # to reach its nominal coverage on held-out data.
        needed = float(np.quantile(y - pred, target_level))
        margin = max(needed, 0.0)
        margins.append(margin)
        report["heads"][name] = {
            "n": int(len(y)),
            "empirical_coverage": coverage,
            "nominal": target_level,
            "coverage_gap": target_level - coverage,
            "margin_minutes": margin,
        }
        log.info(
            "%s head: nominal %.2f, empirical %.3f, margin %.3f min",
            name, target_level, coverage, margin,
        )

    margin = float(max(margins)) if margins else 0.0
    delta = float(cfg.eps * margin)
    report["margin_minutes"] = margin
    report["delta"] = delta
    log.info("coverage-calibrated delta = %.4g (margin %.3f min)", delta, margin)
    return delta, report


def calibrate_delta(
    instances: Sequence[Instance],
    model,
    cfg: UncertaintyConfig,
    grid: Sequence[float] = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
    holdout_scenarios: int = 256,
) -> tuple[float, list[dict]]:
    """Scenario-stability check -- **not** a misspecification test.

    .. warning::
       This returns ``delta = 0`` on this dataset and that result is not
       meaningful: the out-of-sample draw comes from the same fitted model, and
       the predictive quantiles are route-anchored, so nothing here can detect
       a wrong *spread*.  Use :func:`calibrate_delta_from_coverage` for the
       radius and keep this as a Monte-Carlo stability diagnostic.

    Smallest radius on ``grid`` whose nominal ``1-eps`` guarantee holds.

    Protocol: build ``P_hat`` with ``cfg.n_scenarios`` draws, solve nothing --
    just take each route's observed driver sequence sigma* as the route under
    test -- then measure realised on-time probability against an *independent*
    held-out scenario sample of size ``holdout_scenarios``.  A radius passes when
    the routes it declares reliable are empirically reliable at level ``1-eps``.
    """
    from solvers.reliability import ReliabilityEvaluator

    rng_fit = np.random.default_rng(cfg.seed)
    rng_test = np.random.default_rng(cfg.seed + 7919)
    test_cfg = UncertaintyConfig(**{**cfg.__dict__, "n_scenarios": holdout_scenarios})

    records: list[dict] = []
    for delta in grid:
        declared, ontime = 0, 0
        for inst in instances:
            order = inst.actual_sequence or list(range(inst.n))
            amb = AmbiguitySet.from_model(inst, model, cfg, rng_fit).with_radius(delta)
            rep = ReliabilityEvaluator(inst, amb).evaluate(order)
            if rep.total_eta <= 1e-6:      # declared reliable by the DRO layer
                declared += 1
                oos = AmbiguitySet.from_model(inst, model, test_cfg, rng_test)
                oos_rep = ReliabilityEvaluator(inst, oos.with_radius(0.0)).evaluate(order)
                ontime += int(oos_rep.prob_on_time >= 1.0 - cfg.eps)
        cov = ontime / declared if declared else float("nan")
        records.append({"delta": delta, "n_declared": declared, "oos_coverage": cov})
        log.info("delta=%.3g declared=%d oos_coverage=%.3f", delta, declared, cov)

    ok = [r for r in records if r["n_declared"] > 0 and r["oos_coverage"] >= 1.0 - cfg.eps]
    chosen = min((r["delta"] for r in ok), default=max(grid))
    log.info("calibrated delta = %.4g", chosen)
    return float(chosen), records

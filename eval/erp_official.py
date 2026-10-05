"""Wrapper around the **official** LMRRC scoring code.

§7 of the build spec is explicit: reuse the challenge's own scorer, do not
reimplement Edit-distance-with-Real-Penalty.  So this module does not contain a
scoring algorithm -- it locates the official one and calls it.

Install it once::

    git clone --depth 1 https://github.com/MIT-CAVE/rc-cli external/rc-cli

which puts the module at ``external/rc-cli/scoring/score.py`` -- one of the
locations searched below.  Alternatively drop a ``score.py`` defining
``score(actual, sub, cost_mat)`` anywhere under ``external/``, or point
``MTP_SCORER_PATH`` at the directory holding it.

Verified against ``MIT-CAVE/rc-cli@main`` (2026-10-01): ``scoring/score.py``
defines ``score(actual, sub, cost_mat, g=1000)``, returning
``seq_dev * erp_per_edit``, where ``actual`` and ``sub`` are **ordered lists of
stop ids** and ``cost_mat`` is a nested dict.  ``tests/test_erp_adapter.py``
pins the adapter to that contract with a stand-in module, so the adapter stays
tested even where the clone is absent.

``sequence_deviation`` at the bottom is **not** ERP.  It is a cheap, clearly
labelled positional-deviation statistic used by the unit tests and by smoke runs
so that the pipeline is exercisable before the official scorer is installed; it
never appears in a reported result row under the name ``erp``.
"""
from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from core.logging_utils import get_logger
from core.types import DEPOT, Instance

log = get_logger(__name__)

SEARCH_DIRS = [
    Path("external/amzn_scoring"),
    Path("external/rc-cli/scoring"),
    Path("external"),
]


class ScorerUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_official_scorer() -> Callable:
    """Import ``score`` from the official scoring module.

    Raises :class:`ScorerUnavailable` with installation instructions rather than
    silently substituting a home-grown metric.
    """
    candidates = []
    env = os.environ.get("MTP_SCORER_PATH")
    if env:
        candidates.append(Path(env))
    candidates.extend(SEARCH_DIRS)

    for d in candidates:
        for path in sorted(Path(d).rglob("score.py")) if Path(d).exists() else []:
            spec = importlib.util.spec_from_file_location("amzn_score", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as exc:  # pragma: no cover - depends on the clone
                log.warning("could not import %s: %s", path, exc)
                continue
            if hasattr(module, "score"):
                log.info("using the official LMRRC scorer at %s", path)
                return module.score
    raise ScorerUnavailable(
        "the official LMRRC scoring module was not found. Clone it into "
        "external/ (see eval/erp_official.py) or set MTP_SCORER_PATH."
    )


def scorer_available() -> bool:
    try:
        load_official_scorer()
        return True
    except ScorerUnavailable:
        return False


def _as_official_args(inst: Instance, order: Sequence[int]):
    """Convert to the three structures the official ``score`` expects.

    Checked against ``MIT-CAVE/rc-cli@main scoring/score.py``:

        def score(actual, sub, cost_mat, g=1000)

    ``actual`` and ``sub`` are **ordered lists of stop ids closing back on the
    depot** -- the challenge's own ``evaluate()`` passes them through
    ``route2list``, which builds a list of length ``n + 1`` and sets
    ``route_list[-1] = route_list[0]``.  Two consequences, both load-bearing:

    * the ``{stop_id: position}`` submission format is *not* what ``score``
      itself takes;
    * the depot must appear twice, first and last.  ``seq_dev`` strips
      ``[1:-1]`` before comparing, so an open list silently drops a real
      delivery stop from the comparison -- and ``actual.index(i)`` raises
      ``ValueError`` for the stop that was cut.

    ``cost_mat`` is a nested dict of travel times keyed by stop id.

    A fresh ``cost_mat`` is built on every call on purpose: the official
    ``normalize_matrix`` does ``new_mat = mat.copy()``, a shallow copy, and then
    writes into the inner dicts -- so it mutates whatever matrix it is handed.
    Caching one and reusing it would silently score the second route against a
    doubly-normalised matrix.
    """
    ids = [s.stop_id for s in inst.stops]
    depot_id = ids[DEPOT]
    actual = [ids[i] for i in inst.actual_sequence] + [depot_id]
    sub = [ids[i] for i in order] + [depot_id]
    cost = {
        ids[i]: {ids[j]: float(inst.travel_time[i, j]) for j in range(inst.n)}
        for i in range(inst.n)
    }
    return actual, sub, cost


def erp_score(inst: Instance, order: Sequence[int]) -> float:
    """Official ERP route score for ``order`` against this route's sigma*.

    Lower is better: the challenge score is ``seq_dev * erp_per_edit``, and a
    submission identical to sigma* scores 0.
    """
    if inst.actual_sequence is None:
        raise ValueError(f"{inst.route_id} has no actual sequence to score against")
    score = load_official_scorer()
    actual, sub, cost = _as_official_args(inst, order)
    return float(score(actual, sub, cost))


# ------------------------------------------------- explicitly NOT the ERP score
def sequence_deviation(inst: Instance, order: Sequence[int]) -> float:
    """Mean absolute positional deviation from sigma*, normalised by ``n``.

    A transparent stand-in for smoke runs and tests.  Reported under the name
    ``seq_dev``, never ``erp``.
    """
    if inst.actual_sequence is None:
        return float("nan")
    pos_actual = {i: p for p, i in enumerate(inst.actual_sequence)}
    dev = [abs(pos_actual[i] - p) for p, i in enumerate(order)]
    return float(np.mean(dev)) / max(inst.n, 1)


def kendall_tau(inst: Instance, order: Sequence[int]) -> float:
    """Rank correlation with sigma*; another smoke-run diagnostic, not ERP."""
    if inst.actual_sequence is None:
        return float("nan")
    from scipy.stats import kendalltau

    pos_actual = {i: p for p, i in enumerate(inst.actual_sequence)}
    a = [pos_actual[i] for i in order]
    return float(kendalltau(a, list(range(len(order)))).statistic)

"""The ERP adapter must match the official scorer's real signature.

The official module is not vendored in this repository, so these tests install a
*stand-in* ``score.py`` with the same public interface as
``MIT-CAVE/rc-cli@main scoring/score.py``:

    def score(actual, sub, cost_mat, g=1000)   # actual/sub are ORDERED LISTS

and assert that ``eval.erp_official`` calls it correctly.  The stand-in also
reproduces one trap in the real code: ``normalize_matrix`` shallow-copies the
cost matrix and then writes into its inner dicts, mutating the caller's data.
"""
from __future__ import annotations

import textwrap

import pytest

import eval.erp_official as erp

STANDIN = textwrap.dedent('''
    """Stand-in with the official scorer's interface (not the real algorithm)."""
    CALLS = []

    def normalize_matrix(mat):
        new_mat = mat.copy()                 # shallow, exactly like the original
        for o in mat:
            for d in mat[o]:
                new_mat[o][d] = mat[o][d] * 2.0
        return new_mat

    def score(actual, sub, cost_mat, g=1000):
        if not isinstance(actual, list) or not isinstance(sub, list):
            raise TypeError("actual and sub must be ordered lists of stop ids")
        # route2list closes the tour: the depot is repeated at the end.
        if actual[0] != actual[-1] or sub[0] != sub[-1]:
            raise ValueError("route lists must close back on the depot")
        if len(set(actual)) != len(actual) - 1:
            raise ValueError("exactly one stop (the depot) may repeat")
        if not isinstance(cost_mat, dict):
            raise TypeError("cost_mat must be a nested dict")
        for sid in actual + sub:
            if sid not in cost_mat:
                raise KeyError(f"{sid} missing from cost_mat")
        normalize_matrix(cost_mat)           # mutates, as the real one does
        CALLS.append({"actual": list(actual), "sub": list(sub),
                      "n_cost_rows": len(cost_mat), "g": g})
        mismatches = sum(1 for a, b in zip(actual, sub) if a != b)
        return float(mismatches) / max(len(actual), 1)
''')


@pytest.fixture
def standin_scorer(tmp_path, monkeypatch):
    d = tmp_path / "amzn_scoring"
    d.mkdir()
    (d / "score.py").write_text(STANDIN)
    monkeypatch.setenv("MTP_SCORER_PATH", str(d))
    erp.load_official_scorer.cache_clear()
    yield d
    erp.load_official_scorer.cache_clear()


def test_scorer_is_discovered_via_env(standin_scorer):
    assert erp.scorer_available()
    assert callable(erp.load_official_scorer())


def test_adapter_passes_ordered_lists_not_dicts(standin_scorer, toy4):
    """The regression this file exists for: lists, in route order."""
    order = [0, 1, 3, 2]
    value = erp.erp_score(toy4, order)
    scorer_module = erp.load_official_scorer().__globals__
    call = scorer_module["CALLS"][-1]

    ids = [s.stop_id for s in toy4.stops]
    depot = ids[0]
    assert call["actual"] == [ids[i] for i in toy4.actual_sequence] + [depot]
    assert call["sub"] == [ids[i] for i in order] + [depot]
    assert call["n_cost_rows"] == toy4.n
    assert value == pytest.approx(2.0 / 5.0)


def test_identical_sequence_scores_zero(standin_scorer, toy4):
    assert erp.erp_score(toy4, toy4.actual_sequence) == pytest.approx(0.0)


def test_cost_matrix_is_rebuilt_each_call(standin_scorer, toy4):
    """A fresh matrix per call, because the official normaliser mutates it."""
    first = erp.erp_score(toy4, [0, 1, 3, 2])
    second = erp.erp_score(toy4, [0, 1, 3, 2])
    assert first == pytest.approx(second)
    # The instance's own travel times must be untouched by the scorer.
    assert toy4.travel_time[0, 1] == pytest.approx(10.0)


def test_missing_actual_sequence_is_an_error(standin_scorer, toy_zones):
    import dataclasses

    bare = dataclasses.replace(toy_zones) if dataclasses.is_dataclass(toy_zones) else None
    bare.actual_sequence = None
    with pytest.raises(ValueError, match="no actual sequence"):
        erp.erp_score(bare, [0, 1, 2, 3, 4])


def test_absent_scorer_raises_rather_than_substituting(tmp_path, monkeypatch):
    monkeypatch.setenv("MTP_SCORER_PATH", str(tmp_path / "nowhere"))
    monkeypatch.setattr(erp, "SEARCH_DIRS", [tmp_path / "also_nowhere"])
    erp.load_official_scorer.cache_clear()
    try:
        assert not erp.scorer_available()
        with pytest.raises(erp.ScorerUnavailable):
            erp.load_official_scorer()
    finally:
        erp.load_official_scorer.cache_clear()


# ------------------------------------------- against the real scorer, if present
REAL = pytest.mark.skipif(
    not erp.scorer_available(),
    reason="the official scorer is not installed in external/",
)


@REAL
def test_real_scorer_gives_zero_for_the_actual_sequence(toy4):
    erp.load_official_scorer.cache_clear()
    assert erp.erp_score(toy4, toy4.actual_sequence) == pytest.approx(0.0)


@REAL
def test_real_scorer_is_invariant_under_reversal(toy4):
    """A documented property of the official metric, not a bug in the adapter.

    ``seq_dev`` sums ``|rank[k] - rank[k-1]| - 1`` over the submitted order, so
    any strictly monotone ranking -- increasing *or* decreasing -- contributes
    zero.  The challenge score therefore treats the exactly-reversed route as
    perfectly driver-like.  It matters when reading lambda = 1 results: the
    imitation objective is free to pick either direction, and ``-log p_ij`` from
    Stage-1a is *not* reversal-invariant, so the behaviour cost and the reported
    ERP can disagree about direction.  See docs/DATA_NOTES.md.
    """
    erp.load_official_scorer.cache_clear()
    forward = toy4.actual_sequence
    reversed_route = [0] + list(reversed(forward[1:]))
    assert erp.erp_score(toy4, reversed_route) == pytest.approx(
        erp.erp_score(toy4, forward)
    )


@REAL
def test_real_scorer_penalises_interior_swaps(toy4):
    erp.load_official_scorer.cache_clear()
    assert erp.erp_score(toy4, [0, 2, 1, 3]) > 0.0
    assert erp.erp_score(toy4, [0, 2, 3, 1]) > erp.erp_score(toy4, [0, 2, 1, 3])


@REAL
def test_real_scorer_does_not_mutate_the_instance(toy4):
    import numpy as np

    erp.load_official_scorer.cache_clear()
    before = toy4.travel_time.copy()
    for order in ([0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 2, 1]):
        erp.erp_score(toy4, order)
    assert np.allclose(before, toy4.travel_time)

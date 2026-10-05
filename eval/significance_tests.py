"""Paired significance testing between methods (§7).

Route-level paired Wilcoxon signed-rank tests, with effect sizes reported next
to every p-value: the matched-pairs rank-biserial correlation (the effect size
that belongs to the Wilcoxon test) and Cliff's delta.  Holm correction is
applied across the pairs compared within one call, because the §7 protocol
compares *every* method pair.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


@dataclass
class PairedTest:
    metric: str
    method_a: str
    method_b: str
    n_pairs: int
    median_a: float
    median_b: float
    median_diff: float
    statistic: float
    p_value: float
    p_holm: float
    rank_biserial: float
    cliffs_delta: float
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _rank_biserial(diff: np.ndarray) -> float:
    nz = diff[diff != 0]
    if nz.size == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(nz))) + 1.0
    pos = ranks[nz > 0].sum()
    neg = ranks[nz < 0].sum()
    return float((pos - neg) / (pos + neg))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def paired_compare(
    df: pd.DataFrame,
    metric: str,
    method_col: str = "method",
    key: Sequence[str] = ("route_id",),
    methods: Sequence[str] | None = None,
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """Wilcoxon signed-rank test for every method pair on the same routes."""
    df = pd.DataFrame(df).dropna(subset=[metric])
    methods = list(methods or sorted(df[method_col].unique()))
    wide = df.pivot_table(index=list(key), columns=method_col, values=metric,
                          aggfunc="mean")

    rows: list[PairedTest] = []
    for a, b in itertools.combinations(methods, 2):
        if a not in wide.columns or b not in wide.columns:
            continue
        pair = wide[[a, b]].dropna()
        if len(pair) < 2:
            rows.append(PairedTest(metric, a, b, len(pair), float("nan"),
                                   float("nan"), float("nan"), float("nan"),
                                   float("nan"), float("nan"), float("nan"),
                                   float("nan"), "too few paired routes"))
            continue
        x, y = pair[a].to_numpy(), pair[b].to_numpy()
        diff = x - y
        note = ""
        if np.allclose(diff, 0.0):
            stat, p = float("nan"), 1.0
            note = "identical on every route"
        else:
            res = wilcoxon(x, y, alternative=alternative, zero_method="wilcox")
            stat, p = float(res.statistic), float(res.pvalue)
        rows.append(
            PairedTest(
                metric=metric, method_a=a, method_b=b, n_pairs=len(pair),
                median_a=float(np.median(x)), median_b=float(np.median(y)),
                median_diff=float(np.median(diff)), statistic=stat, p_value=p,
                p_holm=float("nan"), rank_biserial=_rank_biserial(diff),
                cliffs_delta=cliffs_delta(x, y), note=note,
            )
        )

    out = pd.DataFrame([r.as_dict() for r in rows])
    return holm(out) if not out.empty else out


def holm(df: pd.DataFrame, p_col: str = "p_value", out_col: str = "p_holm") -> pd.DataFrame:
    """Holm-Bonferroni step-down correction across the rows of ``df``."""
    df = df.copy()
    valid = df[p_col].notna()
    idx = df.index[valid]
    p = df.loc[idx, p_col].to_numpy(dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, (m - rank) * p[k])
        adj[k] = min(running, 1.0)
    df.loc[idx, out_col] = adj
    return df


def lambda_trend_test(df: pd.DataFrame, metric: str = "reliability_index") -> dict:
    """Spearman trend of ``metric`` against lambda -- the RQ2 monotonicity check."""
    from scipy.stats import spearmanr

    d = pd.DataFrame(df).dropna(subset=[metric, "lam"])
    if d["lam"].nunique() < 3:
        return {"metric": metric, "rho": float("nan"), "p_value": float("nan"),
                "note": "fewer than three lambda levels"}
    res = spearmanr(d["lam"], d[metric])
    return {"metric": metric, "rho": float(res.statistic),
            "p_value": float(res.pvalue), "n": int(len(d)), "note": ""}

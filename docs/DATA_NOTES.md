# Data notes, and one modelling decision you must be able to defend

## 1. Source

2021 Amazon Last Mile Routing Research Challenge, AWS Registry of Open Data,
bucket `s3://amazon-last-mile-challenges/almrrc2021/`. Public, no credentials
(`--no-sign-request`). 6,112 training and 3,072 evaluation routes across five US
metros (Los Angeles, Seattle, Chicago, Boston, Austin), 2018 operations.
Described in Merchán et al. (2022), *Transportation Science*.

Everything used in this repository comes from that release. Nothing is
synthesised, and the toy instances in `tests/` are unit-test fixtures that never
appear in a reported number.

## 2. What each raw file gives

| File | Contents used |
| --- | --- |
| `route_data.json` | station code, date, departure time, capacity, `route_score` (High/Medium/Low), per-stop lat/lng, type, `zone_id` |
| `package_data.json` | per-package `time_window`, `planned_service_time_seconds`, `dimensions` |
| `travel_times.json` | the route's point-to-point travel-time matrix, seconds |
| `actual_sequences.json` | $\sigma^\*$, the observed stop order (training only) |
| `invalid_sequence_scores.json` | sequence-validity score (training only) |

Conversions happen once, in `data/preprocess.py`: seconds → minutes, package
windows → minutes from the route's departure epoch, package dimensions → litres,
per-stop service time = sum over its packages, per-stop window = intersection of
its packages' windows (max start, min end).

## 3. The decision that needs defending

**The release contains no realised arrival times.** It gives a *planned* or
*estimated* travel-time matrix per route and *planned* service times per package.
So "fit the travel/service-time distributions from the real data" cannot mean
"fit to observed deviations from plan", because the observations do not exist in
this dataset.

What this repository does instead, and what the thesis must say plainly:

> The quantity whose dispersion we model is the observed variation of the
> challenge's own time fields across routes. The same (zone-pair, time-of-day,
> distance) cell recurs across thousands of routes with materially different
> supplied travel times; that cross-route dispersion is the empirical
> travel-time uncertainty for the cell. Service time is treated the same way
> within (package-count, volume, zone, time-of-day) cells.

Concretely (`models/uncertainty/quantile_model.py`):

1. conditional quantiles of $t_{ij}$ are regressed on
   `[log distance, time-of-day, same-zone, zone-pair frequency, zone-pair
   historical variance, arc historical median, stops so far/remaining, package
   count/volume at the head]`, at levels $\{0.05,0.25,0.5,0.75,0.95\}$, one
   LightGBM quantile model per level;
2. at prediction time the whole quantile curve is **shifted so its median equals
   the route's own supplied $t_{ij}$**. The model therefore contributes only the
   *spread*, conditional on covariates, around the value the challenge gives for
   that specific route.

This is honest and it is defensible: no distributional family is assumed, the
spread is estimated from real cross-route variation, and the route's own planned
time is respected as the central estimate.

### Its limitations, stated up front

* The spread is **estimation dispersion across comparable arcs**, not verified
  execution noise for one arc. If Amazon's supplied matrices were already
  route-specifically accurate, part of the measured spread is genuine
  heterogeneity between routes rather than uncertainty within a route.
* Because arrival times are unobserved, the calibration in
  `eval_uncertainty.py` checks that the model's predictive quantiles cover the
  *supplied times* out of sample. It cannot check coverage of realised times.
  Say so in the thesis; do not call it operational calibration.
* Correlation across the legs of a single route is not identifiable from this
  data and is not modelled. The Wasserstein radius $\delta$ is the only device
  standing in for it, which is one reason `calibrate_delta` picks $\delta$ from
  held-out coverage rather than setting it to a convenient constant.

### If the examiner wants realised times

Two options, both outside the 2021 release:

* the LMRRC 2022/2023 challenge data, if an instance with execution timestamps
  becomes available;
* a public urban-freight or probe-vehicle dataset for the same metros, used only
  to *validate* the dispersion magnitudes, not to train — which keeps the hard
  "real, open, public" constraint intact.

Neither is required for the thesis as scoped; the point is to have the answer
ready.

## 4. Quality filter

$\sigma^\*$ is only taken from routes with `route_score == "High"`, the official
LMRRC protocol, enforced in `data/splits.py` (`high_quality_only=True`, the
default). Medium and Low routes remain available for the uncertainty model,
where the label is about sequence quality rather than about whether the supplied
times are usable — pass `--all-quality` to include them.

## 5. Zones

`zone_id` strings (`P-13.2A`) are station-specific, so embedding them directly
cannot transfer to a held-out metro. Stage-1a therefore encodes zones
*structurally*: a zone becomes its rank by centroid travel time from the station
(`models/behavior/dataset.py::zone_ranks`), and the GBM uses zone-rank transition
frequencies alongside raw zone-pair frequencies. Stops with a missing `zone_id`
(a known quirk of the release) are grouped into a single `UNZONED` block rather
than dropped.

## 6. Leakage discipline

* Every pooled statistic — zone-pair frequency/variance/median, zone service
  medians, zone-rank transition tables — is fit inside
  `fit_historical_stats` / `TransitionTables.fit` on the **training split only**
  and passed explicitly into feature construction.
* `data/splits.py` splits by a stable hash of the route id, so train/validation
  membership does not depend on file order or machine.
* The official 3,072-route evaluation set is reachable only through
  `official_eval_split`, which appends a timestamped record to
  `results/official_eval_uses.jsonl` on every call, so "touched once" is an
  auditable claim.
* The reliability index is measured on an independent scenario draw
  (`eval/metrics.py`, `EVAL_SEED_OFFSET`) with $\delta$ stripped: the optimizer's
  ambiguity set never doubles as the yardstick.

## 7. Properties of the official score you must report

The scoring module is `MIT-CAVE/rc-cli@main scoring/score.py`, vendored into
`external/amzn_scoring/` with its commit sha recorded in `PROVENANCE.json`
(commit `fdafa50`, read 2026-10-01). `score(actual, sub, cost_mat, g=1000)`
returns `seq_dev(actual, sub) * erp_per_edit(actual, sub, normalize_matrix(cost_mat), g)`.
Three details of the real implementation, all of which the adapter in
`eval/erp_official.py` had to be corrected against:

**It takes closed tours, not submission dicts.** `evaluate()` passes routes
through `route2list`, which returns a list of length `n + 1` with
`route_list[-1] = route_list[0]`. `seq_dev` then strips `[1:-1]`. Passing an
open list silently drops a real delivery stop, and raises `ValueError` on the
stop that was cut. The `{stop_id: position}` format is for the *submission
file*, not for `score`.

**It is invariant under reversal.** `seq_dev` sums `|rank[k] - rank[k-1]| - 1`
over the submitted order, so any strictly monotone ranking contributes zero —
decreasing as well as increasing. The exactly-reversed route therefore scores
**0, the same as the driver's own sequence**. This is a property of the
challenge metric, confirmed by direct computation on a hand-built 4-stop case
(`tests/test_erp_adapter.py::test_real_scorer_is_invariant_under_reversal`).

Two consequences to state in the thesis rather than discover in the viva:

* `-log p_ij` from Stage-1a is *not* reversal-invariant, so at λ = 1 the
  behaviour objective and the reported ERP can disagree about direction. A route
  that the pointer network considers implausible may still score a perfect 0.
  Report the behaviour cost alongside ERP, as `eval/metrics.py` does, so the
  disagreement is visible instead of averaged away.
* The reliability term is strongly direction-dependent — arrival times differ
  completely between a route and its reverse. So the λ sweep can buy large
  reliability gains at *zero* ERP cost in exactly those cases, which is a real
  and quotable result rather than an artefact. Do not mistake it for a bug.

**`normalize_matrix` mutates its argument.** It does `new_mat = mat.copy()`, a
shallow copy, then writes into the inner dicts. The adapter therefore rebuilds
the cost dict on every call; caching one and reusing it would score the second
route against a doubly-normalised matrix.

## 8. Measured facts about the real data

From `python -m scripts.data_diagnostics` over 1,500 High/Medium-quality
training routes (`results/data_diagnostics*.json`):

| quantity | mean | median | p90 | max |
| --- | --- | --- | --- | --- |
| stops per route | 148 | 147 | — | 238 |
| zones per route | 20.6 | 20 | 28 | 46 |
| stops with a time window | 10.2 | 9 | 21 | 45 |
| **fraction of stops with a window** | **7.3%** | **5.9%** | 15.3% | 39.5% |
| total planned service time (min) | 265 | 262 | 332 | 453 |
| median arc travel time (min) | 5.9 | 5.6 | 8.3 | 13.1 |
| mean travel-matrix asymmetry (min) | 0.53 | 0.47 | 0.82 | 2.99 |

Two of these drive design decisions:

**Windows are sparse (~6% of stops).** The reliability term only binds at
window-carrying stops, which is exactly why the incremental evaluator in
`solvers/alns/state.py` scales with the window count rather than the stop count.
It also means `sum_i eta_i` is a sum over ~10 terms per route, not ~150.

**The travel matrix is directed.** Mean |t_ij - t_ji| is about half a minute,
so the formulation must not assume symmetry anywhere. It does not.

### 8.1 Zone precedence is not supported by the data

This is the one that changes the model. Measured over the same 1,500 routes:

| | value |
| --- | --- |
| sigma* visits every zone in one unbroken block | **38.3%** of routes |
| extra zone blocks beyond one-per-zone (mean / median / p90 / max) | 2.33 / 1 / 6 / 38 |
| per metro (Boston / Chicago / LA / Austin) | 32.9% / 41.0% / 40.0% / 39.4% |

So **about three fifths of real drivers re-enter at least one sort zone.** The
zone-precedence constraint block of §3 therefore excludes sigma* itself from the
feasible set on the majority of routes. Two consequences:

1. Under that constraint the λ = 1 imitation arm *cannot* reproduce the driver
   sequence, so its ERP is bounded away from zero by construction and every
   comparison against the published imitation-learning results is biased
   against this thesis.
2. It is also unnecessary. Zone cohesion is a *behavioural regularity*, and
   Stage-1a already learns it: `-log p_ij` charges for an implausible
   zone switch because implausible zone switches are rare in the training
   sequences. Expressing the same regularity twice — once learned, once as a
   hard constraint — only removes the model's ability to disagree with it where
   the driver did.

`SolverConfig.zone_precedence` therefore defaults to **False**. The constraint
is fully implemented and tested (MILP, Benders and ALNS all honour it) and is
kept as the §7 ablation, which is the right place for it: the ablation now
answers "what does imposing zone contiguity cost?" with a number.

On synthetic probe instances relaxing it improves the objective by an order of
magnitude (see `docs/BUILD_STATUS.md`); on real routes the effect size is an
experiment to run, not a claim to make here.

## 9. Stage-1b calibration, measured

`python eval_uncertainty.py --held-out-metro Austin --limit 120` over 16.5k
travel rows and 16.5k service rows. Empirical coverage of each nominal
conditional quantile:

| head | split | 0.05 | 0.25 | 0.50 | 0.75 | 0.95 |
| --- | --- | --- | --- | --- | --- | --- |
| service | train | 0.054 | 0.261 | 0.515 | 0.757 | 0.952 |
| service | validation | 0.070 | 0.275 | 0.499 | 0.718 | 0.934 |
| service | held-out metro | — | — | — | — | 0.951 |
| travel | train | 0.052 | 0.259 | 0.515 | 0.767 | 0.957 |
| travel | **validation** | 0.136 | 0.303 | 0.488 | 0.666 | **0.803** |
| travel | **held-out metro** | 0.160 | 0.329 | 0.488 | 0.638 | **0.746** |

Mean pinball loss: travel 0.192, service 0.310.

**The service head is well calibrated; the travel head is not.** In-sample it is
almost exact, but out of sample its intervals are too narrow, and the failure
grows across metros: the nominal 95% quantile covers 80.3% on held-out routes of
the training metros and only 74.6% on a held-out metro. Report this, and report
it before any optimisation result — it is the honest motivation for the DRO
layer rather than a post-hoc justification for it.

### 9.1 How delta is actually calibrated

The protocol in §4b of the build spec -- declare routes reliable, verify on a
held-out draw -- **cannot work here**, and the first implementation of it
returned `delta = 0` for two structural reasons:

* the held-out draw is sampled from the *same fitted model*, so it measures
  Monte-Carlo stability rather than whether the model is right;
* the predictive quantiles are anchored so their median equals the route's own
  supplied `t_ij` (§3 above), so the single realisation available per route sits
  at the centre of its own predictive distribution by construction. Per-route
  coverage is therefore not identifiable and the check is vacuous.

What *is* identifiable is the spread, across routes — exactly what the table
above measures. `calibrate_delta_from_coverage` turns the measured shortfall
into a radius: the DRO inflation adds `delta/eps` per prefix component, so
setting `delta = eps * m` with `m` the per-component margin by which the nominal
upper quantile under-covers makes the worst-case bound carry precisely the
measured deficit. On the held-out-Austin fold at `eps = 0.10`:

| head | nominal | empirical | margin |
| --- | --- | --- | --- |
| travel | 0.90 | 0.803 | **0.453 min** |
| service | 0.90 | 0.934 | 0.000 min (over-covers) |

giving **`delta = 0.0453`**. The old scenario-stability routine is kept, clearly
labelled as a diagnostic, because its curve is still informative: as delta grows
the number of routes it declares reliable falls 10 → 5 → 2 → 1 → 0, and every
declared route holds up.

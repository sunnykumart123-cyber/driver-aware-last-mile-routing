# Build status against the phase gates (§9)

Updated: 2026-10-01. Keep this table honest — it is the thing to look at when
resuming work.

| Phase | Builds | Definition of done | Status |
| --- | --- | --- | --- |
| 1 Setup | repo skeleton, environment, dataset download | `data/raw/` populated; environment reproducible from a lockfile | **code done, data not downloaded.** `data/download.py` (stdlib HTTPS, resumable) + `requirements.lock` exist; all ten file sizes verified against the live bucket on 2026-10-01. Needs 3.1 GB of disk, no AWS CLI |
| 2 Data pipeline | `preprocess.py`, `splits.py`, naive TSP baseline | naive baseline's ERP matches expected order of magnitude on the official scorer | **code done, gate open.** Parser unit-tested against a synthetic LMRRC-shaped record; needs the real files and the official scorer to close |
| 3 Baseline gate | GBM transition model, reproduced da Costa/Mo ERP | reproduced score within a defensible tolerance of the published number | **OPEN, with a measured number.** First GBM (binary objective, 8 negatives, 600 routes) scored ERP **0.0801** on 60 held-out-Austin validation routes -- *worse* than the naive TSP's 0.0660, so not competitive with the published imitation results. Three defects found and fixed (below); retrained with LambdaRank. Closing the gate still needs the published reference value from you |
| 4 Stage 1 complete | pointer network, quantile uncertainty model | both evaluated in isolation: imitation accuracy, calibration diagrams | **Stage-1b DONE**, Stage-1a in progress. Quantile model fit on 111k travel + 111k service rows; calibration measured and reported (`results/uncertainty_calibration_ho_austin.json`, reliability diagram PNG). Service head well calibrated; **travel head overconfident out of sample** (nominal 95% covers 74.6% on a held-out metro). Radius calibrated from that shortfall: **delta = 0.0453**. Pointer network not yet trained on real data |
| 5 Exact formulation | `exact_milp.py` with the full §3 model | matches hand-computed optimum on ≥5 toy cases | **CLOSED.** 5 hand-computed cases plus 8 MILP-vs-enumeration cases pass (`tests/test_milp_toy_cases.py`) |
| 6 Production solvers | `alns/`, `benders/`, `run_pipeline.py` | end-to-end pipeline runs; ALNS and Benders agree within an expected gap | **CLOSED in miniature.** ALNS == brute force at 5 lambda levels, 3 DRO radii and 3 ambiguity geometries, single- and multi-scenario; Benders == brute force; full chain runs in `tests/test_pipeline_end_to_end.py`; per-iteration cost measured up to 240 stops (see below). Re-confirm the ALNS-vs-Benders *gap* on real 60–80-stop routes |
| 7 Full experiments | `baselines/`, `ablations.py`, leave-one-metro-out | all arms produce logged rows in `results/experiment_log.parquet` | **code done, not run.** Needs Phases 1–4 data |
| 8 Sensitivity | `lambda_sweep_rq2.py`, `sensitivity_rq3.py`, `significance_tests.py` | RQ2 and RQ3 curves produced, with Wilcoxon results | **code done, not run** |
| 9 Write-up support | result tables/plots for the thesis | every number traces to a row in `experiment_log.parquet` | **schema fixed** (`eval/metrics.py::RESULT_COLUMNS`); exports pending the runs |

## What "code done, gate open" means

The implementation is finished and unit-tested; the gate is a statement about
*numbers on real data*, which cannot be produced without the 3.1 GB download.
Do not mark such a gate closed until the number exists in `results/`.

## Memory discipline on the raw files

`travel_times.json` is 1.73 GB (training) and 804 MB (evaluation); as a Python
dict of dicts of floats it would need roughly an order of magnitude more than
that. `data/jsonstream.py` therefore provides a byte-offset index over each
raw file's top-level keys, built in one scan that parses no values, so
`data.preprocess` joins the four files on route id by seeking rather than by
holding them in memory. Peak usage is the 4 MiB buffer plus one route's
payload. This was a genuine defect in the first cut of the parser, which called
`json.loads` on the whole file — it would have died on the real data.

## Ordered list of what to do next

1. **Download** — `python -m data.download --plan` to see the ten files and
   their sizes (3.1 GB total), then `python -m data.download --dest data/raw`,
   then `python -m data.download --verify-only` and keep
   `download_manifest.json`. Sizes are checked after the fetch, so a truncated
   file is caught immediately rather than surfacing as a parse error later.
2. ~~**Install the official scorer**~~ — **done.** `external/amzn_scoring/`
   holds `score.py` from `MIT-CAVE/rc-cli@fdafa50` with its sha256 recorded in
   `PROVENANCE.json`. The adapter was wrong in two ways against the real
   signature (it passed `{stop_id: position}` dicts, and open rather than
   closed tours) and is now pinned by tests. See `docs/DATA_NOTES.md` §7 for
   the metric's reversal invariance, which has to be reported.
3. **Preprocess** both splits, check `preprocess_summary_training.json` against
   the published counts (6,112 routes, five metros).
4. **Phase-2 gate** — run the naive TSP baseline through `eval/metrics.py` on a
   few hundred High-quality routes; confirm the ERP magnitude is plausible.
5. **Phase-3 gate** — `train_behavior.py --model gbm` then `eval_behavior.py`;
   compare with the published da Costa / Mo numbers and **write the tolerance
   argument down** before proceeding.
6. **Stage 1** — `train_behavior.py --model pointer --all-folds` and
   `train_uncertainty.py --all-folds --calibrate-delta`, then the two eval
   scripts. Report calibration before any optimisation result.
7. **Phase-6 re-confirmation on real sizes** — pick ~20 routes with 60–80 stops,
   run `pipeline.run_pipeline --method benders` and compare its certified gap
   against `--method alns`. Record the gap distribution; this is the number that
   justifies using the ALNS everywhere else.
8. **Phases 7–8** — `experiments/ablations.py --all-folds`,
   `lambda_sweep_rq2.py --all-folds`, `sensitivity_rq3.py --all-folds`.
9. **Official evaluation set — once.** Only after the above is stable.

## Solver performance (measured, `scripts/benchmark_alns.py`)

CPU-only laptop, S = 64 scenarios, δ = 0.1, λ = 0.5. The real data carries a
time window on about **6% of stops**, so that is the operative density.

| stops | windows | zone precedence | cap | ms/iteration | final objective |
| --- | --- | --- | --- | --- | --- |
| 150 | 8 | on | — | 13.0 | 9.03 |
| 150 | 8 | **off** | 24 | 25.4 | **1.02** |
| 240 | 14 | on | — | 49.0 | 21.17 |
| 240 | 14 | **off** | 24 | 102.8 | **1.02** |
| 240 | 14 | off | 0 | 510.5 | 1.03 |

Cost scales with the number of **window-carrying** stops, not with the stop
count — that is the point of the incremental evaluator. Relaxing zone
precedence costs about 2x per iteration and buys an order of magnitude in
objective, because the constraint was excluding most of the search space (and,
on real routes, σ\* itself). The `candidate_cap = 24` shortlist is a 5x speed-up
at no measurable cost in objective.

At the realistic window density, 4,000 iterations on a 240-stop route is about
7 minutes, so the `--time-limit` governs in practice rather than the iteration
count.

Two rewrites got it there, and both are easy to undo by accident:

1. `ReliabilityEvaluator.risk_array` is the single vectorised implementation of
   the geometry; `eta`, `risk_at` and the ALNS all call it. A per-stop Python
   loop over CVaR calls costs ~4x.
2. `RouteState.insertion_deltas_many` scores all (stop, position) pairs in one
   pass, and the repair operators apply `repair_refresh_every` insertions per
   scan instead of rescanning after each one. A full rescan per insertion is
   O(q²) risk evaluations and costs ~20x; the 240/35 cell measured
   **5,391 ms/iteration** before this change.

If either is reverted, the benchmark table above is the regression test.

## Phase-3 diagnosis: why the first GBM underperformed

Measured, not guessed.  Feature separation on real training rows is strong --
`same_zone` 0.84 (chosen) vs 0.07 (alternatives), `zone_rank_transition_freq`
0.71 vs 0.07, `tt_rank_among_remaining` 1.8 vs 36.9 -- so the signal was there
and the model was failing to use it.  Three causes, all now fixed:

1. **A vacuous feature.** `returns_to_finished_zone` was constant 0 on every
   training row: a zone only counts as *finished* once all its stops are
   visited, so no remaining candidate can ever be in one.  Replaced with
   `returns_to_abandoned_zone` (the zone was worked and left but still has
   unvisited stops), which is the behaviour ~three fifths of real routes
   exhibit.
2. **An exploded feature.** `tt_ratio_to_nearest` reached ~25,000 on routes
   containing a near-zero arc, drowning its own signal.  Now
   `log1p(t / max(nearest, 0.1))`.
3. **A train/inference mismatch.** The binary objective learned to separate the
   driver's choice from 8 *random* alternatives, then at decode time was asked
   to rank ~150 of them.  Now LightGBM `LGBMRanker` with LambdaRank, one query
   group per decision state, and 24 negatives; `objective="binary"` is retained
   for comparison.

After the fixes and retraining with LambdaRank on 900 routes, ERP improved only
from 0.0801 to **0.0768** -- still about level with the zone-respecting naive
TSP (0.0660) and well short of published imitation results. The vacuous
feature did come alive (`returns_to_abandoned_zone` now carries 1.7% of the
gain, from a structural 0%), so the fixes were real, but they were not the
binding constraint.

**Conclusion to carry into the write-up:** a greedy pairwise transition scorer
plateaus around ERP 0.077 on this data. That is a finding about the method
class, not a bug. The published results this gate aims at come from
*zone-sequence* models (Wu et al. 2022) and pool-and-select over many candidate
routes (da Costa et al. 2023), neither of which is a single greedy pass over a
pairwise score. §4a of the build spec designates the **pointer network** as the
primary Stage-1a model and the GBM as a secondary baseline and interpretability
check -- that division is the right one, and the Phase-3 reproduction target
belongs to the pointer network.

The point of recording this here: the Phase-3 gate exists precisely to catch a
Stage-1a model that looks trained but is not competitive, and it did.

## Model changes forced by the real data

* **Zone precedence is off by default.** Only 38.3% of real driver sequences
  visit each sort zone in one unbroken block (1,500 routes; 33-41% by metro).
  Imposing contiguity excludes sigma* on ~three fifths of routes and puts a
  floor on the attainable ERP. The constraint remains implemented and tested;
  it is now the §7 ablation rather than the default. Full argument and numbers:
  `docs/DATA_NOTES.md` §8.1.
* **Windows are sparse** (~6% of stops carry one), which is what makes the
  incremental reliability evaluator cheap, and means `sum_i eta_i` has ~10
  nonzero terms per route.
* **Travel matrices are directed** (mean asymmetry ~0.5 min); nothing in the
  formulation assumes symmetry.

## Known open engineering items

* **Repair staleness.** Within one scan the later insertions in a batch use
  slightly stale costs (see the long comment in `solvers/alns/repair_ops.py`).
  This is a search-quality choice, not a correctness one: `RouteState` recomputes
  the exact objective after every insertion and the tests assert that the
  incremental bookkeeping equals a from-scratch evaluation. Set
  `repair_refresh_every=1` for textbook behaviour.
* **Destroy size is capped** at `SolverConfig.destroy_max_stops = 25` on top of
  the usual fraction, because the reliability term couples the whole route. If a
  tuning study wants larger neighbourhoods, raise it and re-measure.
* **Exact solver scenario count.** `solve_exact(..., max_scenarios=16)`
  sub-samples $\hat P$ by default. Toy tests use the full set; state the
  sub-sampling in any table that reports an exact optimum.
* **`l2` ground norm** is heuristic-only (not MILP-representable); the exact
  solver raises rather than approximating.
* **Gurobi backend** is written but unexercised here (no licence in the dev
  environment). Run `tests/test_milp_toy_cases.py` with
  `SolverConfig(exact_backend="gurobi")` once a licence is available — the tests
  are backend-parametrisable by design.
* **Dependence across legs** is not modelled; see `docs/DATA_NOTES.md` §3.

## First end-to-end result on real data (2026-10-01)

Pilot only -- 4 held-out-Austin routes, GBM behaviour cost, 25 s ALNS budget per
route, eps = 0.10, delta = 0.0453 (coverage-calibrated), 32 scenarios. Official
ERP scorer. `tag = real_smoke` in `results/experiment_log.parquet`.

| lambda | ERP | reliability index | expected lateness (min) | sum eta | behaviour cost |
| --- | --- | --- | --- | --- | --- |
| 1.0 pure imitation | **0.1352** | 0.266 | 72.8 | 431.9 | 2.344 |
| 0.5 combined | **0.1444** | **1.000** | **0.0** | 1.3 | 2.497 |
| 0.0 robust, no behaviour | 0.5442 | 1.000 | 0.0 | 0.0 | 8.627 |

Read with care, but it is the thesis's claim showing up in measurements:

* **RQ1.** Driver-plausible routes do carry large unquantified risk. At
  lambda = 1 three of the four routes have a reliability index at or near zero
  and 68-123 minutes of expected lateness. The imitation objective never sees
  the time windows, so it has no reason to respect them.
* **RQ2 (price of reliability).** Going from lambda = 1 to lambda = 0.5 costs
  ERP 0.1352 -> 0.1444, about **+0.009 (+6.8%)**, and buys reliability
  0.27 -> 1.00 with expected lateness 72.8 -> 0 minutes. On two of the four
  routes the ERP actually *improved*, so the cost is not uniform.
* **The combination is doing the work.** Dropping the behaviour term entirely
  (lambda = 0) costs ERP 0.544 -- four times worse -- for no extra reliability.
  Neither term alone is enough, which is the whole premise.

One honest wrinkle to investigate: the lambda = 1 ALNS arm scores ERP 0.135,
*worse* than simply decoding the GBM greedily (0.0768). Optimising
`sum -log p_ij` harder makes the official score worse, so that sum is an
imperfect surrogate for ERP. That deserves its own paragraph in the write-up --
and it is an argument for the pool-and-select style decoding the literature
uses, rather than exact minimisation of the learned arc cost.

Scale-up needed before any of this goes in a chapter: all five folds, the full
route sets, a proper lambda grid, and the pointer network in place of the GBM.

## Stage-1a on real data (2026-10-01)

Held-out-Austin fold, 500 training routes, 10 epochs, official ERP scorer.

| model | top-1 | top-3 | top-5 | mean rank | ERP (val) | ERP (held-out metro) |
| --- | --- | --- | --- | --- | --- | --- |
| GBM (LambdaRank) | 0.372 | 0.698 | 0.820 | 4.64 | 0.0799 | 0.1044 |
| **pointer network** | **0.709** | **0.900** | **0.950** | **2.34** | **0.0750** | **0.0938** |
| naive TSP (zone-respecting) | — | — | — | — | 0.0660 | — |

Three things to take from this.

**The pointer network needed current-stop-relative features.** Trained as
specified -- a per-stop encoder plus additive attention -- it sat at *exactly*
the uniform-policy loss (3.977 train / 3.954 val against a uniform floor of
3.969) for eight epochs and learned nothing. The encoder is permutation
equivariant over per-stop features, so "how far is candidate j from where the
vehicle is now" is simply not in its representation, and that is the single
strongest signal (it is the GBM's top feature group). Adding an arc-feature
head -- normalised travel time from the current stop, its log, same-zone,
zone-already-visited, zone-abandoned, and rank among remaining candidates --
took val loss to 1.097 and top-1 to 0.709. `tests/test_stage1_models.py` pins
the head into the forward pass.

**Cross-metro generalisation is strong.** Top-1 falls only 0.709 → 0.704 on
Austin, a station never seen in training. The structural zone encoding (rank by
depot travel time rather than the literal sort-zone string) is doing its job.

**One-step accuracy does not buy whole-route similarity.** Despite 71% top-1 and
95% top-5, the greedy rollout agrees with sigma* on only 3% of positions and
scores ERP 0.075 -- about level with a zone-respecting nearest-neighbour TSP.
This is textbook compounding error: ~29% of steps are wrong and one early
mistake derails the remainder. It is also the strongest available argument for
this thesis's framing: greedy decoding of a learned policy is not the right way
to use it, and the published results that beat this come from rollout and
pool-and-select schemes rather than one greedy pass. Using `-log p_ij` as an
*arc cost inside an optimisation model* is a different and better-motivated use
of the same model.

## Second end-to-end result, pointer behaviour cost

5 held-out-Austin routes, delta = 0.0453, eps = 0.10, 32 scenarios, 25 s ALNS
per solve. `tag = pointer_sweep`.

| lambda | ERP | reliability index | expected lateness (min) | sum eta |
| --- | --- | --- | --- | --- |
| 1.00 pure imitation | **0.1078** | 0.151 | 70.0 | 410.4 |
| 0.75 | 0.1565 | 1.000 | 0.0 | 5.9 |
| 0.50 | **0.1346** | 1.000 | 0.0 | 0.6 |
| 0.00 robust, no behaviour | 0.4613 | 1.000 | 0.0 | 0.0 |

**Price of reliability: ERP +0.027 (+25%) to go from reliability 0.151 to 1.000
and from 70 minutes of expected lateness to zero.** Four of the five routes sit
at reliability ~0 under pure imitation and all five reach 1.000 at lambda = 0.5.
Removing the behaviour term entirely costs 4.3x the ERP for no extra
reliability.

The lambda = 0.75 cell is worse than lambda = 0.5 in ERP, which should not be
over-read at n = 5 with a 25-second budget -- but it is consistent with the
surrogate mismatch noted above, and the full sweep needs to settle it.

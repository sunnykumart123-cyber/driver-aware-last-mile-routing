# MTP: Teaching a Delivery Router to Respect Both the Driver and the Clock

**M.Tech (IEOR) thesis codebase — IIT Bombay**
*Integrating Learned Driver Behaviour with Stochastic Service-Level Guarantees in Last-Mile Delivery Routing*

This README is written so that someone who has never seen the thesis brief can
understand what the project does, why it's built this way, and how to run it.
If you want the mathematical derivation or the raw data facts, those live in
`docs/` and are linked at the end — this page is the plain-language front door.

---

## 1. The problem, in one paragraph

Every day, thousands of delivery drivers leave a depot with ~150–200 packages
and decide, largely by instinct and experience, what order to visit stops in.
Academic research on "last-mile routing" has mostly gone one of two ways:

1. **Copy the driver.** Train a machine-learning model to imitate the
   sequences real drivers actually drove (because real drivers are
   surprisingly good at this — they know shortcuts, traffic patterns, and
   building access that no map has). But a route that *looks* like something
   a driver would do says nothing about whether it will actually finish on
   time. A learned model has no idea that stop #47 promised a 2–4pm delivery
   window.
2. **Optimize for guaranteed punctuality.** Classical operations-research
   routing (vehicle routing with time windows, robust/stochastic
   optimization) can mathematically guarantee a service-level target — but it
   has to assume travel and service times come from some known random
   process, and the resulting routes often look nothing like what an
   experienced driver would choose, because the optimizer has no notion of
   "driver-like."

**Nobody had combined both.** This thesis builds one routing model that is
*simultaneously* scored on how closely it imitates real driver behaviour and
on a mathematically certified bound on the probability of missing a delivery
window — and lets you dial a single knob (λ, "lambda") between the two
extremes to see what reliability costs you in driver-likeness, and vice versa.

---

## 2. The three-stage pipeline, in plain words

Think of it as an assembly line with three stations. Each one is built and
tested independently before being wired into the next.

```
  Real delivery data (Amazon's public research dataset)
            │
            ▼
┌───────────────────────┐    ┌───────────────────────┐
│   STAGE 1a             │    │   STAGE 1b             │
│   "How would a driver  │    │   "How uncertain is    │
│    sequence these      │    │    the travel time to  │
│    stops?"              │    │    each stop?"          │
│                        │    │                        │
│  A neural network      │    │  A model that predicts  │
│  (pointer network)     │    │  not just "this arc     │
│  and a gradient-       │    │  takes 6 minutes" but   │
│  boosted scorer, both  │    │  "it takes somewhere    │
│  trained on real       │    │  between 4 and 11       │
│  driver routes.        │    │  minutes, and here's    │
│                        │    │  the full distribution." │
│  Output: a "driver-    │    │  Output: an uncertainty │
│  likeness" cost on     │    │  cloud of plausible     │
│  every possible        │    │  travel/service times   │
│  next-stop choice.     │    │  for every arc.          │
└───────────┬────────────┘    └───────────┬────────────┘
            │                             │
            └──────────────┬──────────────┘
                            ▼
              ┌───────────────────────────┐
              │   STAGE 2                  │
              │   "Find the best route,     │
              │    trading off driver-      │
              │    likeness against a       │
              │    certified punctuality    │
              │    guarantee."               │
              │                             │
              │  An optimization model that │
              │  minimises                  │
              │     λ·(driver-unlikeness)   │
              │   + (1-λ)·(worst-case       │
              │            lateness risk)   │
              │  solved by three different  │
              │  solvers that all have to   │
              │  agree with each other.     │
              └─────────────────────────────┘
```

* **λ = 1** → pure imitation. The route looks exactly like what Stage 1a
  thinks a driver would do; the optimizer never looks at the clock.
* **λ = 0** → pure reliability. The optimizer only cares about not being late,
  using the uncertainty cloud from Stage 1b, and ignores driver behaviour
  entirely.
* **Anywhere in between** → the thesis's actual contribution: a route that is
  both plausible *and* has a mathematically bounded chance of lateness, with
  λ controlling exactly how much of each you're buying.

---

## 3. What "a certified punctuality guarantee" actually means

This is the part that sounds like jargon but is a genuinely useful idea, so
it's worth unpacking.

Normal routing says: "assume travel times are roughly what Google Maps says,
and hope." This project instead asks: *given everything we don't know about
the exact travel time on each arc, what is the worst-case probability that
this stop will be late — not under one random guess, but under the whole
range of plausible travel-time distributions consistent with what we've
observed in the data?* That "whole range" is called an **ambiguity set** (a
mathematically precise, data-calibrated ball of possible probability
distributions), and the worst case over it is a standard risk measure called
**CVaR** (Conditional Value-at-Risk — "the average of the bad outcomes,
not just the worst single one").

The size of that ambiguity ball (called **δ**, "delta") is not a number we
pick by hand. It is calibrated directly from how wrong Stage 1b's uncertainty
model actually turns out to be when checked against held-out real data — see
§7 below, this is one of the real findings of the project, not a tuning knob.

---

## 4. What's actually in this repository

```
MTP/
├── core/              Shared building blocks: the Instance/Solution data
│                      types, config objects, file I/O, and the combined
│                      λ-weighted objective function every solver optimises.
│
├── data/              Everything about getting and parsing the real dataset:
│                      download.py (fetches it), jsonstream.py (reads multi-
│                      gigabyte files without running out of memory),
│                      preprocess.py (turns raw JSON into clean route objects),
│                      splits.py (train/validation/held-out-metro splits with
│                      no data leakage).
│
├── models/
│   ├── behavior/      Stage 1a: pointer_net.py (the neural network),
│   │                  gbm_transition.py (the gradient-boosted alternative),
│   │                  dataset.py (feature engineering), cost.py (turns either
│   │                  model's output into an arc cost for the optimizer).
│   └── uncertainty/    Stage 1b: quantile_model.py (predicts travel/service
│                       time distributions), ambiguity_set.py (builds the
│                       "ambiguity ball" and calibrates its size from data).
│
├── solvers/           Stage 2, three independent ways to solve the same
│                      optimization problem (so they can be cross-checked):
│   ├── reliability.py  The core risk-evaluation math (CVaR over the
│   │                   ambiguity set) — this is the DRO ("distributionally
│   │                   robust optimization") layer.
│   ├── exact_milp.py   An exact mixed-integer solver (via Gurobi or the
│   │                   free CBC solver) — slow, but provably optimal.
│   │                   Used to validate everything else.
│   ├── alns/           Adaptive Large Neighbourhood Search — a fast
│   │                   heuristic that scales to real 150+ stop routes.
│   │                   This is the one actually used for real experiments.
│   └── benders/        Logic-based Benders decomposition — a middle ground
│                       that gives a certified optimality gap on medium-sized
│                       instances.
│
├── pipeline/           run_pipeline.py glues Stages 1 and 2 together: loads
│                       trained models, builds the objective, picks a solver,
│                       and writes results.
│
├── baselines/          Three comparison points: a naive shortest-path TSP, a
│                       pure-imitation baseline, and a pure-reliability
│                       baseline with no behaviour model — so every result can
│                       be judged against "what if you only had half the idea."
│
├── experiments/        The actual research questions as runnable scripts:
│                       ablations.py (which piece of the model matters?),
│                       lambda_sweep_rq2.py (the price-of-reliability curve),
│                       sensitivity_rq3.py (what if the uncertainty model is
│                       worse?), export_thesis_tables.py (turns logged results
│                       into tables/plots for the write-up).
│
├── eval/               How every result is measured: metrics.py (the common
│                       scoring schema), erp_official.py (wraps Amazon's own
│                       official scoring code — never a home-grown substitute),
│                       significance_tests.py (statistical tests between
│                       methods).
│
├── scripts/            Supporting tools: benchmark_alns.py (how fast is the
│                       solver?), data_diagnostics.py (what does the real data
│                       actually look like?), phase_gates.py (checkpoints that
│                       must pass before moving to the next phase of the plan).
│
├── tests/              69 automated tests. Every solver is checked against
│                       hand-computed answers and brute-force enumeration on
│                       toy problems small enough to verify by hand — this is
│                       what makes the heuristics on real 150-stop routes
│                       trustworthy.
│
├── docs/               The detailed technical reference:
│   ├── FORMULATION.md   The full mathematics, derived step by step, with
│   │                    every constant traced back to a line of code.
│   ├── DATA_NOTES.md    What the real dataset does and doesn't contain, and
│   │                    several findings that changed the model (e.g. real
│   │                    drivers don't actually obey the "zone contiguity"
│   │                    rule the literature assumes).
│   └── BUILD_STATUS.md  A running log of what's done, what's measured, and
│                        what's still open — the project's own status board.
│
└── external/            Amazon's official scoring code, fetched (not
                         rewritten) so every reported score is genuinely
                         comparable to the published leaderboard.
```

~10,700 lines of Python, 69 test files, no part of the pipeline left
unvalidated before being pointed at real data.

---

## 5. The data

This uses the **2021 Amazon Last Mile Routing Research Challenge** dataset —
a real, public release of over 6,000 actual delivery routes from five US
cities (Los Angeles, Seattle, Chicago, Boston, Austin), with real stop
sequences, package time windows, and travel-time estimates. It is free and
requires no account, but it is **3.1 GB** across ten files (the biggest single
file, a travel-time matrix, is 1.73 GB on its own).

Nothing in this project is synthetic. Every number reported from Stage 2
onward comes from this real dataset, scored with Amazon's own official
scoring code.

---

## 6. Installing and running it

```bash
# 1. Set up the environment
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                        # 167 tests, ~90 seconds, no data needed

# 2. Get the real data (3.1 GB, public bucket, no account needed)
python -m data.download --plan                  # see sizes first, nothing downloaded yet
python -m data.download --dest data/raw          # actually download it
python -m data.preprocess --raw data/raw --out data/processed --split training
python -m data.preprocess --raw data/raw --out data/processed --split evaluation

# 3. Train the learned models for one fold (holding out Austin to test generalisation)
python train_behavior.py    --held-out-metro Austin --model both
python train_uncertainty.py --held-out-metro Austin --calibrate-delta

# 4. Check the learned models in isolation, before trusting any downstream result
python eval_behavior.py    --held-out-metro Austin
python eval_uncertainty.py --held-out-metro Austin --plot

# 5. Run the actual routing pipeline
python -m pipeline.run_pipeline --held-out-metro Austin --lam 0.5 --limit 50

# 6. Run the full research experiments (all five cities, every baseline, the price-
#    of-reliability curve, sensitivity analysis)
python -m experiments.ablations         --all-folds --limit 100
python -m experiments.lambda_sweep_rq2  --all-folds --limit 25
python -m experiments.sensitivity_rq3   --all-folds --limit 20
```

A `Makefile` wraps all of the above into short commands (`make test`,
`make data`, `make stage1`, `make experiments`, …) if you'd rather not type the
full invocations.

Every run appends rows to one file, `results/experiment_log.parquet`, under
one consistent schema — so nothing ever needs to be re-run just to build a
different table or chart; it's all group-bys over that file.

**Gurobi is optional.** Without a license the exact solver falls back to the
free, open-source CBC solver automatically.

**The official scorer is required for trustworthy numbers.** Amazon's own
scoring code lives in `external/`. Without it, the pipeline will not invent a
substitute score — it clearly labels results with a stand-in metric instead,
so nobody mistakes an approximation for the real thing.

---

## 7. What's been found so far (as of this writing)

This project is mid-flight on real data, not finished. Here's the honest
state of it:

**The dataset parsed cleanly and the baseline makes sense.**
6,112 real routes parsed exactly as published (2,888 from LA, 1,079 Seattle,
1,002 Chicago, 929 Boston, 214 Austin). A simple "just drive the shortest
path" baseline scores a plausible number on the official metric — the whole
pipeline, from raw JSON to Amazon's own scoring code, works end to end.

**Stage 1a (the driver-imitation model) needed a real fix to work at all.**
The pointer network, built exactly as a textbook pointer network, trained for
8 epochs and learned *nothing* — its loss sat at exactly the value you'd get
by guessing uniformly at random every time. The reason: its encoder only saw
features *of each stop*, never "how far is this candidate from where the
truck is right now" — which turns out to be the single most important signal
for predicting a driver's next move. Adding that signal fixed it completely:
71% of the time, the model now correctly predicts a real driver's actual next
stop (and this barely drops — to 70% — on a city it never saw during
training, which is a genuinely good generalisation result).

**One-step accuracy ≠ whole-route accuracy, and that itself is a finding.**
Despite 71% next-stop accuracy, greedily chaining those predictions together
only reproduces about 3% of a real driver's full route, because small early
errors compound. This is exactly the argument for this thesis's approach:
rather than using the learned model to *decide* the route directly (where
errors compound), it's used as a *cost* inside a proper optimizer, which can
still find a good overall route even if no single greedy step is perfect.

**The core claim shows up in the very first real measurement.** On a small
pilot of real held-out routes: a pure-imitation route scores well on
driver-likeness but is late (on average) by over an hour, with less than a 1-in-6
chance of hitting every delivery window. Moving the λ knob to 0.5 raises that
to a **perfect reliability score with zero expected lateness**, at the cost
of only about a **6–25% increase** in the driver-likeness penalty. Dropping
the behaviour signal entirely costs 4× as much in driver-likeness for no
further reliability gain — confirming that *both* halves of the model earn
their place.

**Real data forced one honest change to the model.** The literature usually
assumes a driver finishes an entire delivery zone before moving to the next
one. Checked against 1,500 real routes: only **38%** of real drivers actually
do that — most re-enter a zone they'd already left. Enforcing that rule as a
hard constraint, as the literature does, would make it *impossible* for the
optimizer to reproduce a majority of real drivers' actual sequences. So this
constraint is now **off by default** (kept, fully tested, as an optional
ablation to study directly) — a deliberate, data-justified deviation from the
textbook formulation, explained in full in `docs/DATA_NOTES.md`.

Full detail, including the exact numbers, the diagnosis of every bug found,
and the next steps, is in **`docs/BUILD_STATUS.md`** — treat it as the
project's live status board.

---

## 8. What's rigorously checked vs. what's still provisional

**Checked by the automated test suite (no data download needed, ~90 seconds to run):**
- The exact optimizer reproduces hand-worked-out answers on small toy problems,
  and agrees with brute-force enumeration across dozens of parameter
  combinations.
- The risk-calculation math matches hand-computed values exactly.
- The fast heuristic solver (used for all real experiments) always finds the
  same answer as the slow, provably-optimal solver on every problem small
  enough to check — including under every risk-model setting tested.
- The large-file data reader is checked byte-for-byte against Python's
  built-in JSON parser on deliberately tricky inputs, and verified to never
  load more into memory than a small buffer, regardless of file size.

**Measured on real data, but still a small pilot, not a finished result:**
Everything in section 7 above. The honest next step is running the full five
cities, the complete set of baselines and ablations, and a wider λ sweep — the
code for all of that already exists and is one command away
(`make all-folds`); it just hasn't been run at full scale yet.

---

## 9. Where to go next

- **New to the project?** You've just read it — now try the "Installing and
  running it" section above.
- **Want the math?** → [`docs/FORMULATION.md`](docs/FORMULATION.md)
- **Want to know exactly what's in the dataset, and the data-driven decisions
  that shaped the model?** → [`docs/DATA_NOTES.md`](docs/DATA_NOTES.md)
- **Want the full, continuously-updated status log — every bug found, every
  number measured, every open question?** → [`docs/BUILD_STATUS.md`](docs/BUILD_STATUS.md)

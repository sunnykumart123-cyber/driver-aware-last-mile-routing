# Notebooks

Exploratory only. **Nothing load-bearing lives here** (§8 of the build spec).

If a notebook produces something a thesis chapter cites, move the code into a
module and re-derive the number from `results/experiment_log.parquet`. The rule
for Phase 9 is that every reported number traces to a logged row, and a notebook
cell is not a traceable row.

Suggested exploratory notebooks, none required:

* `01_route_anatomy.ipynb` — distribution of stops, zones and windows per route
  and per metro; sanity-check the parser's output against the published dataset
  description.
* `02_travel_time_dispersion.ipynb` — the cross-route dispersion that Stage-1b
  models. This is the picture to have ready for the defence of the modelling
  decision in `docs/DATA_NOTES.md` §3.
* `03_behaviour_vs_shortest_path.ipynb` — how far $\sigma^\*$ is from the
  shortest-time tour; the motivation for a behaviour term at all.
* `04_lambda_frontier.ipynb` — read `results/rq2_price_of_robustness.parquet`
  and plot the frontier.

Load data through the library, not by re-parsing JSON:

```python
from core.io import iter_instances
from data.splits import model_split

train, val = model_split("data/processed", held_out_metro="Austin")
instances = train.instances(limit=50)
```

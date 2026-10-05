# Phase-ordered entry points (§9).  METRO selects the leave-one-metro-out fold.
PY     ?= python
METRO  ?= Austin
LIMIT  ?= 50

.PHONY: help install test lint data preprocess stage1 stage1-eval \
        pipeline phase6-check experiments rq2 rq3 all-folds clean

help:
	@echo "install        pip install -r requirements.lock"
	@echo "test           run the full unit + integration suite (no data needed)"
	@echo "data           download the LMRRC release into data/raw"
	@echo "preprocess     parse both splits into data/processed"
	@echo "stage1         train Stage-1a and Stage-1b for METRO=$(METRO)"
	@echo "stage1-eval    imitation accuracy + calibration diagrams"
	@echo "pipeline       solve LIMIT=$(LIMIT) held-out-metro routes end to end"
	@echo "phase6-check   ALNS vs. Benders certified gap on medium instances"
	@echo "experiments    baselines + ablations, all five folds"
	@echo "rq2 / rq3      the two sensitivity studies"

install:
	$(PY) -m pip install -r requirements.lock

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

data:
	$(PY) -m data.download --dest data/raw
	$(PY) -m data.download --dest data/raw --verify-only

preprocess:
	$(PY) -m data.preprocess --raw data/raw --out data/processed --split training
	$(PY) -m data.preprocess --raw data/raw --out data/processed --split evaluation

stage1:
	$(PY) train_behavior.py    --held-out-metro "$(METRO)" --model both
	$(PY) train_uncertainty.py --held-out-metro "$(METRO)" --calibrate-delta

stage1-eval:
	$(PY) eval_behavior.py    --held-out-metro "$(METRO)"
	$(PY) eval_uncertainty.py --held-out-metro "$(METRO)" --plot

pipeline:
	$(PY) -m pipeline.run_pipeline --held-out-metro "$(METRO)" --lam 0.5 --limit $(LIMIT)

# The comparison that justifies using the heuristic everywhere else.
phase6-check:
	$(PY) -m pipeline.run_pipeline --held-out-metro "$(METRO)" --method benders \
		--max-stops 80 --limit 20 --time-limit 300 --tag phase6_benders
	$(PY) -m pipeline.run_pipeline --held-out-metro "$(METRO)" --method alns \
		--max-stops 80 --limit 20 --time-limit 60  --tag phase6_alns

experiments:
	$(PY) -m experiments.ablations --all-folds --limit $(LIMIT)

rq2:
	$(PY) -m experiments.lambda_sweep_rq2 --all-folds --limit 25

rq3:
	$(PY) -m experiments.sensitivity_rq3 --all-folds --limit 20

all-folds: stage1 stage1-eval experiments rq2 rq3

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__

# Enterprise HR AI — task runner
# Every step of the build is reproducible from here; nothing is a manual ritual.

CONDA_ENV := hrai
# --no-capture-output streams progress live. Without it `conda run` buffers
# everything until the command exits, so a long pipeline step looks frozen.
RUN       := conda run --no-capture-output -n $(CONDA_ENV)
PY        := $(RUN) python
PIP       := conda run -n $(CONDA_ENV) pip

.DEFAULT_GOAL := help
.PHONY: help setup install lock test test-fast lint format check-invariant clean \
        profile validate clean-data relationships pipeline train explain crosswalk \
        intelligence transfer fairness monitor all api dashboard notebooks

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- phase 0 --
setup: install  ## Full first-time setup (deps + git hooks)
	$(RUN) pre-commit install
	@# macOS: XGBoost links against OpenMP, which pip does not provide.
	@$(RUN) python -c "import xgboost" 2>/dev/null \
	  || echo "NOTE: run 'conda install -c conda-forge llvm-openmp' for XGBoost"
	@echo "Setup complete. Run 'make test' to verify."

install:  ## Install runtime + dev dependencies
	$(PIP) install -r requirements-dev.txt

lock:  ## Freeze the resolved environment to requirements.lock.txt
	$(PIP) freeze --exclude-editable > requirements.lock.txt
	@echo "Locked $$(wc -l < requirements.lock.txt) packages."

# ------------------------------------------------------------------ checks --
test:  ## Run the full test suite with coverage gate
	$(RUN) pytest

test-fast:  ## Unit tests only, no coverage gate
	$(RUN) pytest -m unit --no-cov

lint:  ## Lint without modifying files
	$(RUN) ruff check src app tests scripts
	$(RUN) black --check src app tests scripts

format:  ## Auto-format and auto-fix
	$(RUN) black src app tests scripts
	$(RUN) ruff check --fix src app tests scripts

check-invariant:  ## Fail if the two employee populations are ever joined (F1/R1)
	$(PY) scripts/check_no_cross_population_join.py && echo "OK — no cross-population join."

# -------------------------------------------------------------- pipeline ---
profile:      ## Step 01 — profile all five source datasets
	$(PY) -m hrai.profiling.run

validate:     ## Step 02 — run Pandera schemas over the raw data
	$(PY) -m hrai.validation.run

clean-data:   ## Step 03 — produce data/processed/ from data/raw/
	$(PY) -m hrai.cleaning.run

relationships: ## Step 04 — document how the five tables connect
	$(PY) -m hrai.profiling.relationships

pipeline: profile validate clean-data relationships  ## Day 1 — steps 01-04

train:        ## Day 2 — steps 05-09: features, comparison, calibration, versioning
	$(PY) -m hrai.ml.train
	$(PY) -c "from hrai.ml.explain import write_global_report; write_global_report()"

crosswalk:    ## Step 11 — rebuild the role -> O*NET semantic crosswalk
	$(PY) -m hrai.skills.crosswalk

intelligence: ## Day 3 — steps 10-16: skills, gaps, recommendations, unified table
	$(PY) -m hrai.intelligence.employee_table

transfer:     ## Cross-population external validation
	$(PY) -m hrai.ml.transfer

fairness:     ## Fairness audit across protected attributes
	$(PY) -m hrai.ml.fairness

monitor:      ## Steps 25-27 — drift, performance, retraining decision
	$(PY) -m hrai.monitoring.drift

all: pipeline train transfer intelligence fairness monitor  ## Everything, in order

notebooks:    ## Execute every notebook top-to-bottom (verification)
	$(RUN) jupyter nbconvert --to notebook --execute --inplace \
	  --ExecutePreprocessor.timeout=900 notebooks/*.ipynb

# ------------------------------------------------------------------- apps --
api:          ## Run the FastAPI backend
	$(RUN) uvicorn app.main:app --reload --port 8000

dashboard:    ## Run the Streamlit dashboard
	$(RUN) streamlit run frontend/Home.py

clean:  ## Remove caches and build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov coverage.xml

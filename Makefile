# Battery RUL Prediction — one command per pipeline stage.
# `make all` runs the full chain: raw download -> processed -> features -> train -> evaluate.
# Each stage is also runnable on its own for iteration.

PYTHON := python3.11
VENV   := .venv
BIN    := $(VENV)/bin
PY     := $(BIN)/python

.DEFAULT_GOAL := help

.PHONY: help venv install download process features train evaluate report pipeline all eda test format lint app clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

venv:  ## Create the Python 3.11 virtual environment
	$(PYTHON) -m venv $(VENV)

install: venv  ## Install pinned dependencies + the project (editable) into the venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .  # makes `import src` work in notebooks and scripts

download:  ## Download raw Severson data into data/raw/ (gitignored)
	$(PY) -m src.data.download

process:  ## Parse raw .mat -> tidy per-cell/per-cycle parquet in data/processed/
	$(PY) -m src.data.preprocess

features:  ## Build variance/discharge/full feature tables
	$(PY) -m src.features.build_features

train:  ## Train baseline + ensemble models (seeded), persist to models/
	$(PY) -m src.models.train

evaluate:  ## Evaluate on held-out test split, write metrics + figures
	$(PY) -m src.models.evaluate

report:  ## Regenerate report figures/tables
	$(PY) -m src.models.evaluate --report-only

pipeline:  ## Run the full orchestrated pipeline script
	$(PY) scripts/run_pipeline.py

all: process features train evaluate  ## Full chain (assumes `make download` already run)

eda:  ## Launch Jupyter for the EDA / results notebooks
	$(BIN)/jupyter notebook notebooks/

test:  ## Run the test suite
	$(BIN)/pytest -q

format:  ## Auto-format with black + ruff
	$(BIN)/black src tests scripts
	$(BIN)/ruff check --fix src tests scripts

lint:  ## Lint without modifying files
	$(BIN)/ruff check src tests scripts
	$(BIN)/black --check src tests scripts

app:  ## Run the optional Streamlit demo
	$(BIN)/streamlit run app/streamlit_app.py

clean:  ## Remove caches and generated artifacts (keeps raw/processed data)
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__
	find . -name '*.pyc' -delete

.PHONY: install install-dev download-data etl features train test lint mlflow clean help

PYTHON := python
PIP    := pip
PYTEST := pytest

# ─── Setup ───────────────────────────────────────────────────────────────────

install:
	$(PIP) install -e ".[dev]"

install-all:
	$(PIP) install -e ".[all]"

install-dev:
	$(PIP) install -e ".[dev]"

# ─── Data ────────────────────────────────────────────────────────────────────

download-data:
	$(PYTHON) data/scripts/download_higgs.py

etl:
	$(PYTHON) -m src.ingestion.etl_pipeline

validate-data:
	$(PYTHON) -m src.ingestion.data_validator

# ─── Feature Engineering ─────────────────────────────────────────────────────

features:
	$(PYTHON) -m src.features.feature_store

# ─── Training ────────────────────────────────────────────────────────────────

train:
	$(PYTHON) -m src.pipeline.training_pipeline --model mlp

train-mlp:
	$(PYTHON) -m src.pipeline.training_pipeline --model mlp

train-bdt:
	$(PYTHON) -m src.pipeline.training_pipeline --model bdt

train-gnn:
	$(PYTHON) -m src.pipeline.training_pipeline --model gnn

train-all:
	$(PYTHON) -m src.pipeline.training_pipeline --model all

# ─── Serving ─────────────────────────────────────────────────────────────────

serve:
	uvicorn src.serving.api.main:app --host 0.0.0.0 --port 8000 --reload

# ─── Experiment Tracking ─────────────────────────────────────────────────────

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

# ─── Tests ───────────────────────────────────────────────────────────────────

test:
	$(PYTEST) tests/unit/ -v

test-all:
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing

test-integration:
	$(PYTEST) tests/integration/ -v

# ─── Code Quality ────────────────────────────────────────────────────────────

lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	ruff check --fix src/ tests/
	black src/ tests/

typecheck:
	mypy src/

# ─── Cleanup ─────────────────────────────────────────────────────────────────

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info

clean-data:
	rm -rf data/processed/*
	@echo "Cleaned processed data. Raw data preserved."

clean-mlruns:
	rm -rf mlruns/
	@echo "Cleaned MLflow runs."

# ─── Help ────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "Particle Physics Classifier — Makefile targets"
	@echo "─────────────────────────────────────────────"
	@echo "  make install         Install core + dev dependencies"
	@echo "  make download-data   Download UCI HIGGS dataset"
	@echo "  make etl             Run full ETL pipeline"
	@echo "  make features        Build feature store"
	@echo "  make train           Train MLP baseline (default)"
	@echo "  make train-all       Train all model architectures"
	@echo "  make serve           Start FastAPI inference server"
	@echo "  make mlflow          Start MLflow UI at :5000"
	@echo "  make test            Run unit tests"
	@echo "  make test-all        Run all tests with coverage"
	@echo "  make lint            Lint and format check"
	@echo "  make format          Auto-fix linting issues"
	@echo "  make clean           Remove cache and build artifacts"
	@echo ""

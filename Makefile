PYTHON := python

# ── Phony targets (not real files) ───────────────────────────────────────────
.PHONY: help setup format check clean lint isort test all reports data train 

# Default target when you just type "make"
help:
	@echo "Available commands:"
	@echo "  make setup    - Install dependencies"
	@echo "  make data     - Preprocess raw data"
	@echo "  make train    - Train the model"
	@echo "  make reports      - Run full pipeline (data → train)"
	@echo "  make format   - Auto-format code with Black"
	@echo "  make check    - Check formatting without changing files"
	@echo "  make clean    - Delete generated files"
	@echo "  make lint     - Run linter"
	@echo "  make isort    - Sort imports with isort"
	@echo "  make all      - Run everything (setup, format, check, lint, train)"
	@echo "  make test     - Run tests"

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

all: isort format check lint test reports 

# ── Pipeline (file targets — Make skips if already up to date) ───────────────
reports: reports/metrics.json

# Step 1: preprocess — reruns only if raw CSV changed
data/processed/clean_customers.csv: data/raw/customers.csv configs/config.toml src/data/preprocess.py
	$(PYTHON) src/data/preprocess.py --config configs/config.toml

# Friendly alias
data: data/processed/clean_customers.csv

# Step 2: train — reruns only if processed data or config changed
models/model.pkl reports/metrics.json: data/processed/clean_customers.csv configs/config.toml src/models/train.py
	$(PYTHON) src/models/train.py --config configs/config.toml

# Friendly alias
train: models/model.pkl

# ── Code quality ──────────────────────────────────────────────────────────────
format:
	black src/

check:
	black --check src/

lint:
	flake8 src/

isort:
	isort src/

test:
	pytest tests/ -v
# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	rm -rf data/processed/ models/ reports/ __pycache__ src/**/__pycache__

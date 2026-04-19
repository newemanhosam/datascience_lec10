PYTHON := python3

# ── Phony targets (not real files) ───────────────────────────────────────────
.PHONY: help setup format check clean

# Default target when you just type "make"
help:
	@echo "Available commands:"
	@echo "  make setup    - Install dependencies"
	@echo "  make data     - Preprocess raw data"
	@echo "  make train    - Train the model"
	@echo "  make all      - Run full pipeline (data → train)"
	@echo "  make format   - Auto-format code with Black"
	@echo "  make check    - Check formatting without changing files"
	@echo "  make clean    - Delete generated files"

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:
	$(PYTHON) -m pip install -r requirements.txt

# ── Pipeline (file targets — Make skips if already up to date) ───────────────
all: reports/metrics.json

# Step 1: preprocess — reruns only if raw CSV changed
data/processed/clean_customers.csv: data/raw/customers.csv configs/config.toml
	$(PYTHON) src/data/preprocess.py --config configs/config.toml

# Friendly alias
data: data/processed/clean_customers.csv

# Step 2: train — reruns only if processed data or config changed
models/model.pkl reports/metrics.json: data/processed/clean_customers.csv configs/config.toml
	$(PYTHON) src/models/train.py --config configs/config.toml

# Friendly alias
train: models/model.pkl

# ── Code quality ──────────────────────────────────────────────────────────────
format:
	black src/

check:
	black --check src/

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	rm -rf data/processed/ models/ reports/ __pycache__ src/**/__pycache__

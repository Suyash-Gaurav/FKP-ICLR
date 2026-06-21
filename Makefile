# ─────────────────────────────────────────────────────────────────────────────
# FKP — Feature Knowledge Projection Makefile
# Usage: make <target>
# ─────────────────────────────────────────────────────────────────────────────

PYTHON ?= python
PIP    ?= pip
PYTEST ?= pytest

.PHONY: help install install-dev test test-fast lint format typecheck \
        reproduce-main-table reproduce-ablation export-c clean docker-build

# ── Default target ────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "FKP — Feature Knowledge Projection"
	@echo "======================================"
	@echo ""
	@echo "  make install            Install core dependencies"
	@echo "  make install-dev        Install all dev + model dependencies"
	@echo "  make test               Run full pytest suite with coverage"
	@echo "  make test-fast          Run tests without coverage (faster)"
	@echo "  make lint               Run ruff linter"
	@echo "  make format             Auto-format with ruff + black"
	@echo "  make typecheck          Run mypy type checker"
	@echo "  make reproduce-main-table  Reproduce Table 1 results"
	@echo "  make reproduce-ablation    Reproduce Table 2 (ablation study)"
	@echo "  make export-c           Export weights to C header (set WEIGHTS_DIR)"
	@echo "  make clean              Remove build artifacts and caches"
	@echo "  make docker-build       Build the reproducibility Docker image"
	@echo ""

# ── Installation ──────────────────────────────────────────────────────────────
install:
	$(PIP) install -e ".[models,data]"

install-dev:
	$(PIP) install -e ".[all]"

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	$(PYTEST) tests/ -v --cov=fkp --cov-report=term-missing --cov-report=html

test-fast:
	$(PYTEST) tests/ -v --no-header -q

test-math:
	@echo "Running critical math unit tests only..."
	$(PYTEST) tests/test_ridge_dual.py tests/test_zca.py tests/test_certificate.py -v

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/ scripts/
	ruff check edge_deployment/export_to_c.py

format:
	ruff check --fix src/ tests/ scripts/
	black src/ tests/ scripts/

typecheck:
	mypy src/fkp/ --ignore-missing-imports

# ── Reproducibility ───────────────────────────────────────────────────────────
reproduce-main-table:
	@echo "Reproducing Table 1: Main results..."
	@for teacher in resnet50 vit_b16 clip dinov2_small mobilenetv3; do \
	    for dataset in plantvillage cifar100; do \
	        echo "Running: teacher=$$teacher, dataset=$$dataset"; \
	        $(PYTHON) scripts/run_calibration.py \
	            --teacher $$teacher \
	            --dataset $$dataset \
	            --config configs/experiments/main_table.yaml \
	            --output_dir outputs/main_table/$${teacher}_$${dataset} \
	            --seed 42; \
	    done; \
	done
	$(PYTHON) scripts/generate_figures.py --results_dir outputs/main_table/ --output_dir figures/

reproduce-ablation:
	@echo "Reproducing Table 2: Ablation study..."
	$(PYTHON) scripts/run_calibration.py \
	    --config configs/experiments/ablation_zca.yaml \
	    --output_dir outputs/ablation_zca \
	    --seed 42

# ── Edge deployment ───────────────────────────────────────────────────────────
# Usage: make export-c WEIGHTS_DIR=outputs/plantvillage_resnet50
WEIGHTS_DIR ?= outputs/run
export-c:
	$(PYTHON) edge_deployment/export_to_c.py \
	    --weights_dir $(WEIGHTS_DIR) \
	    --output edge_deployment/include/fkp_inference.h \
	    --model_name FKP

compile-c-sim:
	@echo "Building C simulation binary..."
	mkdir -p build_c
	cmake -B build_c -S edge_deployment/
	cmake --build build_c/
	@echo "Run: ./build_c/fkp_sim"

# ── Data ─────────────────────────────────────────────────────────────────────
download-data:
	bash data/download_datasets.sh

# ── Docker ───────────────────────────────────────────────────────────────────
docker-build:
	docker build -t fkp-iclr2027:latest -f docker/Dockerfile .

docker-run:
	docker run --rm -it \
	    -v $(PWD)/data:/workspace/FKP/data \
	    -v $(PWD)/outputs:/workspace/FKP/outputs \
	    fkp-iclr2027:latest bash

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	rm -rf build_c/ dist/ 2>/dev/null || true
	@echo "Cleaned."

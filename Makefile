# AML-GNN Project Makefile
# Use 'make help' to see available commands

.PHONY: help setup ingest train deploy test clean

# Variables
PYTHON = python
PIP = pip
SCRIPTS_DIR = scripts
DATA_DIR = data

help:
	@echo "Available commands:"
	@echo "  setup    : Install dependencies and setup environment"
	@echo "  ingest   : Populat Neo4j AuraDB with dataset"
	@echo "  train    : Run model training pipeline"
	@echo "  predict  : Run inference on trained model"
	@echo "  api      : Start the FastAPI inference service"
	@echo "  gcp-deploy: Deploy the application to Google Cloud Run"
	@echo "  docker   : Build and start containerized services"
	@echo "  clean    : Remove temporary files and caches"

setup:
	$(PIP) install -r requirements.txt
	$(PYTHON) -m scripts.init_packages

ingest:
	$(PYTHON) scripts/populate_neo4j.py

train:
	$(PYTHON) scripts/run_training.py --epochs 100 --model sage

api:
	uvicorn src.interfaces.api.main:app --host 0.0.0.0 --port 8000 --reload

api-test:
	$(PYTHON) scripts/api_smoke_test.py

gui:
	$(PYTHON) -m streamlit run src/interfaces/frontend/app.py

predict:
	$(PYTHON) scripts/deploy_predictions.py --model_path models/sage_best.pt

test:
	$(PYTHON) -m pytest tests/

gcp-deploy:
	bash scripts/deploy_gcp.sh

docker:
	docker compose -f deploy/docker-compose.yml up --build

clean:
	rm -rf __pycache__ .pytest_cache .ipynb_checkpoints
	find . -type d -name "__pycache__" -exec rm -rf {} +

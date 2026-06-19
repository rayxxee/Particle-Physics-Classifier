# Particle Physics Event Classifier

> End-to-end ML platform for particle physics event classification on real CERN collision data.

## Architecture

```
CERN Open Data / UCI HIGGS
        │
        ▼
  Data Ingestion & ETL     ← uproot, awkward-array, ROOT file parsing
        │
        ▼
  Feature Engineering      ← Invariant mass, rapidity, ΔR, jet substructure
        │
        ▼
     Model Zoo              ← MLP | BDT | GNN | Transformer | Normalizing Flow
        │
        ▼
 Experiment Tracking        ← MLflow: params, metrics, artifacts, model registry
        │
        ▼
  FastAPI REST API          ← /predict /explain /compare /retrain
        │
        ▼
  React Dashboard           ← Live inference | ROC viewer | 3D event display
        │
        ▼
Prometheus + Grafana        ← Latency, score drift, auto-retraining
```

## Quick Start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Download data
make download-data

# 3. Run ETL pipeline
make etl

# 4. Train MLP baseline
make train

# 5. View results in MLflow
make mlflow         # → open http://localhost:5000
```

## Module Map

| Module | Path | Status |
|--------|------|--------|
| 1 — ETL & Ingestion | `src/ingestion/` | ✅ Phase 1 |
| 2 — Feature Engineering | `src/features/` | ✅ Phase 1 |
| 3a — MLP Baseline | `src/models/mlp/` | ✅ Phase 1 |
| 3b — BDT (XGBoost/LightGBM) | `src/models/bdt/` | 🔜 Phase 2 |
| 3c — Graph Neural Network | `src/models/gnn/` | 🔜 Phase 2 |
| 3d — Particle Transformer | `src/models/transformer/` | 🔜 Phase 2 |
| 3e — Normalizing Flow | `src/models/normalizing_flow/` | 🔜 Phase 2 |
| 4 — Experiment Tracking | `src/experiment_tracking/` | ✅ Phase 1 |
| 5 — Evaluation & Physics Metrics | `src/evaluation/` | 🔜 Phase 2 |
| 6 — Explainability (SHAP/LIME) | `src/explainability/` | 🔜 Phase 3 |
| 7 — FastAPI Backend | `src/serving/` | 🔜 Phase 3 |
| 8 — React Dashboard | `frontend/` | 🔜 Phase 4 |
| 9 — Monitoring (Prometheus/Grafana) | `src/monitoring/` | 🔜 Phase 5 |
| 10 — Airflow Pipeline | `pipeline/` | 🔜 Phase 5 |
| 11 — Tests | `tests/` | ✅ Phase 1 (unit) |
| 12 — Containerization | `infra/` | 🔜 Phase 6 |

## Technology Stack

| Layer | Technology |
|-------|------------|
| Data reading | uproot, awkward-array, pandas |
| Data validation | Pandera |
| Data versioning | DVC |
| Feature engineering | NumPy, custom physics library |
| ML — deep learning | PyTorch |
| ML — graphs | PyTorch Geometric |
| ML — boosted trees | XGBoost, LightGBM |
| Hyperparameter search | Optuna |
| Experiment tracking | MLflow |
| Config management | OmegaConf / YAML |
| Explainability | SHAP, LIME |
| API | FastAPI, Pydantic v2 |
| Frontend | React, Recharts, Three.js |
| Monitoring | Prometheus, Grafana, Evidently AI |
| Orchestration | Apache Airflow |
| Testing | pytest |
| Containerization | Docker Compose, Kubernetes |
| CI/CD | GitHub Actions |

## Results (Phase 1)

| Model | Val AUC | Training Time |
|-------|---------|---------------|
| MLP baseline | > 0.80 | — |

## Citation

UCI HIGGS dataset: Baldi, P. et al. (2014). Searching for Exotic Particles in High-Energy Physics with Deep Learning. Nature Communications, 5, 4308.

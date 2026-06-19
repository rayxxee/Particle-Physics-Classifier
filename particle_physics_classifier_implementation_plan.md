# Particle Physics Event Classifier — Industry-Level Implementation Plan
### For Antigravity Build Team | Modular, Production-Grade, Full-Stack ML System

---

## Executive Summary

This is not just a model. It is a full **ML platform** built around particle physics — covering data ingestion from CERN, multi-architecture model comparison, explainability, a REST API, a monitoring dashboard, a web UI, automated retraining pipelines, and reproducible experiment tracking. Every component is modular and independently deployable.

**Target Outcome:** A portfolio-grade, production-ready system that mirrors what real HEP (High Energy Physics) ML teams at CERN, Fermilab, and DESY actually build.

---

## System Architecture Overview

```
CERN Open Data / UCI HIGGS
        │
        ▼
┌─────────────────────┐
│   Data Ingestion    │  ← uproot, awkward-array, ROOT file parsing
│   & ETL Pipeline   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Feature            │  ← Physics-aware feature engineering
│  Engineering Layer  │     (invariant mass, rapidity, deltaR, etc.)
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                   Model Zoo                              │
│  MLP │ BDT/XGBoost │ GNN │ Transformer │ Normalizing   │
│      │             │     │             │ Flows          │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐     ┌─────────────────────┐
│  Experiment         │     │  Explainability      │
│  Tracking (MLflow)  │     │  Layer (SHAP, LIME)  │
└────────┬────────────┘     └──────────┬──────────┘
         │                             │
         ▼                             ▼
┌─────────────────────────────────────────────────┐
│              FastAPI REST Backend                │
│    /predict  /compare  /explain  /retrain       │
└────────┬────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│         React Frontend Dashboard                 │
│   Live inference │ ROC viewer │ Feature plots   │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│         Monitoring & Alerting                    │
│   Prometheus + Grafana │ Data drift detection   │
└─────────────────────────────────────────────────┘
```

---

## Repository Structure

```
particle-classifier/
│
├── data/
│   ├── raw/                        # Original .root / .csv files
│   ├── processed/                  # Cleaned, normalized parquet files
│   ├── schemas/                    # Feature schemas (JSON)
│   └── scripts/
│       ├── download_cern.py        # Pulls from opendata.cern.ch via API
│       ├── download_higgs.py       # UCI HIGGS dataset downloader
│       └── validate_data.py        # Schema + integrity checks
│
├── src/
│   ├── ingestion/
│   │   ├── root_reader.py          # uproot-based ROOT file reader
│   │   ├── etl_pipeline.py         # Full ETL orchestration
│   │   └── data_validator.py       # Great Expectations validation
│   │
│   ├── features/
│   │   ├── physics_features.py     # Invariant mass, deltaR, rapidity
│   │   ├── high_level_features.py  # Jet substructure, b-tagging vars
│   │   ├── low_level_features.py   # Raw 4-vector components
│   │   └── feature_store.py        # Feast-based feature store interface
│   │
│   ├── models/
│   │   ├── base_model.py           # Abstract base class all models inherit
│   │   ├── mlp/
│   │   │   ├── model.py            # Deep MLP with BatchNorm + Dropout
│   │   │   ├── config.py           # Hyperparameter config (dataclass)
│   │   │   └── trainer.py          # Training loop, scheduler, early stop
│   │   ├── bdt/
│   │   │   ├── model.py            # XGBoost + LightGBM wrapper
│   │   │   ├── config.py
│   │   │   └── trainer.py
│   │   ├── gnn/
│   │   │   ├── model.py            # PyTorch Geometric graph network
│   │   │   ├── graph_builder.py    # Converts events to graphs (nodes=particles)
│   │   │   ├── config.py
│   │   │   └── trainer.py
│   │   ├── transformer/
│   │   │   ├── model.py            # Particle Transformer (attention over jets)
│   │   │   ├── config.py
│   │   │   └── trainer.py
│   │   └── normalizing_flow/
│   │       ├── model.py            # Density estimation for anomaly detection
│   │       ├── config.py
│   │       └── trainer.py
│   │
│   ├── explainability/
│   │   ├── shap_explainer.py       # SHAP values for all model types
│   │   ├── lime_explainer.py       # LIME local explanations
│   │   ├── attention_viz.py        # Attention map visualizer (Transformer)
│   │   └── physics_interpretability.py  # Map ML features back to physics
│   │
│   ├── evaluation/
│   │   ├── metrics.py              # AUC, significance, signal efficiency
│   │   ├── roc_analysis.py         # ROC curves with uncertainty bands
│   │   ├── calibration.py          # Probability calibration (Platt, isotonic)
│   │   ├── benchmark.py            # Head-to-head model comparisons
│   │   └── physics_metrics.py      # S/sqrt(B), punzi figure of merit
│   │
│   ├── experiment_tracking/
│   │   ├── mlflow_logger.py        # MLflow run logging wrapper
│   │   ├── wandb_logger.py         # Weights & Biases logger (optional)
│   │   └── experiment_registry.py  # Stores best model per task
│   │
│   ├── serving/
│   │   ├── api/
│   │   │   ├── main.py             # FastAPI app entrypoint
│   │   │   ├── routes/
│   │   │   │   ├── predict.py      # POST /predict — single event inference
│   │   │   │   ├── batch.py        # POST /batch — bulk inference
│   │   │   │   ├── explain.py      # POST /explain — SHAP for one event
│   │   │   │   ├── compare.py      # GET /compare — model leaderboard
│   │   │   │   └── health.py       # GET /health — liveness probe
│   │   │   ├── schemas.py          # Pydantic request/response models
│   │   │   └── model_loader.py     # Loads model from registry at startup
│   │   └── inference_engine.py     # Batching, caching, async inference
│   │
│   ├── monitoring/
│   │   ├── data_drift.py           # Evidently AI drift detection
│   │   ├── model_drift.py          # Prediction distribution monitoring
│   │   ├── prometheus_metrics.py   # Latency, throughput, confidence histograms
│   │   └── alerting.py             # Threshold-based alert triggers
│   │
│   └── pipeline/
│       ├── training_pipeline.py    # End-to-end: ingest → features → train → evaluate
│       ├── retraining_trigger.py   # Auto-retrain on drift detection
│       └── airflow_dag.py          # Apache Airflow DAG definition
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx       # Main metrics overview
│   │   │   ├── LiveInference.jsx   # Submit event, see prediction + SHAP
│   │   │   ├── ModelCompare.jsx    # Side-by-side ROC / AUC comparison
│   │   │   ├── DataExplorer.jsx    # Feature distributions, correlations
│   │   │   └── Monitoring.jsx      # Drift alerts, latency charts
│   │   └── components/
│   │       ├── ROCChart.jsx
│   │       ├── SHAPWaterfall.jsx
│   │       ├── FeatureHeatmap.jsx
│   │       └── EventViewer.jsx     # 3D particle event display
│   └── package.json
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_comparison.ipynb
│   ├── 04_explainability.ipynb
│   ├── 05_physics_validation.ipynb
│   └── 06_gnn_deep_dive.ipynb
│
├── tests/
│   ├── unit/
│   │   ├── test_features.py
│   │   ├── test_models.py
│   │   └── test_metrics.py
│   ├── integration/
│   │   ├── test_api.py
│   │   └── test_pipeline.py
│   └── data/
│       └── sample_events.csv       # 1000-row fixture for tests
│
├── infra/
│   ├── docker/
│   │   ├── Dockerfile.api
│   │   ├── Dockerfile.frontend
│   │   └── Dockerfile.worker
│   ├── docker-compose.yml          # Full local stack
│   ├── k8s/                        # Kubernetes manifests
│   │   ├── api-deployment.yaml
│   │   ├── mlflow-deployment.yaml
│   │   └── grafana-deployment.yaml
│   └── terraform/                  # Optional cloud provisioning (AWS/GCP)
│
├── configs/
│   ├── mlp_default.yaml
│   ├── gnn_default.yaml
│   ├── xgboost_default.yaml
│   └── system.yaml                 # Global settings (paths, DB, ports)
│
├── .github/
│   └── workflows/
│       ├── ci.yml                  # Lint, test, build on every PR
│       └── cd.yml                  # Deploy on merge to main
│
├── pyproject.toml                  # Project metadata + deps
├── Makefile                        # make train / make serve / make test
├── README.md
└── docs/
    ├── architecture.md
    ├── physics_background.md       # Explains the physics to non-experts
    ├── api_reference.md
    └── model_cards/
        ├── mlp_model_card.md       # Model card per architecture
        └── gnn_model_card.md
```

---

## Component-by-Component Build Plan

---

### Module 1 — Data Ingestion & ETL

**Goal:** Ingest real CERN collision data, validate it, and store it in a clean, versioned format.

**Data Sources:**
- Primary: CERN Open Data Portal — CMS Run 2 datasets (opendata.cern.ch)
- Secondary: UCI HIGGS dataset (11M events, 28 features, real simulation data)
- Tertiary: HiggsML challenge dataset (Kaggle, with weights and systematic uncertainties)

**Key files to build:**

`root_reader.py`
- Use `uproot` to open `.root` files without a native ROOT install
- Extract `TTree` branches: lepton 4-vectors (pT, eta, phi, mass), jet b-tags, missing ET
- Convert to `awkward` arrays, then to pandas DataFrames
- Handle variable-length arrays (jets per event is not fixed) using padding/masking

`etl_pipeline.py`
- Read raw files → apply quality cuts (e.g. pT > 25 GeV, |eta| < 2.5)
- Split into train/val/test with stratification on label
- Save to parquet with versioned filenames (SHA of config)

`data_validator.py`
- Use Great Expectations to assert: no nulls, value ranges, class balance within bounds
- Fail loudly if validation breaks — never silently pass bad data downstream

**Physics quality cuts to implement:**
```
lepton pT > 25 GeV
|lepton eta| < 2.4
missing ET > 20 GeV
at least 2 jets with pT > 30 GeV
at least 1 b-tagged jet
```

---

### Module 2 — Feature Engineering

**Goal:** Go beyond raw variables. Compute derived physics features that carry real discriminating power.

**Low-level features (raw):**
- Lepton pT, eta, phi
- Jet 4-vectors for up to 4 jets
- Missing transverse energy (MET) magnitude and phi
- b-tagging discriminant scores

**High-level features (computed):**

`physics_features.py`
```python
def invariant_mass(pt1, eta1, phi1, m1, pt2, eta2, phi2, m2):
    # Full 4-vector addition → invariant mass
    # This is the core observable in HEP — a peak in m_inv is how you discover a particle

def delta_r(eta1, phi1, eta2, phi2):
    # Angular separation: sqrt(dEta^2 + dPhi^2)
    # Measures how "close" two particles are in the detector

def transverse_mass(lepton_pt, lepton_phi, met, met_phi):
    # Proxy for W boson mass

def rapidity(E, pz):
    # Lorentz-invariant measure of "forward-ness"

def ht_scalar(jets):
    # Scalar sum of all jet pTs — measures event "activity"

def centrality(ht, E_total):
    # Ratio of HT to total energy
```

**Jet substructure features** (advanced — very impressive):
```python
def n_subjettiness(jet, N):
    # tau_N: measures how N-prong a jet is
    # tau21 = tau2/tau1 discriminates W/Z jets from QCD

def jet_mass(jet_constituents):
    # Mass of reconstructed jet from its particles

def energy_correlation_functions(jet):
    # C2, D2 — powerful for boosted object tagging
```

**Feature store interface:**
- Wrap all features behind a `FeatureStore` class
- Features are computed once and cached to disk
- New model training reads from store, not raw data

---

### Module 3 — Model Zoo

Build 5 architectures. Each inherits from `BaseModel` and exposes identical `.fit()`, `.predict_proba()`, `.save()`, `.load()` interfaces so they are fully interchangeable.

---

#### 3a. Deep MLP (Baseline)

```
Input (28) → Linear(512) → BN → ReLU → Dropout(0.3)
           → Linear(256) → BN → ReLU → Dropout(0.3)
           → Linear(128) → BN → ReLU → Dropout(0.2)
           → Linear(64)  → ReLU
           → Linear(1)   → Sigmoid
```

- Optimizer: AdamW with weight decay 1e-4
- Scheduler: CosineAnnealingLR
- Early stopping: patience=10 on val AUC
- Mixed precision training (torch.cuda.amp) for speed
- Expected AUC: ~0.81

---

#### 3b. Boosted Decision Tree (BDT) — the traditional HEP baseline

- XGBoost with depth=6, 1000 estimators
- LightGBM as second BDT variant
- Optuna hyperparameter search (100 trials)
- This is what physicists actually used pre-deep-learning — beat it and you have a story

---

#### 3c. Graph Neural Network (GNN) — the state of the art in HEP

Every collision event is a **graph**:
- Nodes = particles (leptons, jets, MET)
- Edges = relationships (deltaR between all pairs, or k-nearest-neighbors)
- Node features = 4-vector components + b-tag score
- Edge features = deltaR, invariant mass of pair

Architecture:
```
Input graph
→ Edge feature embedding (Linear → ReLU) × 2
→ Message passing: EdgeConv layers × 3   (or GATConv with attention)
→ Global pooling (mean + max concatenated)
→ MLP classifier head
```

Use `torch_geometric` (PyG). The graph builder is a separate module — you can swap GNN layers without touching the data.

Expected AUC: ~0.84 (beats MLP)

---

#### 3d. Particle Transformer

Treat the event as a **sequence** of particles and apply self-attention:
- Each particle = token with embedding of its 4-vector features
- Positional encoding: none (physics is permutation-invariant) — use learned class token instead
- 4 transformer encoder layers, 8 attention heads
- CLS token aggregates event-level representation
- MLP head on CLS token → binary classification

This mirrors the **ParT** (Particle Transformer) architecture used at CMS/ATLAS.
Expected AUC: ~0.85

---

#### 3e. Normalizing Flow (Anomaly Detection mode)

Different task: instead of supervised classification, model the density of background events. Signal events appear as **low-probability outliers**.

- RealNVP or MAF (Masked Autoregressive Flow)
- Trained only on background events
- At inference: compute log-probability for new events
- Low log-prob → anomalous → likely signal

This enables **model-agnostic new physics searches** — you don't need to know what the signal looks like.

---

### Module 4 — Experiment Tracking

**MLflow setup:**
- Every training run logs: hyperparameters, dataset version (SHA), all metrics per epoch
- Artifacts stored: model weights, SHAP plots, ROC curves, confusion matrices
- Model registry: tag best model per architecture as "Production"
- UI runs at `localhost:5000`

**Run comparison workflow:**
1. Train all 5 architectures with `make train-all`
2. Open MLflow UI — compare AUC, training time, memory
3. Promote winner to Production tag
4. API loads Production model automatically at startup

**Optuna integration:**
- Each architecture has an `optimize()` function
- Runs N trials, each logged as an MLflow child run
- Best hyperparameters saved to `configs/<model>_best.yaml`

---

### Module 5 — Evaluation & Physics Metrics

Beyond standard ML metrics, implement HEP-specific evaluation:

**Standard ML metrics:**
- AUC-ROC with bootstrap uncertainty bands (N=1000 bootstrap samples → 68% CI)
- AUC-PR (Precision-Recall) — more informative when classes are imbalanced
- Log-loss, Brier score
- Calibration curves (reliability diagrams)

**Physics-specific metrics:**

```python
def signal_significance(s, b):
    # Z = S / sqrt(B) — how many sigma above background
    # This is what physicists actually optimize
    return s / np.sqrt(b)

def punzi_figure_of_merit(signal_eff, b, sigma=5):
    # Punzi FOM: maximizes discovery potential
    # FOM = signal_eff / (sigma/2 + sqrt(B))
    return signal_eff / (sigma / 2 + np.sqrt(b))

def neyman_pearson_efficiency_curve(tpr, fpr, target_fpr=0.01):
    # At fixed background rejection (1/FPR = 100),
    # what is signal efficiency?
    # This is how physicists quote working points
    pass

def rejection_vs_efficiency(tpr, fpr):
    # 1/FPR vs TPR curve — standard HEP plot
    # Often shown on log scale
    pass
```

**Calibration:**
- Raw neural net scores are not well-calibrated probabilities
- Apply isotonic regression or Platt scaling on val set
- Plot reliability diagrams before/after calibration
- Calibrated model is what gets deployed

---

### Module 6 — Explainability Layer

Every prediction must be explainable. This is what separates a research tool from a trusted system.

**SHAP (SHapley Additive exPlanations):**
- TreeExplainer for BDT (fast, exact)
- DeepExplainer for MLP/Transformer (approximate)
- GradientExplainer as fallback
- Output: per-feature contribution for every prediction
- Global: SHAP summary plots, dependence plots, interaction plots

**Physics interpretation module:**
```python
def interpret_shap_physics(shap_values, feature_names):
    """
    Map top SHAP features back to physics.
    e.g. "m_bb has high importance" → "b-jet pair invariant mass
    discriminates Higgs → bb decay from QCD background"
    Returns human-readable physics explanation.
    """
```

**Attention visualization (Transformer only):**
- Extract attention weights from each head
- Visualize which particle pairs the model attends to
- Overlay on event display — see which jets "talk to" which leptons

**LIME (Local Interpretable Model-agnostic Explanations):**
- Fits a local linear model around any single prediction
- More model-agnostic than SHAP, useful as cross-check

---

### Module 7 — FastAPI REST Backend

**Endpoints:**

```
POST /v1/predict
  Body: { "features": [...28 floats...], "model": "gnn" }
  Returns: { "score": 0.87, "label": "signal", "confidence": "high" }

POST /v1/predict/batch
  Body: { "events": [[...], [...], ...], "model": "transformer" }
  Returns: { "predictions": [...], "latencies_ms": [...] }

POST /v1/explain
  Body: { "features": [...], "model": "mlp" }
  Returns: { "shap_values": {...}, "top_features": [...], "physics_notes": [...] }

GET /v1/compare
  Returns: { "leaderboard": [{ "model": "gnn", "auc": 0.843, ... }, ...] }

GET /v1/models
  Returns: list of available models with metadata

POST /v1/retrain
  Body: { "model": "mlp", "data_version": "v3" }
  Triggers async retraining job, returns job_id

GET /v1/health
  Returns: { "status": "ok", "loaded_model": "gnn", "uptime_s": 3600 }
```

**Production features:**
- Pydantic v2 request/response validation — bad input fails fast with clear errors
- Async inference with `asyncio` — multiple requests don't block each other
- In-memory LRU cache for repeated identical inputs
- Request ID in every response for traceability
- Rate limiting middleware (slowapi)
- OpenAPI docs auto-generated at `/docs`

---

### Module 8 — React Frontend Dashboard

**Pages:**

1. **Dashboard** — KPI cards (best AUC, total events processed, uptime), recent predictions feed

2. **Live Inference** — input form for a particle event, submit to API, see:
   - Prediction score as a gauge
   - SHAP waterfall chart (which features pushed it signal vs background)
   - Confidence level and physics interpretation text

3. **Model Comparison** — overlay ROC curves for all 5 architectures, sortable leaderboard table with AUC, training time, inference latency, parameter count

4. **Data Explorer** — feature distribution histograms (signal vs background overlaid), correlation heatmap, class balance pie chart, feature importance ranking

5. **Monitoring** — real-time charts for inference latency (p50/p95/p99), data drift score over time, alert history

**3D Event Display (bonus):**
- Use Three.js to render a simplified detector cross-section
- Show particles as colored tracks radiating from the beam axis
- Color-code by type: electrons (green), muons (red), jets (yellow cones), MET (dashed arrow)

---

### Module 9 — Monitoring & Data Drift Detection

**Prometheus metrics (exposed at `/metrics`):**
- `inference_latency_seconds` — histogram by model
- `prediction_score_histogram` — distribution of output scores
- `requests_total` — counter by endpoint and status code
- `model_auc_gauge` — current deployed model AUC

**Grafana dashboards:**
- System health: latency percentiles, error rate, request throughput
- ML health: score distribution over time, drift score, calibration drift

**Evidently AI drift detection:**
```python
# Run daily on a window of recent predictions
def detect_drift(reference_data, current_data):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    if report.as_dict()["metrics"][0]["result"]["dataset_drift"]:
        trigger_retraining_alert()
```

**Retraining trigger:**
- If drift score > threshold: create MLflow run, retrain best model, run eval
- If new model AUC > production model AUC by > 0.005: auto-promote
- Notify via webhook (Slack / email)

---

### Module 10 — Pipeline Orchestration

**Apache Airflow DAG:**
```
download_data → validate_data → compute_features
    → [train_mlp, train_bdt, train_gnn, train_transformer]  (parallel)
    → evaluate_all → compare_models → promote_best → deploy_api
```

Alternatively use **Prefect** (lighter, easier local setup) or **DVC pipelines** for pure ML lineage.

**DVC (Data Version Control):**
- Track data files and model artifacts in Git-compatible way
- `dvc repro` reruns only changed stages
- `dvc push/pull` syncs artifacts to S3/GCS
- Every experiment is fully reproducible from a single commit

---

### Module 11 — Testing Strategy

**Unit tests (pytest):**
- `test_features.py` — verify physics formulas (invariant mass of known particle, check deltaR = 0 for identical particles)
- `test_models.py` — model forward pass with dummy input, correct output shape
- `test_metrics.py` — AUC of perfect predictor = 1.0, random = 0.5

**Integration tests:**
- `test_api.py` — spin up FastAPI TestClient, hit every endpoint, check response schemas
- `test_pipeline.py` — run full pipeline on 1000-event fixture, verify output files exist and metrics are in expected range

**CI/CD (GitHub Actions):**
- On every PR: lint (ruff, black), type check (mypy), run all unit tests
- On merge to main: run integration tests, build Docker images, push to registry

---

### Module 12 — Containerization & Deployment

**Docker Compose (local full stack):**
```yaml
services:
  api:         # FastAPI inference server
  mlflow:      # Experiment tracking UI
  postgres:    # MLflow backend store
  minio:       # S3-compatible artifact store
  prometheus:  # Metrics collection
  grafana:     # Dashboard UI
  frontend:    # React app
  airflow:     # Pipeline orchestration (optional)
```

Single command to start everything: `docker-compose up`

**Kubernetes (production):**
- API deployment with horizontal pod autoscaling (HPA) on CPU/latency
- MLflow as a separate deployment
- Secrets managed via Kubernetes Secrets or Vault
- Ingress with TLS termination

---

## Technology Stack Summary

| Layer | Technology |
|---|---|
| Data reading | uproot, awkward-array, pandas |
| Data validation | Great Expectations |
| Data versioning | DVC |
| Feature engineering | NumPy, custom physics library |
| ML — deep learning | PyTorch, PyTorch Geometric |
| ML — boosted trees | XGBoost, LightGBM |
| Hyperparameter search | Optuna |
| Experiment tracking | MLflow |
| Explainability | SHAP, LIME |
| API | FastAPI, Pydantic v2, Uvicorn |
| Frontend | React, Recharts, Three.js |
| Monitoring | Prometheus, Grafana, Evidently AI |
| Orchestration | Apache Airflow or Prefect |
| Testing | pytest, pytest-asyncio |
| Containerization | Docker, Docker Compose, Kubernetes |
| CI/CD | GitHub Actions |

---

## Build Order for Antigravity

Build in this sequence — each phase delivers a working, demonstrable system:

**Phase 1 (Week 1–2): Core Data & Baseline Model**
→ Module 1 (ETL), Module 2 (Features), Module 3a (MLP)
→ Deliverable: working training pipeline, AUC > 0.80, MLflow tracking

**Phase 2 (Week 3–4): Model Zoo**
→ Module 3b (BDT), Module 3c (GNN), Module 3d (Transformer)
→ Deliverable: 4 architectures trained and compared, leaderboard

**Phase 3 (Week 5): API + Explainability**
→ Module 6 (SHAP), Module 7 (FastAPI)
→ Deliverable: running REST API with /predict and /explain endpoints

**Phase 4 (Week 6): Frontend**
→ Module 8 (React Dashboard)
→ Deliverable: full web UI, live inference, ROC viewer

**Phase 5 (Week 7): Monitoring + Pipeline**
→ Module 9 (Drift detection), Module 10 (Airflow)
→ Deliverable: Grafana dashboards, automated retraining

**Phase 6 (Week 8): Polish**
→ Module 11 (Tests), Module 12 (Docker), docs, model cards
→ Deliverable: production-ready, fully containerized, CI/CD green

---

## What Goes on the CV / README

> *"End-to-end ML platform for particle physics event classification on real CERN collision data. Implements 5 model architectures (MLP, BDT, GNN, Particle Transformer, Normalizing Flow) with automated hyperparameter search via Optuna, experiment tracking via MLflow, and SHAP-based physics-interpretable explanations. Served via a FastAPI REST API with async batching, monitored via Prometheus/Grafana with automated drift-triggered retraining. Frontend dashboard built in React with live inference, ROC comparison, and a Three.js 3D event display. Fully containerized with Docker Compose, CI/CD via GitHub Actions."*

---

*End of Implementation Plan*

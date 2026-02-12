# AI-Nexus — Design

## Overview
This document describes the architecture, components, data flow, and design decisions for AI-Nexus. It maps requirements to components in the repository and provides implementation-level guidance for maintainers and implementers.

## High-Level Architecture
- Data Sources -> Ingestion -> Preprocessing -> Feature Store -> Training Pipeline -> Model Registry -> Model Serving -> API -> Frontend
- Feedback loop: Frontend/API -> Feedback ingestion -> Label store -> Retraining pipeline

Components
- Data Layer
  - Raw storage: `data/raw/`, `data/external/`
  - Processed artifacts: `data/processed/`
- Preprocessing
  - Modules in `preprocessing/` (validation, text cleaning, feature extraction)
- Models
  - Implementations in `src/models/` (scam_classifier, risk_scoring_model, anomaly_detection)
- Decision Engine
  - `src/decision_engine/alert_decision.py` - combines model outputs and business rules to generate alerts
- API
  - `api/app.py`, `api/routes.py`, `api/schemas.py` expose endpoints for predict, retrain, feedback
- Frontend
  - `frontend/src/` React app for dashboard and feedback capture
- Training & Retraining
  - `src/training/train_model.py`, `retrain_pipeline.py`
- Monitoring & Logging
  - Logs in `logs/`, model metrics in `monitoring/model_metrics.json`

## Component Design Details

### Preprocessing & Feature Extraction
- Text cleaning steps:
  - Lowercase, normalize whitespace, remove HTML and control chars (see `preprocessing/text_cleaning.py`).
  - Tokenization via `nlp_engine/tokenizer.py`.
  - Stopword removal and optional stemming/lemmatization.
- Feature types:
  - Text embeddings (TF-IDF or pretrained embeddings) for `text` fields.
  - Categorical encodings for `metadata` fields.
  - Derived features: message length, suspicious tokens, frequency-based signals.
- Persist feature vectors alongside cleaned records in `data/processed/`.

### Model Design
- `scam_classifier`
  - Input: cleaned text features + metadata features.
  - Model choices: logistic regression or light tree-based model for baseline; optionally transformer-based model for higher accuracy.
  - Output: probability score and binary label.
- `risk_scoring_model`
  - Produces a continuous risk score in [0,1]. May be regression or ordinal classifier.
- `anomaly_detection`
  - Unsupervised model to flag unusual patterns (isolation forest, clustering-based).

### Model Versioning & Registry
- Save model artifacts with metadata (timestamp, dataset snapshot, hyperparams, metrics).
- Naming convention: `modelname_version_timestamp`.
- Registry can be a filesystem directory or external store (S3, MLflow).

### Training Pipeline
- Deterministic pipeline in `src/training/`:
  - Step 1: Read processed data.
  - Step 2: Split train/val/test and persist splits.
  - Step 3: Train model, perform hyperparameter tuning (optional).
  - Step 4: Evaluate and record metrics to `monitoring/model_metrics.json`.
  - Step 5: Register model artifact.
- Retraining triggers:
  - Manual via `POST /retrain`.
  - Scheduled cron job based on new data availability or drift detection.

### API Design
- `POST /predict`
  - Accepts batch payloads; the API should support streaming / chunked inference for large batches.
  - Input validation using `api/schemas.py`.
  - Response includes: `label`, `probability`, `risk_score`, `model_version`, `explanation` (optional e.g., top features).
- `POST /retrain`
  - Accepts config: target models, data range, hyperparam overrides.
  - Restricted to authenticated admin.
- `POST /feedback`
  - Stores user-provided labels and metadata to feedback store (`feedback/user_feedback_handler.py`).
- `GET /metrics` and `GET /health`
  - Provide system and model health data.

### Decision Engine
- Combines outputs using configurable rules (thresholds, business logic) in `alert_decision.py`.
- Rule config lives in `src/utils/config_loader.py` or a YAML file for operators to tune.

### Frontend
- Dashboard pages (`frontend/src/pages/Dashboard.jsx`) show:
  - Recent alerts with scores and explanation.
  - Aggregate metrics and model performance charts.
  - Feedback form per prediction.

### Deployment
- Local dev: `docker-compose.yml` builds API and frontend services, mounts local volumes for data and logs.
- Production: `kubernetes.yml` describes deployments for API pods, worker pods (training), and frontend. Use Horizontal Pod Autoscaler for API.
- Secrets: inject via environment variables or a secrets manager; never commit credentials.

### Observability
- Logs: structured logging in `src/utils/logger.py` with JSON output for ingestion by log collector.
- Metrics: push model metrics and service metrics to a metrics backend (Prometheus). Keep a JSON snapshot in `monitoring/model_metrics.json` for quick access.
- Alerts: notify on training failure, model drift, or elevated error rates.

### Security
- Authentication: API keys for service-to-service; user authentication for dashboard.
- Authorization: role-based access for retraining and admin operations.
- Data protection: encrypt sensitive data at rest; use TLS for transport.

### Scaling & Performance Considerations
- Keep inference code stateless to allow horizontal scaling.
- Use batching and asynchronous workers for heavy workloads.
- Cache model artifacts in memory per pod to avoid repeated loads.
- Consider GPU nodes for heavy model inference if using deep learning models.

### Testing Strategy
- Unit tests for preprocessing, model utilities, and decision logic.
- Integration tests for API endpoints using a lightweight test server.
- End-to-end tests in staging that exercise ingestion -> training -> serving -> feedback.

### Operational Runbooks (brief)
- Deploying a model: run training pipeline, verify metrics, register artifact, update serving config, rollout in canary mode.
- Rolling back: revert to prior model version in model registry and restart pods.
- Handling drift: if metrics fall below threshold, raise alert and flag for retraining.

## File Mappings (implementation pointers)
- API: [api/app.py](api/app.py), [api/routes.py](api/routes.py), [api/schemas.py](api/schemas.py)
- Models: [src/models/scam_classifier.py](src/models/scam_classifier.py), [src/models/risk_scoring_model.py](src/models/risk_scoring_model.py)
- Preprocessing: [preprocessing/text_cleaning.py](preprocessing/text_cleaning.py), [preprocessing/feature_extraction.py](preprocessing/feature_extraction.py)
- Decision engine: [src/decision_engine/alert_decision.py](src/decision_engine/alert_decision.py)
- Training: [src/training/train_model.py](src/training/train_model.py)

## Open Design Decisions / Trade-offs
- Model types: start with lightweight models for faster iteration, move to transformer-based models if necessary.
- Registry: simple filesystem-based registry for v1, migrate to MLflow or similar when scaling.
- Monitoring: local JSON snapshots vs. full Prometheus+Grafana. Start with snapshots, plan for metrics backend.

## Next Steps for Implementation
- Add CI job to run tests and build Docker images.
- Implement model registry helper and standardize model metadata format.
- Add authentication to API and dashboard.


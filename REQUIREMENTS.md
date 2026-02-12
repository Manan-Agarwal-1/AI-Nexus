# AI-Nexus — Requirements

## Project Overview
AI-Nexus is a system for detecting scams, scoring risk, and providing actionable alerts through an API and a web dashboard. It processes raw datasets, trains and serves machine learning models (scam classifier, risk scoring, anomaly detection), and supports a feedback loop for continuous improvement.

## Goals & Scope
- Ingest raw scam-related datasets and third-party sources.
- Clean, validate, and extract features for model training and inference.
- Train and serve multiple models via a REST API.
- Provide a frontend dashboard for viewing alerts, model scores, and metrics.
- Capture user feedback and support retraining pipelines.
- Deploy using containerization and orchestration (Docker, Kubernetes).
- Monitor model performance and system health.

## Functional Requirements
- Data Ingestion
  - Accept CSVs and common structured formats from `data/raw/` and `data/external/`.
  - Store cleaned output in `data/processed/`.

- Preprocessing & Feature Extraction
  - Normalize, clean, and tokenize text fields.
  - Extract numerical features and categorical encodings.
  - Validate input schema and drop/flag invalid records.

- Model Training & Retraining
  - Support offline training pipelines (`src/training/train_model.py`, `retrain_pipeline.py`).
  - Retrain on schedule or on-demand via an API endpoint.
  - Persist models (versioned) and training metadata.

- Model Serving & Decision Engine
  - Expose prediction endpoints for: scam classification, risk scoring, anomaly detection.
  - Provide a combined decision output used by `decision_engine/alert_decision.py`.

- API
  - Health check: `GET /health`.
  - Predict: `POST /predict` (accepts single or batch records, returns scores and labels).
  - Retrain trigger: `POST /retrain` (restricted access).
  - Feedback: `POST /feedback` (captures user corrections/labels).
  - Metrics: `GET /metrics` (model metrics, inference counts).

- Frontend Dashboard
  - Visualize alerts, recent predictions, risk distribution, and model metrics.
  - Allow users to submit feedback on predictions.

- Logging & Monitoring
  - Centralized logs for API requests, model inference, training runs.
  - Export model metrics to `monitoring/model_metrics.json` (or external monitoring).

- Testing & Validation
  - Unit tests for model code and preprocessing (`tests/unit/test_models.py`).
  - Integration tests for API endpoints (`tests/integration/test_api.py`).

## Non-Functional Requirements
- Performance
  - Inference latency: < 200ms per request for single-record predictions under normal load.
  - Batch throughput: scalable based on cluster size.

- Scalability
  - Stateless API pods to allow horizontal scaling.
  - Storage for datasets and model artifacts that can scale (S3 or persistent volumes).

- Availability
  - Target 99.9% uptime for API endpoints.
  - Graceful degradation: if ML service unavailable, return fallback rule-based decision.

- Security
  - All API endpoints require authentication (API keys or OAuth in production).
  - Secure storage of secrets; do not commit credentials to repo.
  - Rate-limiting on public endpoints.

- Maintainability
  - Clear modular structure: `preprocessing/`, `models/`, `training/`, `api/`, `frontend/`.
  - Config-driven behavior via `envirnment.env` and `src/utils/config_loader.py`.

- Compliance & Privacy
  - Data retention policy for PII; anonymize where possible.
  - Support data deletion requests.

## Data Requirements
- Input formats: CSV, JSON lines with fields: `id`, `text`, `metadata`.
- Minimum schema: timestamp, source, content, uid (if available).
- Storage paths:
  - Raw: `data/raw/`
  - External: `data/external/`
  - Processed: `data/processed/`

## Model Requirements
- Models to include:
  - `scam_classifier` (binary classifier)
  - `risk_scoring_model` (regression or ordinal classifier)
  - `anomaly_detection` (unsupervised)
- Metrics: precision, recall, F1, ROC-AUC (where applicable); calibration metrics for scores.
- Versioning: maintain model version metadata (training data snapshot, hyperparams).

## API Contract (examples)
- `POST /predict`
  - Request: {"records": [{"id": "...", "text": "...", "metadata": {...}}]}
  - Response: {"predictions": [{"id": "...", "label": "scam|not_scam", "score": 0.87, "risk": 0.62}]}

- `POST /feedback`
  - Request: {"id": "...", "correct_label": "scam|not_scam", "comments": "..."}
  - Response: {"status": "accepted"}

## Deployment Requirements
- Use `Dockerfile` and `docker-compose.yml` for local dev and `kubernetes.yml` for production.
- CI pipeline should run linting, unit tests, and build images.

## Monitoring & Alerting
- Track model performance drift, inference error rate, and throughput.
- Alerts for model degradation and failed training runs.

## Testing Requirements
- Code coverage threshold (e.g., 70%+) for core modules.
- Integration tests for API endpoints run in CI.

## Acceptance Criteria
- End-to-end flow from raw data ingestion to API predictions functions in dev environment.
- Dashboard displays live predictions and metrics.
- Retrain pipeline can be triggered and persists a new model version.
- Unit and integration tests pass in CI.

## References
- Source layout: see `api/`, `src/`, `frontend/` folders for implementation.
- Dependency manifests: `requirements.txt`, `frontend/package.json`.

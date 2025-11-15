# Project Plan (Summary of `ML Ops Project Proposal.pdf`)

## Team & Communication

- **Team name:** Drift Detectives  
- **Members:** Pedram Yazdinia, Ben Hartwick, Ahanaf Hassan Rodoshi, Tausif Ahmed  
- **Cadence:** Weekly in-person sync (Mon 6 pm ET) + ad-hoc Discord calls; 24 h async SLA.  
- **Tracking:** Trello board with Backlog / In Progress / Review / Done columns.  
- **Decision making:** Document options → 10 min discussion → vote; pipeline lead breaks ties.

## Goals & Success Criteria

1. Ship a reliable, monitored credit-risk pipeline with reproducible end-to-end runs.
2. Deploy a containerised FastAPI endpoint (ECS/GHCR) with approval-gated CI/CD and automatic rollbacks.
3. Enforce data contracts (blocking >5% schema drift) and maintain ≥95% nightly pipeline success.
4. Maintain an MLflow registry with at least two promoted versions plus rollback drills.
5. Provide observability dashboards for drift + performance (Pandera/Evidently/CloudWatch).
6. Ensure every teammate owns a major workstream and contributes 3–4 substantive PRs.

## Technical Scope

- **Data platform:** DVC-tracked Kaggle snapshots stored under `data/raw/` plus optional S3/Azure remotes; DuckDB performs feature engineering locally.
- **ML stack:** sklearn + XGBoost pipeline with SMOTE/downsampling controlled via `configs/baseline.yaml`; Pandera enforces data contracts at every hop.
- **Experimentation:** MLflow (tracking + registry) with promotion helpers, SHAP/permutation importance reserved for future interpretability work.
- **Feature store:** DuckDB SQL + Parquet artefacts served by the `creditrisk.features` module.
- **Deployment:** Docker + GitHub Actions produce GHCR images for FastAPI + batch; CD optionally redeploys an ECS service and smoke-tests `/predict`.
- **Monitoring:** Evidently drift reports, production drift monitor with CloudWatch publishing, structured logging, and freshness tracking via `creditrisk.utils.data_freshness`.

## Risk & Quality Management

- Schema + missingness checks before every stage using Pandera + ValidationRunner; policy-based retrain triggers when production drift exceeds the configured threshold.
- Failure protocol: 30 min call within 24 h for disagreements, task redistribution for missed deadlines.
- Quality bar: ≥70% unit/integration coverage on core libs, demo-ready docs (README, ARCHITECTURE, OPS).

## Timeline Highlights

- **Weeks 1–3:** Baseline data contracts, repo scaffolding (this deliverable), CI wiring.
- **Weeks 4–6:** Feature pipelines, experiment tracking, registry integration.
- **Weeks 7–8:** Deployment, monitoring, canary/shadow testing.
- **Week 9:** Hardening, resilience testing, final demo.

Keep the full PDF (`docs/ML_Ops_Project_Proposal.pdf`) for the authoritative source;
this summary exists so the essentials stay version-controlled with the code.

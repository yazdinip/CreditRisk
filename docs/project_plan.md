# Project Plan (Summary of `ML Ops Project Proposal.pdf`)

## Team & Communication

- **Team name:** Drift Detectives  
- **Members:** Pedram Yazdinia, Ben Hartwick, Ahanaf Hassan Rodoshi, Tausif Ahmed  
- **Cadence:** Weekly in-person sync (Mon 6 pm ET) + ad-hoc Discord calls; 24 h async SLA.  
- **Tracking:** Trello board with Backlog / In Progress / Review / Done columns.  
- **Decision making:** Document options → 10 min discussion → vote; pipeline lead breaks ties.

## Goals & Success Criteria

1. Ship a reliable, monitored credit-risk pipeline with reproducible end-to-end runs.
2. Deploy a SageMaker endpoint with approval-gated CI/CD.
3. Enforce data contracts (blocking >5% schema drift) and maintain ≥95% nightly pipeline success.
4. Maintain an MLflow-style registry with at least two promoted versions plus rollback drills.
5. Provide observability dashboards for drift + performance (Great Expectations, Evidently, CloudWatch).
6. Ensure every teammate owns a major workstream and contributes 3–4 substantive PRs.

## Technical Scope

- **Data Platform:** S3 bronze/silver/gold buckets, AWS Glue/Athena (or DuckDB locally), Step Functions.
- **ML Stack:** SageMaker Processing/Training/Pipelines, XGBoost/LightGBM, optional SMOTE variants.
- **Experimentation:** MLflow (tracking + registry), SHAP for interpretability, permutation importance.
- **Feature Store:** SageMaker Feature Store (offline/online) or a light-weight OSS equivalent.
- **Deployment:** Docker + GitHub Actions for CI/CD, IaC via CDK/Terraform, signed images.
- **Monitoring:** Great Expectations, Evidently, CloudWatch canaries, drift/covariate shift simulations.

## Risk & Quality Management

- Schema + missingness checks before every stage; policy-based retrain triggers on drift scenarios.
- Failure protocol: 30 min call within 24 h for disagreements, task redistribution for missed deadlines.
- Quality bar: ≥70% unit/integration coverage on core libs, demo-ready docs (README, ARCHITECTURE, OPS).

## Timeline Highlights

- **Weeks 1–3:** Baseline data contracts, repo scaffolding (this deliverable), CI wiring.
- **Weeks 4–6:** Feature pipelines, experiment tracking, registry integration.
- **Weeks 7–8:** Deployment, monitoring, canary/shadow testing.
- **Week 9:** Hardening, resilience testing, final demo.

Keep the full PDF (`docs/ML_Ops_Project_Proposal.pdf`) for the authoritative source;
this summary exists so the essentials stay version-controlled with the code.

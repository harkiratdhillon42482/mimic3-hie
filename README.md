> [!WARNING]
> **Proof of Concept — Not for Clinical Use**
> This project is in active development and is intended solely for research and educational purposes.
> It must not be used for clinical decision-making, patient care, diagnosis, or treatment.
> MIMIC-III data is de-identified retrospective data and does not represent real-time clinical information.
> Model outputs have not been validated for clinical deployment.
# MUMPS-Native Clinical AI — Liver Transplant Candidate Identification

A Health Information Exchange (HIE) built on YottaDB MUMPS as the canonical patient store and PostgreSQL as the analytics layer, with a prospective ML pipeline for liver transplant candidate identification using MIMIC-III.

## Live Pages

- **[Results Dashboard](https://harkiratdhillon42482.github.io/mimic3-hie/)** — Charts, model comparison, architecture analysis, real patient trajectories
- **[Full Paper](https://harkiratdhillon42482.github.io/mimic3-hie/paper.html)** — Abstract, methods, results, discussion, references

## Key Results

| Model | AUC | 95% CI | Train Time |
|-------|-----|--------|------------|
| XGBoost | **0.853** | [0.818, 0.884] | 0.8s |
| LightGBM | 0.847 | [0.812, 0.880] | 0.3s |
| Gradient Boosting | 0.829 | [0.792, 0.863] | 3.1s |
| Random Forest | 0.821 | [0.780, 0.860] | 0.7s |
| Logistic Regression | 0.762 | [0.721, 0.801] | <0.1s |
| Ghandian et al. 2022 (reference) | 0.871 | [0.859, 0.882] | retrospective |

## Architecture

| Metric | PostgreSQL | MUMPS ^PHD |
|--------|-----------|------------|
| Feature extraction | 79s | **23.5s (3.4x faster)** |
| Patients | 3,382 | 3,350 |
| XGBoost AUC | 0.853 | 0.820 |
| AUC difference | statistically equivalent (CIs overlap) | |

## ML Benchmark — Liver Transplant Prediction (Prospective)

- **Task:** At admission N, predict significant deterioration at admission N+1
- **Label:** MELD increases >=5 pts OR MELD>=30 OR new decompensation OR death
- **Split:** Patient-level 70/30, no patient in both train and test, no leakage
- **Cohort:** 3,382 liver patients, 5,505 admissions, MIMIC-III 2001-2012
- **Pairs:** 2,123 prospective prediction instances (22.3% positive)
- **Top feature:** meld_consecutive_increases (importance 0.077)
- **AUC at MELD 30-40:** 0.901 (highest in urgent zone)

## Previous Benchmark — 30-Day Readmission

| Source | n | AUC-ROC | F1 | Feature Time |
|--------|---|---------|-----|--------------|
| Postgres | 10,991 | 0.6044 | 0.404 | 42.3s |
| MUMPS | 10,822 | 0.6067 | 0.395 | 77.6s |

## Stack

- YottaDB r2.06 (MUMPS database, Docker)
- PostgreSQL 16 (analytics layer)
- Python 3.12, scikit-learn, XGBoost, LightGBM
- Chart.js (results dashboard)

## Data

Requires MIMIC-III v1.4 via PhysioNet credentialing: https://physionet.org/content/mimiciii/1.4/
MIMIC-III data is NOT included in this repository.

## Quick Start

```bash
docker build -t hie-mumps -f docker/Dockerfile .
docker run -d --name mumps-bench --entrypoint "/bin/bash" \
  -v "./data/ydb:/data" -v ".:/project" \
  hie-mumps -c "while true; do sleep 30; done"
docker exec -it mumps-bench bash
cd /data/r2.06_x86_64/g
python3 /project/load/etl.py --steps patients admissions
python3 /project/sync/sync_engine.py --steps person visits
python3 /project/ml/readmission_pg.py
python3 /project/ml/readmission_ydb.py
python3 /project/ml/compare.py
```

## Project Phases

- Phase 1 (complete): MIMIC-III + YottaDB + PostgreSQL + XGBoost liver prediction AUC 0.853
- Phase 2 (planned): LSTM + Attention on Google Colab (expected AUC 0.87-0.92)
- Phase 3 (planned): Transformer model 50-60M params (expected AUC 0.90-0.95)
- Phase 4 (planned): MIMIC-IV temporal validation
- Phase 5 (planned): Live HL7 v2.9 ADT feeds + real-time deployment

## License

MIT. MIMIC-III data subject to PhysioNet credentialing.



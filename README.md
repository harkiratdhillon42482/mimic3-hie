# mimic3-hie

A Health Information Exchange (HIE) built on YottaDB MUMPS as the canonical patient store and PostgreSQL as the analytics layer, designed for LLM training pipeline development on MIMIC-III clinical data.

## Architecture
## ML Benchmark Results

30-day hospital readmission (Logistic Regression, 26 features):

| Source   | n      | AUC-ROC | F1    | Feature Time |
|----------|--------|---------|-------|--------------|
| Postgres | 10,991 | 0.6044  | 0.404 | 42.3s        |
| MUMPS    | 10,822 | 0.6067  | 0.395 | 77.6s        |

Model quality identical. Postgres wins cohort aggregation. MUMPS wins point lookups and LLM context assembly.

## Stack

- YottaDB r2.06 (MUMPS database, Docker)
- PostgreSQL 16 (analytics layer)
- Python 3.12, scikit-learn 1.8

## Data

Requires MIMIC-III v1.4 via PhysioNet credentialing:
https://physionet.org/content/mimiciii/1.4/

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

- Phase 1 (complete): MIMIC-III + YottaDB + PostgreSQL
- Phase 2 (planned): MIMIC-IV + identity resolution
- Phase 3 (planned): Live HL7 v2.9 ADT feeds

## License

MIT. MIMIC-III data subject to PhysioNet credentialing.

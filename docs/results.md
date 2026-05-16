# Results

## 30-Day Readmission Benchmark

| Source   | n      | AUC-ROC | F1    | Feature Time |
|----------|--------|---------|-------|--------------|
| Postgres | 10,991 | 0.6044  | 0.404 | 42.3s        |
| MUMPS    | 10,822 | 0.6067  | 0.395 | 77.6s        |

## Top Features

| Feature | Direction | Clinical Meaning |
|---------|-----------|-----------------|
| is_emergency | down | Emergency resolves acutely |
| is_elective | down | Planned care, controlled |
| n_diagnoses | up | Higher disease burden |
| pct_abnormal_labs | up | Clinical instability |

## Architectural Finding

Postgres wins cohort aggregation (SQL indexes).
MUMPS wins point lookups and LLM context assembly (B-tree, co-located data).

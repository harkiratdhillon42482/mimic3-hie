"""
readmission_pg.py
30-Day Hospital Readmission Prediction — Postgres path
Reads features from hie.* tables in MIMICold
Trains logistic regression, stores results in hie.model_result

Run inside container:
  cd /data/r2.06_x86_64/g
  python3 /project/readmission_pg.py
"""

import os
import sys
import time
import json
import uuid
import psycopg2
import numpy as np
import pandas as pd
from datetime import datetime

PG_CONFIG = {
    "host":     "host.docker.internal",
    "port":     5432,
    "dbname":   "MIMICold",
    "user":     "postgres",
    "password": "Panjwar4633",
}

def get_pg():
    return psycopg2.connect(**PG_CONFIG)

# =============================================================================
# STEP 1: EXTRACT FEATURES FROM POSTGRES
# =============================================================================

FEATURE_QUERY = """
WITH

-- Index admissions: exclude in-hospital deaths
-- These are the admissions we're trying to predict readmission for
index_admissions AS (
    SELECT
        vo.visit_occurrence_id,
        vo.person_id,
        vo.visit_start_datetime    AS admit_dt,
        vo.visit_end_datetime      AS discharge_dt,
        vo.los_hours,
        vo.admission_type          AS visit_type,
        vo.insurance,
        vo.has_icu_stay,
        vo.hospital_expire_flag,
        p.gender_concept_code      AS sex,
        p.year_of_birth
    FROM hie.visit_occurrence vo
    JOIN hie.person p ON vo.person_id = p.person_id
    WHERE vo.hospital_expire_flag = 0        -- alive at discharge
      AND vo.visit_end_datetime IS NOT NULL  -- has discharge time
      AND vo.visit_start_datetime IS NOT NULL
),

-- Next admission per patient (to compute readmission label)
next_admission AS (
    SELECT
        person_id,
        visit_start_datetime,
        LAG(visit_end_datetime) OVER (
            PARTITION BY person_id
            ORDER BY visit_start_datetime
        ) AS prev_discharge_dt,
        LAG(visit_occurrence_id) OVER (
            PARTITION BY person_id
            ORDER BY visit_start_datetime
        ) AS prev_visit_id
    FROM hie.visit_occurrence
    WHERE hospital_expire_flag = 0
),

-- Readmission labels
labels AS (
    SELECT
        prev_visit_id                                            AS visit_occurrence_id,
        CASE
            WHEN EXTRACT(EPOCH FROM (visit_start_datetime - prev_discharge_dt))/86400 <= 30
            THEN 1 ELSE 0
        END                                                      AS readmitted_30d
    FROM next_admission
    WHERE prev_visit_id IS NOT NULL
),

-- Diagnosis features
dx_features AS (
    SELECT
        visit_occurrence_id,
        COUNT(*)                                                 AS n_diagnoses,
        -- ICD-9 chapter groups (first digit)
        SUM(CASE WHEN condition_concept_code ~ '^[0-9]'
            THEN 1 ELSE 0 END)                                   AS n_icd9_numeric,
        -- Common condition groups
        SUM(CASE WHEN condition_concept_code LIKE '428%'
            THEN 1 ELSE 0 END)                                   AS has_chf,
        SUM(CASE WHEN condition_concept_code LIKE '250%'
            THEN 1 ELSE 0 END)                                   AS has_diabetes,
        SUM(CASE WHEN condition_concept_code LIKE '496%'
            OR condition_concept_code LIKE '491%'
            OR condition_concept_code LIKE '492%'
            THEN 1 ELSE 0 END)                                   AS has_copd,
        SUM(CASE WHEN condition_concept_code LIKE '585%'
            THEN 1 ELSE 0 END)                                   AS has_ckd,
        SUM(CASE WHEN condition_concept_code LIKE '410%'
            OR condition_concept_code LIKE '411%'
            THEN 1 ELSE 0 END)                                   AS has_ami,
        -- Elixhauser comorbidity count (simplified)
        COUNT(DISTINCT LEFT(condition_concept_code, 3))          AS n_unique_dx_groups
    FROM hie.condition_occurrence
    GROUP BY visit_occurrence_id
),

-- Medication features
med_features AS (
    SELECT
        visit_occurrence_id,
        COUNT(*)                                                 AS n_medications,
        COUNT(DISTINCT drug_name)                                AS n_unique_drugs
    FROM hie.drug_exposure
    GROUP BY visit_occurrence_id
),

-- Lab features
lab_features AS (
    SELECT
        visit_occurrence_id,
        COUNT(*)                                                 AS n_labs,
        SUM(CASE WHEN abnormal_flag IS NOT NULL
            AND abnormal_flag != ''
            THEN 1 ELSE 0 END)                                   AS n_abnormal_labs,
        ROUND(100.0 * SUM(CASE WHEN abnormal_flag IS NOT NULL
            AND abnormal_flag != ''
            THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2)        AS pct_abnormal_labs
    FROM hie.measurement
    GROUP BY visit_occurrence_id
),

-- Note features
note_features AS (
    SELECT
        visit_occurrence_id,
        COUNT(*)                                                 AS n_notes,
        SUM(CASE WHEN note_category = 'Discharge summary'
            THEN 1 ELSE 0 END)                                   AS n_discharge_summaries,
        AVG(char_count)                                          AS avg_note_length
    FROM hie.note
    GROUP BY visit_occurrence_id
),

-- ICU features
icu_features AS (
    SELECT
        visit_occurrence_id,
        COUNT(*)                                                 AS n_icu_stays,
        SUM(los_hours)                                           AS total_icu_los_hours
    FROM hie.visit_detail
    GROUP BY visit_occurrence_id
)

-- Final feature matrix
SELECT
    ia.visit_occurrence_id,
    ia.person_id,

    -- Target
    COALESCE(l.readmitted_30d, 0)                               AS readmitted_30d,

    -- Demographics
    CASE WHEN ia.sex = 'M' THEN 1 ELSE 0 END                    AS is_male,
    COALESCE(
        EXTRACT(YEAR FROM ia.admit_dt) - ia.year_of_birth, 65)  AS age_at_admit,

    -- Admission features
    COALESCE(ia.los_hours, 0)                                    AS los_hours,
    CASE WHEN ia.visit_type = 'EMERGENCY' THEN 1 ELSE 0 END     AS is_emergency,
    CASE WHEN ia.visit_type = 'ELECTIVE'  THEN 1 ELSE 0 END     AS is_elective,
    CASE WHEN ia.visit_type = 'URGENT'    THEN 1 ELSE 0 END     AS is_urgent,
    CASE WHEN ia.insurance = 'Medicare'   THEN 1 ELSE 0 END     AS is_medicare,
    CASE WHEN ia.insurance = 'Medicaid'   THEN 1 ELSE 0 END     AS is_medicaid,
    CASE WHEN ia.has_icu_stay             THEN 1 ELSE 0 END     AS had_icu_stay,

    -- Diagnosis features
    COALESCE(dx.n_diagnoses, 0)                                  AS n_diagnoses,
    COALESCE(dx.n_unique_dx_groups, 0)                           AS n_unique_dx_groups,
    COALESCE(dx.has_chf, 0)                                      AS has_chf,
    COALESCE(dx.has_diabetes, 0)                                 AS has_diabetes,
    COALESCE(dx.has_copd, 0)                                     AS has_copd,
    COALESCE(dx.has_ckd, 0)                                      AS has_ckd,
    COALESCE(dx.has_ami, 0)                                      AS has_ami,

    -- Medication features
    COALESCE(med.n_medications, 0)                               AS n_medications,
    COALESCE(med.n_unique_drugs, 0)                              AS n_unique_drugs,

    -- Lab features
    COALESCE(lab.n_labs, 0)                                      AS n_labs,
    COALESCE(lab.n_abnormal_labs, 0)                             AS n_abnormal_labs,
    COALESCE(lab.pct_abnormal_labs, 0)                           AS pct_abnormal_labs,

    -- Note features
    COALESCE(nt.n_notes, 0)                                      AS n_notes,
    COALESCE(nt.n_discharge_summaries, 0)                        AS n_discharge_summaries,
    COALESCE(nt.avg_note_length, 0)                              AS avg_note_length,

    -- ICU features
    COALESCE(icu.n_icu_stays, 0)                                 AS n_icu_stays,
    COALESCE(icu.total_icu_los_hours, 0)                         AS total_icu_los_hours

FROM index_admissions ia
LEFT JOIN labels       l   ON ia.visit_occurrence_id = l.visit_occurrence_id
LEFT JOIN dx_features  dx  ON ia.visit_occurrence_id = dx.visit_occurrence_id
LEFT JOIN med_features med  ON ia.visit_occurrence_id = med.visit_occurrence_id
LEFT JOIN lab_features lab  ON ia.visit_occurrence_id = lab.visit_occurrence_id
LEFT JOIN note_features nt  ON ia.visit_occurrence_id = nt.visit_occurrence_id
LEFT JOIN icu_features icu  ON ia.visit_occurrence_id = icu.visit_occurrence_id

ORDER BY ia.admit_dt;
"""

FEATURE_COLS = [
    "is_male", "age_at_admit", "los_hours",
    "is_emergency", "is_elective", "is_urgent",
    "is_medicare", "is_medicaid", "had_icu_stay",
    "n_diagnoses", "n_unique_dx_groups",
    "has_chf", "has_diabetes", "has_copd", "has_ckd", "has_ami",
    "n_medications", "n_unique_drugs",
    "n_labs", "n_abnormal_labs", "pct_abnormal_labs",
    "n_notes", "n_discharge_summaries", "avg_note_length",
    "n_icu_stays", "total_icu_los_hours",
]

# =============================================================================
# STEP 2: TRAIN AND EVALUATE
# =============================================================================

def train_and_evaluate(df, run_id, source):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        roc_auc_score, accuracy_score,
        precision_score, recall_score, f1_score
    )

    print(f"\n  Dataset: {len(df):,} admissions")
    print(f"  Positive rate: {df['readmitted_30d'].mean():.1%}")

    X = df[FEATURE_COLS].fillna(0).values
    y = df["readmitted_30d"].values

    # Train/test split — time-based (use chronological order)
    # Already ordered by admit_dt from the query
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Train
    t_train = time.time()
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)
    train_time = time.time() - t_train

    # Predict
    t_pred = time.time()
    y_prob = model.predict_proba(X_test_s)[:, 1]
    y_pred = model.predict(X_test_s)
    pred_time = time.time() - t_pred

    # Metrics
    auc  = roc_auc_score(y_test, y_prob)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    # Top features by coefficient magnitude
    coefs = sorted(
        zip(FEATURE_COLS, model.coef_[0]),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:10]
    top_features = {k: round(float(v), 4) for k, v in coefs}

    print(f"\n  ── Results ──────────────────────────")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Train time: {train_time:.3f}s")
    print(f"  Pred time:  {pred_time:.3f}s")
    print(f"\n  Top features:")
    for feat, coef in list(top_features.items())[:5]:
        direction = "↑" if coef > 0 else "↓"
        print(f"    {direction} {feat:<30} {coef:+.4f}")

    return {
        "run_id":           run_id,
        "model_type":       "LogisticRegression",
        "target":           "30day_readmission",
        "data_source":      source,
        "n_total":          len(df),
        "n_train":          len(X_train),
        "n_test":           len(X_test),
        "n_features":       len(FEATURE_COLS),
        "positive_rate":    float(y.mean()),
        "auc_roc":          float(auc),
        "accuracy":         float(acc),
        "precision_score":  float(prec),
        "recall_score":     float(rec),
        "f1_score":         float(f1),
        "train_time_s":     float(train_time),
        "predict_time_s":   float(pred_time),
        "top_features_json": top_features,
        "config_json":      {"max_iter": 1000, "random_state": 42},
    }

# =============================================================================
# STEP 3: STORE RESULTS
# =============================================================================

def store_results(pg, result):
    sql = """
        INSERT INTO hie.model_result (
            run_id, model_type, target, data_source,
            n_total, n_train, n_test, n_features, positive_rate,
            auc_roc, accuracy, precision_score, recall_score, f1_score,
            train_time_s, predict_time_s, total_time_s,
            top_features_json, config_json
        ) VALUES (
            %(run_id)s, %(model_type)s, %(target)s, %(data_source)s,
            %(n_total)s, %(n_train)s, %(n_test)s, %(n_features)s, %(positive_rate)s,
            %(auc_roc)s, %(accuracy)s, %(precision_score)s, %(recall_score)s, %(f1_score)s,
            %(train_time_s)s, %(predict_time_s)s, %(total_time_s)s,
            %(top_features_json)s, %(config_json)s
        )
        ON CONFLICT (run_id) DO UPDATE SET
            auc_roc          = EXCLUDED.auc_roc,
            accuracy         = EXCLUDED.accuracy,
            total_time_s     = EXCLUDED.total_time_s
    """
    result["top_features_json"] = json.dumps(result["top_features_json"])
    result["config_json"]       = json.dumps(result["config_json"])
    with pg.cursor() as cur:
        cur.execute(sql, result)
    pg.commit()
    print(f"\n  ✓ Results stored in hie.model_result (run_id: {result['run_id']})")

# =============================================================================
# MAIN
# =============================================================================

def main():
    run_id = f"PG-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    print("=" * 58)
    print("  30-Day Readmission — Postgres Path")
    print(f"  Run ID: {run_id}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 58)

    pg = get_pg()

    # Extract features
    print("\n[1] Extracting features from hie.* tables...")
    t_feat = time.time()
    df = pd.read_sql(FEATURE_QUERY, pg)
    feat_time = time.time() - t_feat
    print(f"  ✓ {len(df):,} rows extracted in {feat_time:.2f}s")

    # Train and evaluate
    print("\n[2] Training logistic regression...")
    result = train_and_evaluate(df, run_id, "POSTGRES")
    result["feature_time_s"] = float(feat_time)
    result["total_time_s"]   = float(feat_time + result["train_time_s"] + result["predict_time_s"])

    # Store results
    print("\n[3] Storing results...")
    store_results(pg, result)

    pg.close()

    print("\n" + "=" * 58)
    print(f"  Done. Total time: {result['total_time_s']:.2f}s")
    print(f"  AUC-ROC: {result['auc_roc']:.4f}")
    print("=" * 58)

if __name__ == "__main__":
    main()

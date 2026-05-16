"""
readmission_ydb.py
30-Day Hospital Readmission Prediction â€” MUMPS path
Reads features directly from YottaDB ^PHD globals
Same model as readmission_pg.py â€” compares speed

Run inside container:
  cd /data/r2.06_x86_64/g
  python3 /project/readmission_ydb.py
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime

# â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
YDB_ENV = {
    "ydb_dist":     "/opt/yottadb/current",
    "ydb_gbldir":   "/data/r2.06_x86_64/g/yottadb.gld",
    "ydb_routines": "/opt/yottadb/current/libyottadbutil.so",
}

PG_CONFIG = {
    "host":     "host.docker.internal",
    "port":     5432,
    "dbname":   "MIMICold",
    "user":     "postgres",
    "password": "Panjwar4633",
}

# Same features as Postgres path
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

# â”€â”€ YDB Setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def setup_ydb():
    for k, v in YDB_ENV.items():
        os.environ[k] = v
    os.chdir("/data/r2.06_x86_64/g")
    import yottadb as ydb
    return ydb

def yget(ydb, *args):
    try:
        val = ydb.get(*args)
        if val is None: return ""
        return val.decode() if isinstance(val, bytes) else str(val)
    except Exception:
        return ""

def ynext(ydb, gbl, subs):
    try:
        val = ydb.subscript_next(gbl, subs)
        if val is None: return None
        return val.decode() if isinstance(val, bytes) else str(val)
    except Exception:
        return None

def walk(ydb, gbl, path):
    sub = ""
    while True:
        sub = ynext(ydb, gbl, path + [sub])
        if not sub: break
        yield sub

def parse_dt(s):
    """Parse YDB compact datetime to Python datetime."""
    if not s or len(s) < 8:
        return None
    try:
        from datetime import datetime as dt
        digits = "".join(c for c in str(s) if c.isdigit())
        if len(digits) < 8: return None
        yr, mo, dy = int(digits[0:4]), int(digits[4:6]), int(digits[6:8])
        if yr < 1800 or yr > 2200 or mo < 1 or mo > 12 or dy < 1 or dy > 31:
            return None
        hr = int(digits[8:10]) if len(digits) >= 10 else 0
        mn = int(digits[10:12]) if len(digits) >= 12 else 0
        return dt(yr, mo, dy, hr, mn)
    except Exception:
        return None

# =============================================================================
# FEATURE EXTRACTION FROM ^PHD
# =============================================================================

# ICD-9 prefixes for common comorbidities
CHF_CODES      = ("428",)
DIABETES_CODES = ("250",)
COPD_CODES     = ("496","491","492")
CKD_CODES      = ("585",)
AMI_CODES      = ("410","411")

def extract_patient_features(ydb, pg):
    """
    Traverse ^PHD for all patients and extract features.
    Also queries Postgres once for readmission labels (30-day window).
    Returns a DataFrame with same columns as the Postgres path.
    """

    # Get readmission labels from Postgres â€” one query, then close
    print("  Loading readmission labels from Postgres...")
    pg_cur = pg.cursor()
    pg_cur.execute("""
        WITH next_adm AS (
            SELECT
                person_id,
                visit_start_datetime,
                LAG(visit_end_datetime) OVER (
                    PARTITION BY person_id ORDER BY visit_start_datetime
                ) AS prev_disch,
                LAG(visit_occurrence_id) OVER (
                    PARTITION BY person_id ORDER BY visit_start_datetime
                ) AS prev_vid
            FROM hie.visit_occurrence
            WHERE hospital_expire_flag = 0
        )
        SELECT
            prev_vid,
            CASE WHEN EXTRACT(EPOCH FROM
                (visit_start_datetime - prev_disch))/86400 <= 30
            THEN 1 ELSE 0 END AS readmitted
        FROM next_adm
        WHERE prev_vid IS NOT NULL
    """)
    labels = {row[0]: row[1] for row in pg_cur.fetchall()}
    pg_cur.close()
    print(f"  âœ“ {len(labels):,} labels loaded")

    # Also get year_of_birth from Postgres (MIMIC DOB shift makes YDB DOB unreliable for age)
    pg_cur2 = pg.cursor()
    pg_cur2.execute("SELECT person_id, year_of_birth FROM hie.person")
    yob_map = {row[0]: row[1] for row in pg_cur2.fetchall()}
    pg_cur2.close()

    records = []
    n_patients = 0
    n_encounters = 0

    print("  Traversing ^PHD globals...")
    t0 = time.time()

    for pid in walk(ydb, "^PHD", []):
        if not pid.startswith("HIE-"):
            continue
        n_patients += 1

        yob = yob_map.get(pid)

        for eid in walk(ydb, "^PHD", [pid, "VISIT"]):
            root = yget(ydb, "^PHD", [pid, "VISIT", eid, "0"])
            if not root:
                continue

            parts = (root + "^^^^^^^").split("^")
            admit_raw  = parts[0]
            disch_raw  = parts[1]
            adm_type   = parts[2]
            insurance  = parts[4]
            expire     = parts[5]
            los_s      = parts[7] if len(parts) > 7 else ""

            # Skip in-hospital deaths
            if expire == "1":
                continue

            # Skip if no discharge
            if not disch_raw:
                continue

            admit_dt = parse_dt(admit_raw)
            if not admit_dt:
                continue

            # Only process index admissions that have a label
            if eid not in labels:
                continue
            label = labels[eid]

            # LOS
            try:
                los_h = float(los_s) if los_s and los_s != "None" else 0.0
            except Exception:
                los_h = 0.0

            # Age
            age = (admit_dt.year - yob) if yob else 65

            # Admission type flags
            adm_upper = (adm_type or "").upper()
            is_emergency = 1 if "EMERGENCY" in adm_upper else 0
            is_elective  = 1 if "ELECTIVE"  in adm_upper else 0
            is_urgent    = 1 if "URGENT"    in adm_upper else 0

            # Insurance
            ins_upper = (insurance or "").upper()
            is_medicare = 1 if "MEDICARE" in ins_upper else 0
            is_medicaid = 1 if "MEDICAID" in ins_upper else 0

            # ICU
            had_icu = 1 if yget(ydb, "^PHD", [pid, "VISIT", eid, "HAS_ICU"]) == "1" else 0

            # Sex
            sex = yget(ydb, "^PHD", [pid, "PID", "SEX"])
            is_male = 1 if sex == "M" else 0

            # â”€â”€ Diagnoses â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            n_dx = 0
            dx_groups = set()
            has_chf = has_dm = has_copd = has_ckd = has_ami = 0

            for seq in walk(ydb, "^PHD", [pid, "VISIT", eid, "DX"]):
                val = yget(ydb, "^PHD", [pid, "VISIT", eid, "DX", seq])
                code = val.split("^")[0] if val else ""
                if code:
                    n_dx += 1
                    dx_groups.add(code[:3])
                    if any(code.startswith(p) for p in CHF_CODES):      has_chf  = 1
                    if any(code.startswith(p) for p in DIABETES_CODES): has_dm   = 1
                    if any(code.startswith(p) for p in COPD_CODES):     has_copd = 1
                    if any(code.startswith(p) for p in CKD_CODES):      has_ckd  = 1
                    if any(code.startswith(p) for p in AMI_CODES):      has_ami  = 1

            # â”€â”€ Medications â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            n_meds = 0
            unique_drugs = set()
            for idx in walk(ydb, "^PHD", [pid, "VISIT", eid, "MED"]):
                val = yget(ydb, "^PHD", [pid, "VISIT", eid, "MED", idx])
                drug = val.split("^")[0] if val else ""
                if drug:
                    n_meds += 1
                    unique_drugs.add(drug)

            # â”€â”€ Labs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            n_labs = 0
            n_abnormal = 0
            for idx in walk(ydb, "^PHD", [pid, "VISIT", eid, "LAB"]):
                val = yget(ydb, "^PHD", [pid, "VISIT", eid, "LAB", idx])
                parts_lab = (val + "^^^^").split("^")
                flag = parts_lab[4] if len(parts_lab) > 4 else ""
                n_labs += 1
                if flag and flag not in ("", "None"):
                    n_abnormal += 1

            pct_abnormal = round(100.0 * n_abnormal / n_labs, 2) if n_labs > 0 else 0.0

            # â”€â”€ Notes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            n_notes = 0
            n_discharge = 0
            total_note_len = 0
            for idx in walk(ydb, "^PHD", [pid, "VISIT", eid, "NOTE"]):
                meta = yget(ydb, "^PHD", [pid, "VISIT", eid, "NOTE", idx])
                cat  = meta.split("^")[0] if meta else ""
                txt  = yget(ydb, "^PHD", [pid, "VISIT", eid, "NOTETXT", idx])
                n_notes += 1
                total_note_len += len(txt) if txt else 0
                if cat == "Discharge summary":
                    n_discharge += 1

            avg_note_len = total_note_len / n_notes if n_notes > 0 else 0.0

            # â”€â”€ ICU stays â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            n_icu = 0
            total_icu_los = 0.0
            for icuid in walk(ydb, "^PHD", [pid, "VISIT", eid, "ICU"]):
                icu_root = yget(ydb, "^PHD", [pid, "VISIT", eid, "ICU", icuid, "0"])
                n_icu += 1
                try:
                    icu_los = float(icu_root.split("^")[3]) if icu_root else 0.0
                    total_icu_los += icu_los
                except Exception:
                    pass

            records.append({
                "visit_occurrence_id":  eid,
                "person_id":            pid,
                "admit_dt":             admit_dt,
                "readmitted_30d":       label,
                "is_male":              is_male,
                "age_at_admit":         max(0, min(120, age)),
                "los_hours":            los_h,
                "is_emergency":         is_emergency,
                "is_elective":          is_elective,
                "is_urgent":            is_urgent,
                "is_medicare":          is_medicare,
                "is_medicaid":          is_medicaid,
                "had_icu_stay":         had_icu,
                "n_diagnoses":          n_dx,
                "n_unique_dx_groups":   len(dx_groups),
                "has_chf":              has_chf,
                "has_diabetes":         has_dm,
                "has_copd":             has_copd,
                "has_ckd":              has_ckd,
                "has_ami":              has_ami,
                "n_medications":        n_meds,
                "n_unique_drugs":       len(unique_drugs),
                "n_labs":               n_labs,
                "n_abnormal_labs":      n_abnormal,
                "pct_abnormal_labs":    pct_abnormal,
                "n_notes":              n_notes,
                "n_discharge_summaries": n_discharge,
                "avg_note_length":      avg_note_len,
                "n_icu_stays":          n_icu,
                "total_icu_los_hours":  total_icu_los,
            })
            n_encounters += 1

        if n_patients % 1000 == 0:
            elapsed = time.time() - t0
            rate = n_patients / elapsed
            print(f"\r  Patients: {n_patients:,}  Encounters: {n_encounters:,}  "
                  f"Speed: {rate:.0f} pts/s", end="", flush=True)

    print(f"\n  âœ“ {n_patients:,} patients, {n_encounters:,} encounters traversed "
          f"in {time.time()-t0:.1f}s")

    df = pd.DataFrame(records)
    df = df.sort_values("admit_dt").reset_index(drop=True)
    return df

# =============================================================================
# TRAIN AND EVALUATE (same as Postgres path)
# =============================================================================

def train_and_evaluate(df, run_id, source):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        roc_auc_score, accuracy_score,
        precision_score, recall_score, f1_score
    )

    # Filter to only admissions that have a readmission label
    df_model = df.copy()
    print(f"\n  Dataset: {len(df_model):,} admissions")
    print(f"  Positive rate: {df_model['readmitted_30d'].mean():.1%}")

    X = df_model[FEATURE_COLS].fillna(0).values
    y = df_model["readmitted_30d"].values.astype(int)

    # Time-based split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    t_train = time.time()
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    model.fit(X_train_s, y_train)
    train_time = time.time() - t_train

    t_pred = time.time()
    y_prob = model.predict_proba(X_test_s)[:, 1]
    y_pred = model.predict(X_test_s)
    pred_time = time.time() - t_pred

    auc  = roc_auc_score(y_test, y_prob)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    coefs = sorted(
        zip(FEATURE_COLS, model.coef_[0]),
        key=lambda x: abs(x[1]), reverse=True
    )[:10]
    top_features = {k: round(float(v), 4) for k, v in coefs}

    print(f"\n  â”€â”€ Results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Train time: {train_time:.3f}s")
    print(f"  Pred time:  {pred_time:.3f}s")
    print(f"\n  Top features:")
    for feat, coef in list(top_features.items())[:5]:
        direction = "â†‘" if coef > 0 else "â†“"
        print(f"    {direction} {feat:<30} {coef:+.4f}")

    return {
        "run_id":           run_id,
        "model_type":       "LogisticRegression",
        "target":           "30day_readmission",
        "data_source":      source,
        "n_total":          len(df_model),
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
# STORE RESULTS
# =============================================================================

def store_results(pg, result):
    sql = """
        INSERT INTO hie.model_result (
            run_id, model_type, target, data_source,
            n_total, n_train, n_test, n_features, positive_rate,
            auc_roc, accuracy, precision_score, recall_score, f1_score,
            feature_time_s, train_time_s, predict_time_s, total_time_s,
            top_features_json, config_json
        ) VALUES (
            %(run_id)s, %(model_type)s, %(target)s, %(data_source)s,
            %(n_total)s, %(n_train)s, %(n_test)s, %(n_features)s, %(positive_rate)s,
            %(auc_roc)s, %(accuracy)s, %(precision_score)s, %(recall_score)s, %(f1_score)s,
            %(feature_time_s)s, %(train_time_s)s, %(predict_time_s)s, %(total_time_s)s,
            %(top_features_json)s, %(config_json)s
        )
        ON CONFLICT (run_id) DO UPDATE SET
            auc_roc      = EXCLUDED.auc_roc,
            total_time_s = EXCLUDED.total_time_s
    """
    result["top_features_json"] = json.dumps(result["top_features_json"])
    result["config_json"]       = json.dumps(result["config_json"])
    with pg.cursor() as cur:
        cur.execute(sql, result)
    pg.commit()
    print(f"\n  âœ“ Results stored in hie.model_result (run_id: {result['run_id']})")

# =============================================================================
# MAIN
# =============================================================================

def main():
    run_id = f"YDB-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    print("=" * 58)
    print("  30-Day Readmission â€” MUMPS Path")
    print(f"  Run ID: {run_id}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 58)

    ydb = setup_ydb()
    print("âœ“ YottaDB connected")

    pg = psycopg2.connect(**PG_CONFIG)
    print("âœ“ Postgres connected (labels only)")

    # Feature extraction from ^PHD
    print("\n[1] Extracting features from ^PHD globals...")
    t_feat = time.time()
    df = extract_patient_features(ydb, pg)
    feat_time = time.time() - t_feat
    print(f"  âœ“ Feature extraction: {feat_time:.2f}s")

    # Train and evaluate
    print("\n[2] Training logistic regression...")
    result = train_and_evaluate(df, run_id, "MUMPS")
    result["feature_time_s"] = float(feat_time)
    result["total_time_s"]   = float(
        feat_time + result["train_time_s"] + result["predict_time_s"])

    # Store
    print("\n[3] Storing results...")
    store_results(pg, result)

    pg.close()

    print("\n" + "=" * 58)
    print(f"  Done. Total time: {result['total_time_s']:.2f}s")
    print(f"  AUC-ROC: {result['auc_roc']:.4f}")
    print("=" * 58)

if __name__ == "__main__":
    main()






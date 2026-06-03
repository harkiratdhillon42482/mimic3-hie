"""
xgboost_liver.py
Phase 1 Baseline: XGBoost Liver Transplant Candidate Prediction

Features: hand-crafted clinical features from liver_pg.py / liver_ydb.py
Target:   label=1 if MELD increased >=5 points OR peak MELD >=25
Stores:   results in hie.model_result

Run inside container:
  cd /data/r2.06_x86_64/g
  python3 /project/liver/xgboost_liver.py --source pg
  python3 /project/liver/xgboost_liver.py --source ydb
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime

PG_CONFIG = {
    "host":     "host.docker.internal",
    "port":     5432,
    "dbname":   "MIMICold",
    "user":     "postgres",
    "password": "YOUR_PASSWORD",
}

# ── Feature columns ────────────────────────────────────────────────────────
# Static features (snapshot at last admission)
STATIC_FEATURES = [
    'age_at_admit',
    'is_male',
    'had_icu_stay',
    'los_hours',
]

# MELD features (most important for transplant)
MELD_FEATURES = [
    'meld_na',
    'meld',
    'bilirubin',
    'creatinine',
    'inr',
    'sodium',
    'peak_meld',
    'max_meld_delta',
]

# Clinical state features
CLINICAL_FEATURES = [
    'has_enceph',
    'has_hepatorenal',
    'has_varices_bleed',
    'has_sbp',
    'has_hcc',
    'has_cirrhosis',
    'has_portal_htn',
    'has_hep_c',
    'has_alcoholic',
    'has_high_acuity',
    'n_diagnoses',
]

# Medication features
MED_FEATURES = [
    'on_lactulose',
    'on_rifaximin',
    'on_spironolactone',
    'on_furosemide',
    'on_albumin',
    'on_nadolol',
    'n_medications',
    'n_unique_drugs',
]

# Temporal features (trajectory)
TEMPORAL_FEATURES = [
    'n_admissions',
    'days_first_to_last',
    'avg_days_between_admits',
    'meld_slope',            # MELD change per day
    'visit_acceleration',    # are visits getting more frequent?
]

ALL_FEATURES = (STATIC_FEATURES + MELD_FEATURES +
                CLINICAL_FEATURES + MED_FEATURES + TEMPORAL_FEATURES)


def get_pg():
    return psycopg2.connect(**PG_CONFIG)


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def build_temporal_features(df_admissions):
    """
    Build trajectory features per patient from admission-level data.
    These are the features that capture the MELD trend over time.
    """
    df = df_admissions.copy()
    df['admittime'] = pd.to_datetime(df['admittime'])
    df = df.sort_values(['subject_id', 'admittime'])

    patient_features = []

    for subject_id, group in df.groupby('subject_id'):
        group = group.sort_values('admittime').reset_index(drop=True)
        n_adm = len(group)

        # Time span
        first_admit = group['admittime'].iloc[0]
        last_admit  = group['admittime'].iloc[-1]
        days_span   = (last_admit - first_admit).days

        # Average days between admissions
        if n_adm > 1:
            gaps = [(group['admittime'].iloc[i] -
                     group['admittime'].iloc[i-1]).days
                    for i in range(1, n_adm)]
            avg_gap = np.mean(gaps)
            # Acceleration: are recent gaps shorter? (negative = accelerating)
            if len(gaps) >= 3:
                recent_gap = np.mean(gaps[-2:])
                early_gap  = np.mean(gaps[:2])
                accel = recent_gap - early_gap  # negative = visits speeding up
            else:
                accel = 0.0
        else:
            avg_gap = 0.0
            accel   = 0.0

        # MELD slope (change per day)
        meld_vals = group['meld_na'].dropna()
        if len(meld_vals) >= 2 and days_span > 0:
            meld_start = meld_vals.iloc[0]
            meld_end   = meld_vals.iloc[-1]
            meld_slope = (meld_end - meld_start) / max(days_span, 1)
        else:
            meld_slope = 0.0

        patient_features.append({
            'subject_id':              subject_id,
            'n_admissions':            n_adm,
            'days_first_to_last':      days_span,
            'avg_days_between_admits': avg_gap,
            'meld_slope':              meld_slope,
            'visit_acceleration':      accel,
        })

    return pd.DataFrame(patient_features)


def prepare_features(df_cohort, df_admissions):
    """
    Merge all features for XGBoost training.
    df_cohort:     one row per patient (last admission snapshot)
    df_admissions: all admissions (for temporal features)
    """
    df = df_cohort.copy()

    # Gender to binary
    df['is_male'] = (df['gender'].str.upper() == 'M').astype(int)

    # LOS
    if 'admittime' in df.columns and 'dischtime' in df.columns:
        df['admittime'] = pd.to_datetime(df['admittime'])
        df['dischtime'] = pd.to_datetime(df['dischtime'])
        df['los_hours'] = (df['dischtime'] - df['admittime']).dt.total_seconds() / 3600
    elif 'los_hours' not in df.columns:
        df['los_hours'] = 0.0

    # Temporal features from admission history
    df_temp = build_temporal_features(df_admissions)
    df = df.merge(df_temp, on='subject_id', how='left')

    # Fill missing
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    df[ALL_FEATURES] = df[ALL_FEATURES].fillna(0.0)

    return df


# =============================================================================
# TRAIN AND EVALUATE
# =============================================================================

def train_xgboost(df, run_id, source):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  Installing xgboost...")
        os.system("pip3 install --break-system-packages xgboost")
        from xgboost import XGBClassifier

    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.metrics import (roc_auc_score, accuracy_score,
                                  precision_score, recall_score, f1_score)
    from sklearn.preprocessing import StandardScaler

    print(f"\n  Dataset: {len(df):,} patients")
    print(f"  Positive rate: {df['label'].mean():.1%}")

    X = df[ALL_FEATURES].values
    y = df['label'].values

    # Time-based split — use admittime order
    df_sorted = df.sort_values('admittime') if 'admittime' in df.columns else df
    split_idx = int(len(df_sorted) * 0.8)

    X_train = df_sorted[ALL_FEATURES].values[:split_idx]
    X_test  = df_sorted[ALL_FEATURES].values[split_idx:]
    y_train = df_sorted['label'].values[:split_idx]
    y_test  = df_sorted['label'].values[split_idx:]

    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

    # Class weight
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    # Train
    t_train = time.time()
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
    train_time = time.time() - t_train

    # Predict
    t_pred = time.time()
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    pred_time = time.time() - t_pred

    # Metrics
    auc  = roc_auc_score(y_test, y_prob)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    # Feature importance
    importance = sorted(
        zip(ALL_FEATURES, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )[:10]
    top_features = {k: round(float(v), 4) for k, v in importance}

    print(f"\n  ── Results ──────────────────────────────")
    print(f"  AUC-ROC:   {auc:.4f}  (published MIMIC: 0.75-0.82)")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Train:     {train_time:.2f}s")
    print(f"\n  Top features:")
    for feat, imp in list(top_features.items())[:8]:
        bar = "█" * int(imp * 50)
        print(f"    {feat:<30} {imp:.4f} {bar}")

    return {
        "run_id":            run_id,
        "model_type":        "XGBoost",
        "target":            "liver_transplant_candidate",
        "data_source":       source,
        "n_total":           len(df),
        "n_train":           len(X_train),
        "n_test":            len(X_test),
        "n_features":        len(ALL_FEATURES),
        "positive_rate":     float(y.mean()),
        "auc_roc":           float(auc),
        "accuracy":          float(acc),
        "precision_score":   float(prec),
        "recall_score":      float(rec),
        "f1_score":          float(f1),
        "train_time_s":      float(train_time),
        "predict_time_s":    float(pred_time),
        "top_features_json": top_features,
        "config_json": {
            "n_estimators": 300, "max_depth": 6,
            "learning_rate": 0.05, "scale_pos_weight": round(pos_weight, 2)
        },
    }


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
            %(n_total)s, %(n_train)s, %(n_test)s, %(n_features)s,
            %(positive_rate)s, %(auc_roc)s, %(accuracy)s,
            %(precision_score)s, %(recall_score)s, %(f1_score)s,
            %(train_time_s)s, %(predict_time_s)s, %(total_time_s)s,
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
    print(f"\n  ✓ Results stored (run_id: {result['run_id']})")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['pg','ydb'], default='pg')
    args = parser.parse_args()

    source = args.source.upper()
    run_id = f"XGB-LIVER-{source}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    print("=" * 60)
    print(f"  XGBoost Liver Transplant — {source} Path")
    print(f"  Run ID: {run_id}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load cohort data
    cohort_file = f'/project/liver/liver_cohort_{args.source}.csv'
    admissions_file = f'/project/liver/liver_admissions_{args.source}.csv'

    if not os.path.exists(cohort_file):
        print(f"\n  ✗ {cohort_file} not found.")
        print(f"  Run liver_{args.source}.py first.")
        sys.exit(1)

    print(f"\n[1] Loading cohort data from {cohort_file}...")
    t0 = time.time()
    df_cohort     = pd.read_csv(cohort_file)
    df_admissions = pd.read_csv(admissions_file)
    print(f"  ✓ {len(df_cohort):,} patients, "
          f"{len(df_admissions):,} admissions ({time.time()-t0:.1f}s)")

    print(f"\n[2] Engineering features...")
    t0 = time.time()
    df = prepare_features(df_cohort, df_admissions)
    feat_time = time.time() - t0
    print(f"  ✓ {len(ALL_FEATURES)} features ready ({feat_time:.1f}s)")

    print(f"\n[3] Training XGBoost...")
    result = train_xgboost(df, run_id, source)
    result["feature_time_s"] = float(feat_time)
    result["total_time_s"]   = float(
        feat_time + result["train_time_s"] + result["predict_time_s"])

    print(f"\n[4] Storing results...")
    pg = get_pg()
    store_results(pg, result)
    pg.close()

    print("\n" + "=" * 60)
    print(f"  Done. Total: {result['total_time_s']:.1f}s")
    print(f"  AUC-ROC: {result['auc_roc']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
liver_pg.py
Liver Transplant Candidate Identification — Postgres Path

Reads from:  public.* (MIMIC-III source tables)
             hie.*    (our OMOP-aligned analytics layer)
Computes:    MELD / MELD-Na per admission
Labels:      Positive = MELD increased >= 5 points across admissions
             Negative = liver patient, stable or low MELD
Outputs:     hie.liver_cohort table + liver_cohort_pg.csv

Run inside container:
  cd /data/r2.06_x86_64/g
  python3 /project/liver/liver_pg.py
"""

import os
import sys
import time
import json
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

# ── ICD-9 codes for liver transplant candidates ────────────────────────────
LIVER_ICD9 = [
    # Alcoholic liver disease
    '5710', '5711', '5712', '5713',
    # Cirrhosis
    '5715', '5716', '5718', '5719',
    # Hepatic failure / encephalopathy
    '5720', '5722', '5724', '5728',
    # Ascites / SBP
    '7891', '5671',
    # Portal hypertension / varices
    '5723', '45620', '45621', '45680',
    # HCC
    '1550', '1551', '1552',
    # Chronic hepatitis
    '07054', '07044', '07032', '07070',
]

# ── High-acuity codes (decompensated liver disease) ───────────────────────
HIGH_ACUITY = {
    '5720', '5722', '5724',    # failure, encephalopathy, hepatorenal
    '45620',                    # bleeding varices
    '5671',                     # SBP
}

# ── MELD lab itemids ───────────────────────────────────────────────────────
BILI_ITEMS   = [50885]
CREAT_ITEMS  = [50912]
INR_ITEMS    = [51237]
SODIUM_ITEMS = [50983, 50824]

# =============================================================================
# MELD COMPUTATION
# =============================================================================

def compute_meld(bili, creat, inr, on_dialysis=False):
    """MELD score per UNOS formula."""
    if any(v is None or np.isnan(v) for v in [bili, creat, inr]):
        return None
    bili  = max(1.0, min(bili, 82.0))
    inr   = max(1.0, min(inr, 10.0))
    creat = 4.0 if on_dialysis else max(1.0, min(4.0, creat))
    meld  = 3.78 * np.log(bili) + 11.2 * np.log(inr) + 9.57 * np.log(creat) + 6.43
    return round(min(40.0, max(6.0, meld)), 1)


def compute_meld_na(bili, creat, inr, sodium, on_dialysis=False):
    """MELD-Na score (current UNOS standard since 2016)."""
    meld = compute_meld(bili, creat, inr, on_dialysis)
    if meld is None or sodium is None or np.isnan(sodium):
        return None
    sodium  = max(125.0, min(137.0, sodium))
    meld_na = meld + 1.32 * (137 - sodium) - (0.033 * meld * (137 - sodium))
    return round(min(40.0, max(6.0, meld_na)), 1)


def meld_severity(score):
    if score is None:      return 'Unknown'
    if score <= 9:         return 'Low'
    if score <= 19:        return 'Moderate'
    if score <= 29:        return 'High'
    if score <= 39:        return 'Very_High'
    return 'Maximum'

# =============================================================================
# STEP 1: IDENTIFY LIVER PATIENTS
# =============================================================================

COHORT_QUERY = """
WITH liver_diagnoses AS (
    SELECT DISTINCT
        d.subject_id,
        d.hadm_id,
        d.icd9_code,
        d.seq_num,
        di.short_title
    FROM public.diagnoses_icd d
    JOIN public.d_icd_diagnoses di ON d.icd9_code = di.icd9_code
    WHERE d.icd9_code = ANY(%(codes)s)
),
liver_admissions AS (
    SELECT DISTINCT
        a.subject_id,
        a.hadm_id,
        a.admittime,
        a.dischtime,
        a.admission_type,
        a.hospital_expire_flag,
        p.gender,
        p.dob,
        p.dod,
        p.expire_flag,
        -- Age at admission (MIMIC uses shifted DOB)
        EXTRACT(YEAR FROM AGE(a.admittime, p.dob)) AS age_at_admit
    FROM public.admissions a
    JOIN public.patients p ON a.subject_id = p.subject_id
    WHERE a.subject_id IN (SELECT DISTINCT subject_id FROM liver_diagnoses)
      AND a.admittime IS NOT NULL
      AND a.dischtime IS NOT NULL
),
icu_stays AS (
    SELECT hadm_id, MAX(los) AS max_icu_los
    FROM public.icustays
    GROUP BY hadm_id
)
SELECT
    la.*,
    COALESCE(icu.max_icu_los, 0)   AS max_icu_los,
    icu.max_icu_los IS NOT NULL    AS had_icu_stay
FROM liver_admissions la
LEFT JOIN icu_stays icu ON la.hadm_id = icu.hadm_id
WHERE la.age_at_admit BETWEEN 18 AND 100
ORDER BY la.subject_id, la.admittime;
"""

# =============================================================================
# STEP 2: GET MELD LABS PER ADMISSION
# =============================================================================

MELD_LAB_QUERY = """
SELECT
    l.subject_id,
    l.hadm_id,
    l.itemid,
    l.valuenum,
    l.charttime
FROM public.labevents l
WHERE l.subject_id = ANY(%(pids)s)
  AND l.itemid = ANY(%(items)s)
  AND l.valuenum IS NOT NULL
  AND l.valuenum > 0
ORDER BY l.subject_id, l.hadm_id, l.charttime;
"""

# =============================================================================
# STEP 3: GET DIAGNOSES PER ADMISSION
# =============================================================================

DX_QUERY = """
SELECT
    d.subject_id,
    d.hadm_id,
    d.icd9_code,
    d.seq_num,
    di.short_title
FROM public.diagnoses_icd d
JOIN public.d_icd_diagnoses di ON d.icd9_code = di.icd9_code
WHERE d.subject_id = ANY(%(pids)s)
ORDER BY d.subject_id, d.hadm_id, d.seq_num;
"""

# =============================================================================
# STEP 4: GET MEDICATIONS
# =============================================================================

MED_QUERY = """
SELECT
    subject_id,
    hadm_id,
    drug,
    drug_name_generic,
    route,
    startdate,
    enddate
FROM public.prescriptions
WHERE subject_id = ANY(%(pids)s)
  AND drug IS NOT NULL
ORDER BY subject_id, hadm_id, startdate;
"""

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def get_meld_labs(pg, subject_ids):
    """Fetch all MELD-relevant labs for the cohort."""
    all_items = BILI_ITEMS + CREAT_ITEMS + INR_ITEMS + SODIUM_ITEMS
    cur = pg.cursor()
    cur.execute(MELD_LAB_QUERY, {
        'pids':  list(subject_ids),
        'items': all_items
    })
    rows = cur.fetchall()
    cur.close()

    df = pd.DataFrame(rows, columns=[
        'subject_id','hadm_id','itemid','valuenum','charttime'])
    df['charttime'] = pd.to_datetime(df['charttime'])
    return df


def compute_meld_per_admission(df_cohort, df_labs):
    """
    For each admission compute MELD using most recent lab within admission window.
    """
    results = []

    for _, row in df_cohort.iterrows():
        pid   = row['subject_id']
        hid   = row['hadm_id']
        admit = pd.to_datetime(row['admittime'])
        disch = pd.to_datetime(row['dischtime'])

        # Labs within this admission window
        mask = (
            (df_labs['subject_id'] == pid) &
            (df_labs['hadm_id'] == hid) &
            (df_labs['charttime'] >= admit) &
            (df_labs['charttime'] <= disch)
        )
        adm_labs = df_labs[mask]

        def latest(items):
            sub = adm_labs[adm_labs['itemid'].isin(items)]
            if len(sub) == 0: return None
            return sub.sort_values('charttime', ascending=False)['valuenum'].iloc[0]

        bili   = latest(BILI_ITEMS)
        creat  = latest(CREAT_ITEMS)
        inr    = latest(INR_ITEMS)
        sodium = latest(SODIUM_ITEMS)

        meld    = compute_meld(bili, creat, inr)
        meld_na = compute_meld_na(bili, creat, inr, sodium)

        results.append({
            'subject_id':   pid,
            'hadm_id':      hid,
            'admittime':    admit,
            'dischtime':    disch,
            'gender':       row['gender'],
            'age_at_admit': row['age_at_admit'],
            'admission_type': row['admission_type'],
            'hospital_expire_flag': row['hospital_expire_flag'],
            'had_icu_stay': row['had_icu_stay'],
            'max_icu_los':  row['max_icu_los'],
            'bilirubin':    bili,
            'creatinine':   creat,
            'inr':          inr,
            'sodium':       sodium,
            'meld':         meld,
            'meld_na':      meld_na,
            'severity':     meld_severity(meld_na),
        })

    return pd.DataFrame(results)


def assign_labels(df_meld):
    """
    Assign transplant candidate labels based on MELD trajectory.

    Positive (label=1): Patient's MELD increased >= 5 points across
                        any two consecutive admissions
    Negative (label=0): Liver patient with stable or low MELD

    Also flag high-acuity patients who always qualify regardless.
    """
    df = df_meld.sort_values(['subject_id', 'admittime']).copy()

    # Compute MELD change per patient across admissions
    df['meld_prev'] = df.groupby('subject_id')['meld_na'].shift(1)
    df['meld_delta'] = df['meld_na'] - df['meld_prev']
    df['days_since_prev'] = (
        df['admittime'] -
        df.groupby('subject_id')['admittime'].shift(1)
    ).dt.days

    # MELD trajectory per patient: max MELD increase across any admission
    patient_max_delta = df.groupby('subject_id')['meld_delta'].max()

    # Peak MELD per patient
    patient_peak_meld = df.groupby('subject_id')['meld_na'].max()

    # Assign label at patient level
    df['peak_meld']     = df['subject_id'].map(patient_peak_meld)
    df['max_meld_delta'] = df['subject_id'].map(patient_max_delta)

    # Label logic:
    # Positive = peak MELD >= 15 AND (MELD increased >= 5 points OR peak >= 25)
    # Negative = peak MELD < 15 OR stable (delta < 5 and peak < 25)
    df['label'] = 0
    pos_mask = (
        (df['peak_meld'] >= 15) &
        ((df['max_meld_delta'] >= 5) | (df['peak_meld'] >= 25))
    ).astype(int)
    df['label'] = pos_mask

    # Use the LAST admission per patient as the prediction row
    # (most complete history at that point)
    df_last = df.sort_values('admittime').groupby('subject_id').last().reset_index()

    return df, df_last


def get_dx_features(pg, subject_ids):
    """Get diagnosis features per admission."""
    cur = pg.cursor()
    cur.execute(DX_QUERY, {'pids': list(subject_ids)})
    rows = cur.fetchall()
    cur.close()
    df = pd.DataFrame(rows, columns=[
        'subject_id','hadm_id','icd9_code','seq_num','short_title'])

    liver_high = set(HIGH_ACUITY)

    features = df.groupby(['subject_id','hadm_id']).apply(lambda g: pd.Series({
        'n_diagnoses':      len(g),
        'has_enceph':       int(g['icd9_code'].isin(['5722']).any()),
        'has_hepatorenal':  int(g['icd9_code'].isin(['5724']).any()),
        'has_varices_bleed':int(g['icd9_code'].isin(['45620']).any()),
        'has_sbp':          int(g['icd9_code'].isin(['5671']).any()),
        'has_hcc':          int(g['icd9_code'].isin(['1550','1551','1552']).any()),
        'has_high_acuity':  int(g['icd9_code'].isin(liver_high).any()),
        'has_cirrhosis':    int(g['icd9_code'].isin(['5715','5712','5716']).any()),
        'has_portal_htn':   int(g['icd9_code'].isin(['5723']).any()),
        'has_hep_c':        int(g['icd9_code'].isin(['07054','07070']).any()),
        'has_alcoholic':    int(g['icd9_code'].isin(['5710','5711','5712','5713']).any()),
    })).reset_index()

    return features


def get_med_features(pg, subject_ids):
    """Get medication features — liver-relevant drugs."""
    cur = pg.cursor()
    cur.execute(MED_QUERY, {'pids': list(subject_ids)})
    rows = cur.fetchall()
    cur.close()
    df = pd.DataFrame(rows, columns=[
        'subject_id','hadm_id','drug','drug_name_generic','route','startdate','enddate'])

    drug_lower = df['drug'].str.lower().fillna('')

    df['is_lactulose']     = drug_lower.str.contains('lactulose', na=False).astype(int)
    df['is_rifaximin']     = drug_lower.str.contains('rifaximin', na=False).astype(int)
    df['is_spironolactone']= drug_lower.str.contains('spironolactone', na=False).astype(int)
    df['is_furosemide']    = drug_lower.str.contains('furosemide', na=False).astype(int)
    df['is_albumin']       = drug_lower.str.contains('albumin', na=False).astype(int)
    df['is_nadolol']       = drug_lower.str.contains('nadolol|propranolol', na=False).astype(int)
    df['is_norfloxacin']   = drug_lower.str.contains('norfloxacin|ciprofloxacin', na=False).astype(int)

    features = df.groupby(['subject_id','hadm_id']).agg(
        n_medications    = ('drug', 'count'),
        n_unique_drugs   = ('drug', 'nunique'),
        on_lactulose     = ('is_lactulose', 'max'),
        on_rifaximin     = ('is_rifaximin', 'max'),
        on_spironolactone= ('is_spironolactone', 'max'),
        on_furosemide    = ('is_furosemide', 'max'),
        on_albumin       = ('is_albumin', 'max'),
        on_nadolol       = ('is_nadolol', 'max'),
        on_norfloxacin   = ('is_norfloxacin', 'max'),
    ).reset_index()

    return features


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  Liver Transplant Cohort — Postgres Path")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    pg = psycopg2.connect(**PG_CONFIG)

    # ── Step 1: Identify liver patients ───────────────────────────────────
    print("\n[1] Identifying liver patients...")
    t0 = time.time()
    cur = pg.cursor()
    cur.execute(COHORT_QUERY, {'codes': LIVER_ICD9})
    rows = cur.fetchall()
    cur.close()

    cols = ['subject_id','hadm_id','admittime','dischtime',
            'admission_type','hospital_expire_flag','gender',
            'dob','dod','expire_flag','age_at_admit',
            'max_icu_los','had_icu_stay']
    df_cohort = pd.DataFrame(rows, columns=cols)
    print(f"  ✓ {df_cohort['subject_id'].nunique():,} patients, "
          f"{len(df_cohort):,} admissions ({time.time()-t0:.1f}s)")

    subject_ids = df_cohort['subject_id'].unique().tolist()

    # ── Step 2: MELD labs ──────────────────────────────────────────────────
    print("\n[2] Fetching MELD labs...")
    t0 = time.time()
    df_labs = get_meld_labs(pg, subject_ids)
    print(f"  ✓ {len(df_labs):,} lab results ({time.time()-t0:.1f}s)")

    # ── Step 3: Compute MELD ───────────────────────────────────────────────
    print("\n[3] Computing MELD scores...")
    t0 = time.time()
    df_meld = compute_meld_per_admission(df_cohort, df_labs)
    n_scored = df_meld['meld_na'].notna().sum()
    print(f"  ✓ {n_scored:,} admissions scored ({time.time()-t0:.1f}s)")
    print(f"  MELD-Na distribution:")
    print(df_meld['severity'].value_counts().sort_index().to_string())

    # ── Step 4: Diagnosis features ─────────────────────────────────────────
    print("\n[4] Extracting diagnosis features...")
    t0 = time.time()
    df_dx = get_dx_features(pg, subject_ids)
    print(f"  ✓ {len(df_dx):,} admission-level dx features ({time.time()-t0:.1f}s)")

    # ── Step 5: Medication features ────────────────────────────────────────
    print("\n[5] Extracting medication features...")
    t0 = time.time()
    df_meds = get_med_features(pg, subject_ids)
    print(f"  ✓ {len(df_meds):,} admission-level med features ({time.time()-t0:.1f}s)")

    # ── Step 6: Assign labels ──────────────────────────────────────────────
    print("\n[6] Assigning trajectory labels...")
    df_all, df_last = assign_labels(df_meld)
    n_pos = (df_last['label'] == 1).sum()
    n_neg = (df_last['label'] == 0).sum()
    print(f"  Positive (transplant candidate): {n_pos:,} ({n_pos/(n_pos+n_neg)*100:.1f}%)")
    print(f"  Negative (medically managed):    {n_neg:,} ({n_neg/(n_pos+n_neg)*100:.1f}%)")

    # ── Step 7: Merge all features ─────────────────────────────────────────
    print("\n[7] Merging feature sets...")
    df_final = df_last.merge(df_dx,  on=['subject_id','hadm_id'], how='left')
    df_final = df_final.merge(df_meds, on=['subject_id','hadm_id'], how='left')

    # Fill missing features
    bool_cols = [c for c in df_final.columns if c.startswith(('has_','on_','n_'))]
    df_final[bool_cols] = df_final[bool_cols].fillna(0)

    # Also save full admission-level data for tokenizer
    df_all_merged = df_all.merge(df_dx,  on=['subject_id','hadm_id'], how='left')
    df_all_merged = df_all_merged.merge(df_meds, on=['subject_id','hadm_id'], how='left')
    df_all_merged[bool_cols] = df_all_merged[[c for c in bool_cols
                                if c in df_all_merged.columns]].fillna(0)

    # ── Step 8: Save ───────────────────────────────────────────────────────
    print("\n[8] Saving results...")
    df_final.to_csv('/project/liver/liver_cohort_pg.csv', index=False)
    df_all_merged.to_csv('/project/liver/liver_admissions_pg.csv', index=False)
    print(f"  ✓ liver_cohort_pg.csv     ({len(df_final):,} patients)")
    print(f"  ✓ liver_admissions_pg.csv ({len(df_all_merged):,} admissions)")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Liver patients:          {df_final['subject_id'].nunique():,}")
    print(f"  Total admissions:        {len(df_all_merged):,}")
    print(f"  MELD-scoreable:          {df_all_merged['meld_na'].notna().sum():,}")
    print(f"  Positive labels:         {n_pos:,} ({n_pos/(n_pos+n_neg)*100:.1f}%)")
    print(f"  Negative labels:         {n_neg:,}")
    print(f"\n  MELD stats (patients):")
    print(f"    Mean MELD-Na:  {df_final['meld_na'].mean():.1f}")
    print(f"    Median MELD-Na: {df_final['meld_na'].median():.1f}")
    print(f"    Max MELD-Na:   {df_final['meld_na'].max():.1f}")

    pg.close()
    print("\n  ✓ Done")


if __name__ == "__main__":
    main()

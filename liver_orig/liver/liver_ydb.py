"""
liver_ydb.py
Liver Transplant Candidate Identification — MUMPS Path

Reads from:  YottaDB ^PHD globals
Computes:    MELD / MELD-Na per admission from ^PHD LAB nodes
Labels:      Same trajectory logic as liver_pg.py
Outputs:     liver_cohort_ydb.csv + liver_admissions_ydb.csv

Run inside container:
  cd /data/r2.06_x86_64/g
  python3 /project/liver/liver_ydb.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime

YDB_ENV = {
    "ydb_dist":    "/opt/yottadb/current",
    "ydb_gbldir":  "/data/r2.06_x86_64/g/yottadb.gld",
    "ydb_routines": "/opt/yottadb/current/libyottadbutil.so",
}

PG_CONFIG = {
    "host":     "host.docker.internal",
    "port":     5432,
    "dbname":   "MIMICold",
    "user":     "postgres",
    "password": "YOUR_PASSWORD",
}

# ── Liver ICD-9 codes ──────────────────────────────────────────────────────
LIVER_ICD9 = {
    '5710','5711','5712','5713',
    '5715','5716','5718','5719',
    '5720','5722','5724','5728',
    '7891','5671','5723',
    '45620','45621','45680',
    '1550','1551','1552',
    '07054','07044','07032','07070',
}

HIGH_ACUITY = {'5720','5722','5724','45620','5671'}

# MELD lab itemids in ^PHD LAB nodes: stored as itemid^value^valuenum^uom^flag^dt
BILI_ITEMS   = {50885}
CREAT_ITEMS  = {50912}
INR_ITEMS    = {51237}
SODIUM_ITEMS = {50983, 50824}
ALL_MELD_ITEMS = BILI_ITEMS | CREAT_ITEMS | INR_ITEMS | SODIUM_ITEMS

# ── YDB helpers ────────────────────────────────────────────────────────────

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
    """YDB compact datetime YYYYMMDDHHMMSS → datetime or None"""
    if not s or len(s) < 8: return None
    try:
        digits = "".join(c for c in str(s) if c.isdigit())
        if len(digits) < 8: return None
        yr = int(digits[0:4])
        mo = int(digits[4:6])
        dy = int(digits[6:8])
        if not (1800 <= yr <= 2200 and 1 <= mo <= 12 and 1 <= dy <= 31):
            return None
        hr = int(digits[8:10]) if len(digits) >= 10 else 0
        mn = int(digits[10:12]) if len(digits) >= 12 else 0
        return datetime(yr, mo, dy, hr, mn)
    except Exception:
        return None

# =============================================================================
# MELD COMPUTATION (same as liver_pg.py)
# =============================================================================

def compute_meld(bili, creat, inr, on_dialysis=False):
    if any(v is None or (isinstance(v, float) and np.isnan(v))
           for v in [bili, creat, inr]):
        return None
    bili  = max(1.0, min(float(bili),  82.0))
    inr   = max(1.0, min(float(inr),   10.0))
    creat = 4.0 if on_dialysis else max(1.0, min(4.0, float(creat)))
    meld  = 3.78 * np.log(bili) + 11.2 * np.log(inr) + 9.57 * np.log(creat) + 6.43
    return round(min(40.0, max(6.0, meld)), 1)


def compute_meld_na(bili, creat, inr, sodium, on_dialysis=False):
    meld = compute_meld(bili, creat, inr, on_dialysis)
    if meld is None or sodium is None: return None
    sodium  = max(125.0, min(137.0, float(sodium)))
    meld_na = meld + 1.32 * (137 - sodium) - (0.033 * meld * (137 - sodium))
    return round(min(40.0, max(6.0, meld_na)), 1)


def meld_severity(score):
    if score is None:  return 'Unknown'
    if score <= 9:     return 'Low'
    if score <= 19:    return 'Moderate'
    if score <= 29:    return 'High'
    if score <= 39:    return 'Very_High'
    return 'Maximum'

# =============================================================================
# FEATURE EXTRACTION FROM ^PHD
# =============================================================================

def extract_patient_features(ydb, pg):
    """
    Traverse ^PHD for all liver patients.
    Returns two DataFrames:
      df_admissions: one row per admission with MELD + features
      df_labs_raw:   raw lab readings for trend analysis
    """

    # Load readmission labels from Postgres for age/gender (not in ^PHD directly)
    print("  Loading demographics from Postgres...")
    cur = pg.cursor()
    cur.execute("""
        SELECT person_id, gender_concept_code, year_of_birth, deceased
        FROM hie.person
    """)
    demo = {row[0]: {'sex': row[1], 'yob': row[2], 'deceased': row[3]}
            for row in cur.fetchall()}
    cur.close()
    print(f"  ✓ {len(demo):,} patient demographics loaded")

    records     = []
    n_patients  = 0
    n_liver_pts = 0
    n_encounters = 0
    t0          = time.time()

    print("  Traversing ^PHD globals...")

    for pid in walk(ydb, "^PHD", []):
        if not pid.startswith("HIE-"):
            continue
        n_patients += 1

        # ── Walk encounters ────────────────────────────────────────────────
        is_liver_patient = False

        for eid in walk(ydb, "^PHD", [pid, "VISIT"]):
            root = yget(ydb, "^PHD", [pid, "VISIT", eid, "0"])
            if not root: continue

            parts = (root + "^^^^^^^").split("^")
            admit_raw = parts[0]
            disch_raw = parts[1]
            adm_type  = parts[2]
            expire    = parts[5]
            los_s     = parts[7] if len(parts) > 7 else ""

            admit_dt = parse_dt(admit_raw)
            disch_dt = parse_dt(disch_raw)
            if not admit_dt or not disch_dt: continue

            # ── Diagnoses ──────────────────────────────────────────────────
            dx_codes = set()
            for seq in walk(ydb, "^PHD", [pid, "VISIT", eid, "DX"]):
                val = yget(ydb, "^PHD", [pid, "VISIT", eid, "DX", seq])
                code = val.split("^")[0] if val else ""
                if code: dx_codes.add(code)

            # Skip non-liver encounters
            if not dx_codes.intersection(LIVER_ICD9):
                continue

            is_liver_patient = True
            n_encounters += 1

            # ── MELD labs from ^PHD ────────────────────────────────────────
            # ^PHD(pid,"VISIT",eid,"LAB",n) = itemid^value^valuenum^uom^flag^dt
            bili_vals   = []
            creat_vals  = []
            inr_vals    = []
            sodium_vals = []

            for idx in walk(ydb, "^PHD", [pid, "VISIT", eid, "LAB"]):
                val = yget(ydb, "^PHD", [pid, "VISIT", eid, "LAB", idx])
                if not val: continue
                lp = (val + "^^^^^").split("^")
                try:
                    item_id  = int(lp[0]) if lp[0].isdigit() else 0
                    valuenum = float(lp[2]) if lp[2] else None
                except Exception:
                    continue
                if valuenum is None or valuenum <= 0: continue

                if item_id in BILI_ITEMS:   bili_vals.append(valuenum)
                if item_id in CREAT_ITEMS:  creat_vals.append(valuenum)
                if item_id in INR_ITEMS:    inr_vals.append(valuenum)
                if item_id in SODIUM_ITEMS: sodium_vals.append(valuenum)

            # Use most recent (last in list since stored chronologically)
            bili   = bili_vals[-1]   if bili_vals   else None
            creat  = creat_vals[-1]  if creat_vals  else None
            inr    = inr_vals[-1]    if inr_vals    else None
            sodium = sodium_vals[-1] if sodium_vals else None

            meld    = compute_meld(bili, creat, inr)
            meld_na = compute_meld_na(bili, creat, inr, sodium)

            # ── Medication features ────────────────────────────────────────
            med_names = []
            for idx in walk(ydb, "^PHD", [pid, "VISIT", eid, "MED"]):
                val = yget(ydb, "^PHD", [pid, "VISIT", eid, "MED", idx])
                drug = val.split("^")[0].lower() if val else ""
                if drug: med_names.append(drug)

            n_meds = len(med_names)
            med_str = " ".join(med_names)

            # ── Notes count ────────────────────────────────────────────────
            n_notes = sum(1 for _ in walk(ydb, "^PHD",
                                           [pid, "VISIT", eid, "NOTE"]))

            # ── ICU ────────────────────────────────────────────────────────
            n_icu = sum(1 for _ in walk(ydb, "^PHD",
                                         [pid, "VISIT", eid, "ICU"]))

            # ── Demographics ───────────────────────────────────────────────
            d = demo.get(pid, {})
            sex = d.get('sex', 'U')
            yob = d.get('yob')
            age = (admit_dt.year - yob) if yob else None
            if age is not None and not (18 <= age <= 100):
                continue

            # ── Diagnosis feature flags ────────────────────────────────────
            records.append({
                'pid':          pid,
                'eid':          eid,
                'subject_id':   int(pid.replace('HIE-MIII-', '')),
                'hadm_id':      int(eid.replace('ENC-MIII-', '')),
                'admittime':    admit_dt,
                'dischtime':    disch_dt,
                'gender':       sex,
                'age_at_admit': age,
                'admission_type': adm_type,
                'hospital_expire_flag': int(expire) if expire.isdigit() else 0,
                'had_icu_stay': int(n_icu > 0),
                'bilirubin':   bili,
                'creatinine':  creat,
                'inr':         inr,
                'sodium':      sodium,
                'meld':        meld,
                'meld_na':     meld_na,
                'severity':    meld_severity(meld_na),
                'n_diagnoses': len(dx_codes),
                'has_enceph':         int('5722' in dx_codes),
                'has_hepatorenal':    int('5724' in dx_codes),
                'has_varices_bleed':  int('45620' in dx_codes),
                'has_sbp':            int('5671' in dx_codes),
                'has_hcc':            int(bool(dx_codes & {'1550','1551','1552'})),
                'has_high_acuity':    int(bool(dx_codes & HIGH_ACUITY)),
                'has_cirrhosis':      int(bool(dx_codes & {'5715','5712','5716'})),
                'has_portal_htn':     int('5723' in dx_codes),
                'has_hep_c':          int(bool(dx_codes & {'07054','07070'})),
                'has_alcoholic':      int(bool(dx_codes & {'5710','5711','5712','5713'})),
                'n_medications':      n_meds,
                'n_unique_drugs':     len(set(med_names)),
                'on_lactulose':       int('lactulose' in med_str),
                'on_rifaximin':       int('rifaximin' in med_str),
                'on_spironolactone':  int('spironolactone' in med_str),
                'on_furosemide':      int('furosemide' in med_str),
                'on_albumin':         int('albumin' in med_str),
                'on_nadolol':         int('nadolol' in med_str or 'propranolol' in med_str),
                'n_notes':            n_notes,
            })

        if is_liver_patient:
            n_liver_pts += 1

        if n_patients % 2000 == 0:
            rate = n_patients / (time.time() - t0)
            print(f"\r  Patients: {n_patients:,}  Liver: {n_liver_pts:,}  "
                  f"Encounters: {n_encounters:,}  Speed: {rate:.0f} pt/s",
                  end="", flush=True)

    print(f"\n  ✓ {n_patients:,} patients scanned, "
          f"{n_liver_pts:,} liver patients, "
          f"{n_encounters:,} encounters in {time.time()-t0:.1f}s")

    return pd.DataFrame(records)


def assign_labels(df):
    """Same label logic as liver_pg.py."""
    df = df.sort_values(['subject_id', 'admittime']).copy()
    df['meld_prev']     = df.groupby('subject_id')['meld_na'].shift(1)
    df['meld_delta']    = df['meld_na'] - df['meld_prev']
    df['days_since_prev'] = (
        df['admittime'] -
        df.groupby('subject_id')['admittime'].shift(1)
    ).dt.days

    patient_max_delta = df.groupby('subject_id')['meld_delta'].max()
    patient_peak_meld = df.groupby('subject_id')['meld_na'].max()

    df['peak_meld']      = df['subject_id'].map(patient_peak_meld)
    df['max_meld_delta'] = df['subject_id'].map(patient_max_delta)

    pos_mask = (
        (df['peak_meld'] >= 15) &
        ((df['max_meld_delta'] >= 5) | (df['peak_meld'] >= 25))
    ).astype(int)
    df['label'] = pos_mask

    df_last = df.sort_values('admittime').groupby('subject_id').last().reset_index()
    return df, df_last

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  Liver Transplant Cohort — MUMPS Path")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print("\nConnecting to YottaDB...")
    ydb = setup_ydb()
    print("  ✓ YottaDB connected")

    print("Connecting to Postgres (demographics only)...")
    pg = psycopg2.connect(**PG_CONFIG)
    print("  ✓ Postgres connected")

    # Feature extraction
    print("\n[1] Extracting features from ^PHD globals...")
    t_feat = time.time()
    df_admissions = extract_patient_features(ydb, pg)
    feat_time = time.time() - t_feat
    print(f"  ✓ Feature extraction: {feat_time:.1f}s")

    if len(df_admissions) == 0:
        print("  ✗ No liver patients found in ^PHD")
        return

    # Assign labels
    print("\n[2] Assigning trajectory labels...")
    df_all, df_last = assign_labels(df_admissions)
    n_pos = (df_last['label'] == 1).sum()
    n_neg = (df_last['label'] == 0).sum()
    print(f"  Positive: {n_pos:,} ({n_pos/(n_pos+n_neg)*100:.1f}%)")
    print(f"  Negative: {n_neg:,} ({n_neg/(n_pos+n_neg)*100:.1f}%)")

    # Save
    print("\n[3] Saving results...")
    os.makedirs('/project/liver', exist_ok=True)
    df_last.to_csv('/project/liver/liver_cohort_ydb.csv', index=False)
    df_all.to_csv('/project/liver/liver_admissions_ydb.csv', index=False)
    print(f"  ✓ liver_cohort_ydb.csv     ({len(df_last):,} patients)")
    print(f"  ✓ liver_admissions_ydb.csv ({len(df_all):,} admissions)")

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Liver patients:    {df_last['subject_id'].nunique():,}")
    print(f"  Total admissions:  {len(df_all):,}")
    print(f"  MELD-scoreable:    {df_all['meld_na'].notna().sum():,}")
    print(f"  Mean MELD-Na:      {df_last['meld_na'].mean():.1f}")
    print(f"  Feature time:      {feat_time:.1f}s")

    pg.close()
    print("\n  ✓ Done")


if __name__ == "__main__":
    main()

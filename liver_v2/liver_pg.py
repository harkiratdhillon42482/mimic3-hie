"""
liver_pg.py
Liver Disease Progression & Transplant Candidate Identification — Postgres Path

Label Strategy: OPTION C (Composite — ICU-appropriate for MIMIC-III)
  Positive = ANY of:
    - Peak MELD-Na >= 25 across admissions
    - MELD increased >= 5 points across admissions
    - Had hepatorenal syndrome (5724)
    - Had hepatic failure / encephalopathy (5722)
    - Had bleeding varices (45620)
    - Died in hospital (hospital_expire_flag=1) with peak MELD >= 15

  Why Option C for MIMIC-III:
    MIMIC is ICU-only data. Most liver patients are already decompensated.
    A composite label captures the full spectrum of transplant-level acuity
    rather than relying solely on MELD trajectory, which is noisier in ICU.

Ghandian et al. (2022) framework with extensions:
  - Lab summary statistics (prior mean/std/min/max/count)
  - MELD trajectory (slope, acceleration, variability)
  - Disease stage tracking (NAFL→NASH→Fibrosis→Cirrhosis→Failure)
  - Medication trajectory as effect modifiers
  - Multi-horizon prediction targets (30d, 60d, 90d)
  - Utilization features (visit acceleration, ED visits)

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
from datetime import datetime, timedelta

PG_CONFIG = {
    "host":     "host.docker.internal",
    "port":     5432,
    "dbname":   "MIMICold",
    "user":     "postgres",
    "password": "YOUR_PASSWORD",
}

# ── Liver ICD-9 codes ──────────────────────────────────────────────────────
LIVER_ICD9 = [
    '5710','5711','5712','5713',
    '5715','5716','5718','5719',
    '5720','5722','5724','5728',
    '7891','5671','5723',
    '45620','45621','45680',
    '1550','1551','1552',
    '07054','07044','07032','07070',
]

# ── Option C decompensation codes ─────────────────────────────────────────
HIGH_ACUITY = {'5720','5722','5724','45620','5671'}

# ── Disease stage classification ──────────────────────────────────────────
ICD9_STAGES = {
    'NAFL':            ['5718','5719'],
    'NASH':            ['5718'],
    'FIBROSIS':        ['5715','57149'],
    'CIRRHOSIS':       ['5712','5715','5716'],
    'HEPATIC_FAILURE': ['5720','5722','5724'],
    'PORTAL_HTN':      ['5723'],
    'VARICES':         ['45620','45621','45680'],
    'ASCITES':         ['7891','78959'],
    'SBP':             ['5671'],
    'HCC':             ['1550','1551','1552'],
    'TRANSPLANT':      ['V427','99682'],
    'VIRAL_HEP':       ['07054','07044','07032','07070'],
    'ALCOHOLIC':       ['5710','5711','5712','5713'],
}

STAGE_HIERARCHY = {
    'NAFL':0,'NASH':1,'FIBROSIS':2,
    'CIRRHOSIS':3,'HEPATIC_FAILURE':3.5,'TRANSPLANT':4,
}

# ── Lab itemids ────────────────────────────────────────────────────────────
BILI_ITEMS    = [50885]
CREAT_ITEMS   = [50912]
INR_ITEMS     = [51237]
SODIUM_ITEMS  = [50983,50824]
AST_ITEMS     = [50878]
ALT_ITEMS     = [50861]
ALBUMIN_ITEMS = [50862]
PLATELET_ITEMS= [51265]
FERRITIN_ITEMS= [50924]
GGT_ITEMS     = [50927]
ALP_ITEMS     = [50863]
WBC_ITEMS     = [51301,51300]
HGB_ITEMS     = [51222,51221]
AMMONIA_ITEMS = [50867]

ALL_LAB_ITEMS = (BILI_ITEMS+CREAT_ITEMS+INR_ITEMS+SODIUM_ITEMS+
                 AST_ITEMS+ALT_ITEMS+ALBUMIN_ITEMS+PLATELET_ITEMS+
                 FERRITIN_ITEMS+GGT_ITEMS+ALP_ITEMS+
                 WBC_ITEMS+HGB_ITEMS+AMMONIA_ITEMS)

ITEMID_TO_LAB = {}
for _id in BILI_ITEMS:     ITEMID_TO_LAB[_id]='bilirubin'
for _id in CREAT_ITEMS:    ITEMID_TO_LAB[_id]='creatinine'
for _id in INR_ITEMS:      ITEMID_TO_LAB[_id]='inr'
for _id in SODIUM_ITEMS:   ITEMID_TO_LAB[_id]='sodium'
for _id in AST_ITEMS:      ITEMID_TO_LAB[_id]='ast'
for _id in ALT_ITEMS:      ITEMID_TO_LAB[_id]='alt'
for _id in ALBUMIN_ITEMS:  ITEMID_TO_LAB[_id]='albumin'
for _id in PLATELET_ITEMS: ITEMID_TO_LAB[_id]='platelets'
for _id in FERRITIN_ITEMS: ITEMID_TO_LAB[_id]='ferritin'
for _id in GGT_ITEMS:      ITEMID_TO_LAB[_id]='ggt'
for _id in ALP_ITEMS:      ITEMID_TO_LAB[_id]='alkaline_phosphatase'
for _id in WBC_ITEMS:      ITEMID_TO_LAB[_id]='wbc'
for _id in HGB_ITEMS:      ITEMID_TO_LAB[_id]='hemoglobin'
for _id in AMMONIA_ITEMS:  ITEMID_TO_LAB[_id]='ammonia'

# =============================================================================
# MELD COMPUTATION
# =============================================================================

def compute_meld(bili, creat, inr, on_dialysis=False):
    if any(v is None or (isinstance(v,float) and np.isnan(v))
           for v in [bili,creat,inr]):
        return None
    bili  = max(1.0, min(float(bili), 82.0))
    inr   = max(1.0, min(float(inr),  10.0))
    creat = 4.0 if on_dialysis else max(1.0, min(4.0, float(creat)))
    meld  = 3.78*np.log(bili) + 11.2*np.log(inr) + 9.57*np.log(creat) + 6.43
    return round(min(40.0, max(6.0, meld)), 1)

def compute_meld_na(bili, creat, inr, sodium, on_dialysis=False):
    meld = compute_meld(bili, creat, inr, on_dialysis)
    if meld is None or sodium is None or np.isnan(sodium): return None
    sodium  = max(125.0, min(137.0, float(sodium)))
    meld_na = meld + 1.32*(137-sodium) - (0.033*meld*(137-sodium))
    return round(min(40.0, max(6.0, meld_na)), 1)

def meld_severity(score):
    if score is None:  return 'Unknown'
    if score <= 9:     return 'Low'
    if score <= 19:    return 'Moderate'
    if score <= 29:    return 'High'
    if score <= 39:    return 'Very_High'
    return 'Maximum'

# =============================================================================
# OPTION C: COMPOSITE LABEL ASSIGNMENT
# =============================================================================

def assign_labels_option_c(df_meld, df_dx_flags):
    """
    Option C — Composite label for ICU-only MIMIC-III data.

    Positive (label=1) if ANY of:
      1. Peak MELD-Na >= 25 across all admissions
      2. MELD increased >= 5 points across any two consecutive admissions
      3. Had hepatorenal syndrome (has_hepatorenal=1)
      4. Had hepatic failure / encephalopathy (has_enceph=1)
      5. Had bleeding varices (has_varices_bleed=1)
      6. Died in hospital (hospital_expire_flag=1) with peak MELD >= 15

    Negative (label=0) if ALL of:
      - Peak MELD < 15 across all admissions
      - No decompensation markers (enceph, hepatorenal, bleeding varices)
      - Did NOT die with significant liver disease

    Why this matters for MIMIC-III:
      Unlike outpatient data, ICU patients are already sick.
      Requiring MELD >= 25 OR decompensation captures the full spectrum
      of patients who clinically would be evaluated for transplant.
    """
    df = df_meld.sort_values(['subject_id','admittime']).copy()

    # MELD trajectory features
    df['meld_prev']  = df.groupby('subject_id')['meld_na'].shift(1)
    df['meld_delta'] = df['meld_na'] - df['meld_prev']

    patient_max_delta = df.groupby('subject_id')['meld_delta'].max()
    patient_peak_meld = df.groupby('subject_id')['meld_na'].max()
    patient_max_expire= df.groupby('subject_id')['hospital_expire_flag'].max()

    df['peak_meld']       = df['subject_id'].map(patient_peak_meld)
    df['max_meld_delta']  = df['subject_id'].map(patient_max_delta)
    df['ever_expired']    = df['subject_id'].map(patient_max_expire)

    # Merge diagnosis flags
    pat_dx = df_dx_flags.groupby('subject_id').agg(
        has_hepatorenal  =('has_hepatorenal','max'),
        has_enceph       =('has_enceph','max'),
        has_varices_bleed=('has_varices_bleed','max'),
        has_sbp          =('has_sbp','max'),
        has_hcc          =('has_hcc','max'),
    ).reset_index()

    df = df.merge(pat_dx, on='subject_id', how='left')
    for col in ['has_hepatorenal','has_enceph','has_varices_bleed',
                'has_sbp','has_hcc']:
        df[col] = df[col].fillna(0)

    # Option C composite label
    pos_mask = (
        (df['peak_meld'] >= 25)                        |  # criterion 1
        (df['max_meld_delta'] >= 5)                    |  # criterion 2
        (df['has_hepatorenal'] == 1)                   |  # criterion 3
        (df['has_enceph'] == 1)                        |  # criterion 4
        (df['has_varices_bleed'] == 1)                 |  # criterion 5
        (                                                  # criterion 6
            (df['ever_expired'] == 1) &
            (df['peak_meld'] >= 15)
        )
    )
    df['label'] = pos_mask.astype(int)

    # Use last admission per patient as prediction row
    df_last = df.sort_values('admittime').groupby('subject_id').last().reset_index()

    return df, df_last

# =============================================================================
# SQL QUERIES
# =============================================================================

COHORT_QUERY = """
WITH liver_admissions AS (
    SELECT DISTINCT
        a.subject_id, a.hadm_id,
        a.admittime, a.dischtime,
        a.admission_type, a.hospital_expire_flag,
        p.gender, p.dob, p.dod, p.expire_flag,
        EXTRACT(YEAR FROM AGE(a.admittime, p.dob)) AS age_at_admit
    FROM public.admissions a
    JOIN public.patients p ON a.subject_id = p.subject_id
    WHERE a.subject_id IN (
        SELECT DISTINCT subject_id FROM public.diagnoses_icd
        WHERE icd9_code = ANY(%(codes)s)
    )
    AND a.admittime IS NOT NULL
    AND a.dischtime IS NOT NULL
),
icu_stays AS (
    SELECT hadm_id, MAX(los) AS max_icu_los
    FROM public.icustays GROUP BY hadm_id
)
SELECT la.*, COALESCE(icu.max_icu_los,0) AS max_icu_los,
       icu.max_icu_los IS NOT NULL AS had_icu_stay
FROM liver_admissions la
LEFT JOIN icu_stays icu ON la.hadm_id = icu.hadm_id
WHERE la.age_at_admit BETWEEN 18 AND 100
ORDER BY la.subject_id, la.admittime;
"""

LAB_QUERY = """
SELECT l.subject_id, l.hadm_id, l.itemid, l.valuenum, l.charttime
FROM public.labevents l
WHERE l.subject_id = ANY(%(pids)s)
  AND l.itemid = ANY(%(items)s)
  AND l.valuenum IS NOT NULL AND l.valuenum > 0
ORDER BY l.subject_id, l.hadm_id, l.charttime;
"""

DX_QUERY = """
SELECT d.subject_id, d.hadm_id, d.icd9_code, d.seq_num, di.short_title
FROM public.diagnoses_icd d
JOIN public.d_icd_diagnoses di ON d.icd9_code = di.icd9_code
WHERE d.subject_id = ANY(%(pids)s)
ORDER BY d.subject_id, d.hadm_id, d.seq_num;
"""

MED_QUERY = """
SELECT subject_id, hadm_id, drug, drug_name_generic, route, startdate, enddate
FROM public.prescriptions
WHERE subject_id = ANY(%(pids)s) AND drug IS NOT NULL
ORDER BY subject_id, hadm_id, startdate;
"""

# =============================================================================
# FEATURE COMPUTATION
# =============================================================================

def get_all_labs(pg, subject_ids):
    cur = pg.cursor()
    cur.execute(LAB_QUERY, {'pids':list(subject_ids),'items':ALL_LAB_ITEMS})
    rows = cur.fetchall()
    cur.close()
    df = pd.DataFrame(rows, columns=['subject_id','hadm_id','itemid','valuenum','charttime'])
    df['charttime'] = pd.to_datetime(df['charttime'])
    df['lab_name']  = df['itemid'].map(ITEMID_TO_LAB)
    return df


def compute_meld_per_admission(df_cohort, df_labs):
    results = []
    for _, row in df_cohort.iterrows():
        pid   = row['subject_id']
        hid   = row['hadm_id']
        admit = pd.to_datetime(row['admittime'])
        disch = pd.to_datetime(row['dischtime'])

        mask = ((df_labs['subject_id']==pid) & (df_labs['hadm_id']==hid) &
                (df_labs['charttime']>=admit) & (df_labs['charttime']<=disch))
        adm  = df_labs[mask]

        def latest(items):
            sub = adm[adm['itemid'].isin(items)]
            return sub.sort_values('charttime',ascending=False)['valuenum'].iloc[0] if len(sub) else None

        bili   = latest(BILI_ITEMS)
        creat  = latest(CREAT_ITEMS)
        inr    = latest(INR_ITEMS)
        sodium = latest(SODIUM_ITEMS)

        results.append({
            'subject_id':  pid, 'hadm_id': hid,
            'admittime':   admit, 'dischtime': disch,
            'gender':      row['gender'],
            'age_at_admit':row['age_at_admit'],
            'admission_type':row['admission_type'],
            'hospital_expire_flag':row['hospital_expire_flag'],
            'had_icu_stay':row['had_icu_stay'],
            'max_icu_los': row['max_icu_los'],
            'bilirubin':   bili, 'creatinine': creat,
            'inr':         inr,  'sodium':     sodium,
            'meld':        compute_meld(bili,creat,inr),
            'meld_na':     compute_meld_na(bili,creat,inr,sodium),
            'severity':    meld_severity(compute_meld_na(bili,creat,inr,sodium)),
            'ast':         latest(AST_ITEMS),
            'alt':         latest(ALT_ITEMS),
            'albumin':     latest(ALBUMIN_ITEMS),
            'platelets':   latest(PLATELET_ITEMS),
            'ferritin':    latest(FERRITIN_ITEMS),
            'ggt':         latest(GGT_ITEMS),
            'alkaline_phosphatase': latest(ALP_ITEMS),
            'wbc':         latest(WBC_ITEMS),
            'hemoglobin':  latest(HGB_ITEMS),
            'ammonia':     latest(AMMONIA_ITEMS),
        })
    return pd.DataFrame(results)


def compute_ghandian_lab_summary(df_labs, df_cohort):
    """Ghandian et al. prior/current lab summary statistics."""
    print("  Computing Ghandian-style lab summary statistics...")
    target_labs = ['bilirubin','creatinine','inr','sodium',
                   'ast','alt','albumin','platelets','ferritin','ggt']
    last_admit = df_cohort.groupby('subject_id')['admittime'].max().to_dict()

    records = []
    for subject_id, patient_labs in df_labs.groupby('subject_id'):
        ref_date = pd.to_datetime(last_admit.get(subject_id))
        if ref_date is None: continue

        features = {'subject_id': subject_id}
        current_cutoff = ref_date - timedelta(hours=24)
        current_labs   = patient_labs[patient_labs['charttime'] > current_cutoff]
        prior_labs     = patient_labs[
            (patient_labs['charttime'] <= current_cutoff) &
            (patient_labs['charttime'] >= ref_date - timedelta(days=365))
        ]

        for lab in target_labs:
            prior  = prior_labs[prior_labs['lab_name']==lab]['valuenum']
            cur    = current_labs[current_labs['lab_name']==lab]['valuenum']
            if len(prior) > 0:
                features[f'prior_{lab}_mean']  = prior.mean()
                features[f'prior_{lab}_std']   = prior.std() if len(prior)>1 else 0
                features[f'prior_{lab}_min']   = prior.min()
                features[f'prior_{lab}_max']   = prior.max()
                features[f'prior_{lab}_count'] = len(prior)
            else:
                for s in ['mean','std','min','max','count']:
                    features[f'prior_{lab}_{s}'] = None
            features[f'current_{lab}_mean'] = cur.mean() if len(cur)>0 else None
            features[f'current_{lab}_max']  = cur.max()  if len(cur)>0 else None

        records.append(features)

    result = pd.DataFrame(records)
    print(f"  ✓ Lab summary for {len(result):,} patients, "
          f"{len(result.columns)-1} features")
    return result


def compute_meld_trajectory(df_admissions):
    """MELD slope, acceleration, variability, consecutive increases."""
    print("  Computing MELD trajectory features...")
    df = df_admissions.sort_values(['subject_id','admittime']).copy()
    df['admittime'] = pd.to_datetime(df['admittime'])

    records = []
    for subject_id, group in df.groupby('subject_id'):
        group = group.sort_values('admittime').reset_index(drop=True)
        meld_s = group[group['meld_na'].notna()][['admittime','meld_na']]
        if len(meld_s) == 0: continue

        ref_date = group['admittime'].max()
        feats = {'subject_id': subject_id,
                 'meld_current': meld_s['meld_na'].iloc[-1]}

        for days_back, label in [(90,'90d'),(180,'180d'),(365,'1yr')]:
            cutoff = ref_date - timedelta(days=days_back)
            hist = meld_s[meld_s['admittime'] <= cutoff]
            feats[f'meld_{label}_ago'] = hist['meld_na'].iloc[-1] if len(hist) else None

        for days_win, label in [(90,'90d'),(180,'180d'),(365,'1yr')]:
            cutoff = ref_date - timedelta(days=days_win)
            win = meld_s[meld_s['admittime'] >= cutoff]
            if len(win) >= 2:
                days_e = (win['admittime'] - win['admittime'].iloc[0]).dt.days
                if days_e.iloc[-1] > 0:
                    slope = np.polyfit(days_e.values.astype(float),
                                       win['meld_na'].values, 1)[0]
                    feats[f'meld_slope_{label}'] = slope * 30
                else:
                    feats[f'meld_slope_{label}'] = 0.0
            else:
                feats[f'meld_slope_{label}'] = None

        s90  = feats.get('meld_slope_90d')
        s180 = feats.get('meld_slope_180d')
        feats['meld_acceleration'] = (s90-s180 if s90 is not None and s180 is not None else None)

        for days_win, label in [(90,'90d'),(180,'180d')]:
            cutoff = ref_date - timedelta(days=days_win)
            win = meld_s[meld_s['admittime'] >= cutoff]
            feats[f'meld_max_{label}'] = win['meld_na'].max() if len(win) else None
            feats[f'meld_min_{label}'] = win['meld_na'].min() if len(win) else None

        feats['meld_std'] = meld_s['meld_na'].tail(10).std() if len(meld_s)>=3 else 0.0

        if len(meld_s) >= 2:
            diffs = meld_s['meld_na'].diff().dropna()
            consec = 0
            for d in reversed(diffs.values):
                if d > 0: consec += 1
                else: break
            feats['meld_consecutive_increases'] = consec
            increases = meld_s[diffs.values > 0] if len(diffs) else pd.DataFrame()
            feats['days_since_meld_increase'] = (
                (ref_date - increases['admittime'].iloc[-1]).days
                if len(increases) > 0 else 9999
            )
        else:
            feats['meld_consecutive_increases'] = 0
            feats['days_since_meld_increase']   = None

        feats['meld_measurement_count'] = len(meld_s)
        records.append(feats)

    result = pd.DataFrame(records)
    print(f"  ✓ MELD trajectory for {len(result):,} patients")
    return result


def compute_medication_trajectory(df_meds_raw, df_admissions):
    """Medication escalation, max treatment, treatment response."""
    print("  Computing medication trajectory features...")
    df_adm = df_admissions.copy()
    df_adm['admittime'] = pd.to_datetime(df_adm['admittime'])

    liver_drugs = ['lactulose','rifaximin','xifaxan','spironolactone','aldactone',
                   'furosemide','lasix','nadolol','propranolol','albumin',
                   'norfloxacin','ciprofloxacin','metformin','pioglitazone',
                   'atorvastatin','rosuvastatin','simvastatin']

    records = []
    for subject_id, pat_meds in df_meds_raw.groupby('subject_id'):
        pat_adm = df_adm[df_adm['subject_id']==subject_id]
        if len(pat_adm) == 0: continue

        ref_date = pat_adm['admittime'].max()
        feats    = {'subject_id': subject_id}

        last_hadm = pat_adm.sort_values('admittime').iloc[-1]['hadm_id']
        recent_meds = pat_meds[pat_meds['hadm_id']==last_hadm]
        recent_drugs = recent_meds['drug'].str.lower().fillna('')

        liver_count = sum(1 for d in recent_drugs if any(ld in d for ld in liver_drugs))
        feats['total_liver_meds_current'] = liver_count

        six_mo_cutoff = ref_date - timedelta(days=180)
        old_adm = pat_adm[(pat_adm['admittime'] <= six_mo_cutoff) &
                          (pat_adm['admittime'] >= six_mo_cutoff - timedelta(days=90))]
        if len(old_adm) > 0:
            old_meds  = pat_meds[pat_meds['hadm_id'].isin(old_adm['hadm_id'])]
            old_drugs = old_meds['drug'].str.lower().fillna('')
            feats['total_liver_meds_6mo_ago'] = sum(
                1 for d in old_drugs if any(ld in d for ld in liver_drugs)
            )
        else:
            feats['total_liver_meds_6mo_ago'] = 0

        feats['medication_escalation_count'] = max(
            0, feats['total_liver_meds_current'] - feats['total_liver_meds_6mo_ago']
        )
        feats['max_treatment_reached'] = int(feats['total_liver_meds_current'] >= 3)

        meld_vals = pat_adm[pat_adm['meld_na'].notna()].sort_values('admittime') \
            if 'meld_na' in pat_adm.columns else pd.DataFrame()
        if len(meld_vals) >= 2:
            feats['treatment_response']          = int(meld_vals['meld_na'].iloc[-1] < meld_vals['meld_na'].iloc[-2])
            feats['meld_change_after_treatment'] = meld_vals['meld_na'].iloc[-1] - meld_vals['meld_na'].iloc[-2]
        else:
            feats['treatment_response']          = None
            feats['meld_change_after_treatment'] = None

        for drug_key in ['lactulose','rifaximin','spironolactone','furosemide',
                         'albumin','nadolol','metformin','pioglitazone']:
            feats[f'on_{drug_key}'] = int(any(drug_key in d for d in recent_drugs))

        feats['on_encephalopathy_tx']  = int(feats.get('on_lactulose',0) or feats.get('on_rifaximin',0))
        feats['on_ascites_tx']         = int(feats.get('on_spironolactone',0) or feats.get('on_furosemide',0))
        feats['on_metabolic_modifier'] = int(feats.get('on_metformin',0) or
                                             any('statin' in d for d in recent_drugs))
        records.append(feats)

    result = pd.DataFrame(records)
    print(f"  ✓ Medication trajectory for {len(result):,} patients")
    return result


def compute_disease_stage(df_all_dx):
    """ICD-9 based disease stage per patient."""
    print("  Computing disease stage features...")
    records = []
    for subject_id, pat_dx in df_all_dx.groupby('subject_id'):
        all_codes = set(pat_dx['icd9_code'].values)
        feats = {'subject_id': subject_id}
        max_level, max_stage = -1, 'UNKNOWN'
        for stage, level in STAGE_HIERARCHY.items():
            stage_codes = ICD9_STAGES.get(stage, [])
            if any(code in all_codes for code in stage_codes):
                if level > max_level:
                    max_level, max_stage = level, stage
        has_failure = any(c in all_codes for c in ICD9_STAGES.get('HEPATIC_FAILURE',[]))
        has_complications = any(
            any(c in all_codes for c in ICD9_STAGES.get(comp,[]))
            for comp in ['PORTAL_HTN','VARICES','ASCITES','SBP','HCC']
        )
        if has_failure and max_level < 3.5:
            max_level, max_stage = 3.5, 'HEPATIC_FAILURE'
        feats['current_stage']       = max_stage
        feats['current_stage_level'] = max_level
        feats['has_decompensation']  = int(has_complications or has_failure)
        feats['etiology_alcoholic']  = int(any(c in all_codes for c in ICD9_STAGES.get('ALCOHOLIC',[])))
        feats['etiology_viral']      = int(any(c in all_codes for c in ICD9_STAGES.get('VIRAL_HEP',[])))
        feats['etiology_nafld']      = int(any(c in all_codes for c in ICD9_STAGES.get('NAFL',[])))
        records.append(feats)
    result = pd.DataFrame(records)
    print(f"  ✓ Disease stage for {len(result):,} patients")
    return result


def compute_utilization_features(df_admissions):
    """Visit counts, acceleration, ED visits."""
    print("  Computing utilization features...")
    df = df_admissions.copy()
    df['admittime'] = pd.to_datetime(df['admittime'])
    df = df.sort_values(['subject_id','admittime'])

    records = []
    for subject_id, group in df.groupby('subject_id'):
        group = group.sort_values('admittime').reset_index(drop=True)
        ref_date, n_adm = group['admittime'].max(), len(group)
        feats = {'subject_id': subject_id}

        for days, label in [(90,'90d'),(180,'6mo'),(365,'1yr')]:
            cutoff = ref_date - timedelta(days=days)
            feats[f'visit_count_{label}'] = len(group[group['admittime']>=cutoff])

        feats['ed_visits_90d'] = len(group[
            (group['admittime'] >= ref_date - timedelta(days=90)) &
            (group['admission_type'].str.upper().str.contains('EMERG', na=False))
        ])
        feats['icu_ever'] = int(group['had_icu_stay'].max()) if 'had_icu_stay' in group.columns else 0

        if n_adm > 2:
            gaps = [(group['admittime'].iloc[i]-group['admittime'].iloc[i-1]).days
                    for i in range(1, n_adm)]
            feats['visit_acceleration']       = (np.mean(gaps[:2])-np.mean(gaps[-2:])) if len(gaps)>=3 else 0.0
            feats['avg_days_between_admits']  = np.mean(gaps)
            feats['min_visit_interval']       = min(gaps)
        else:
            feats['visit_acceleration']      = 0.0
            feats['avg_days_between_admits'] = 0.0
            feats['min_visit_interval']      = None

        feats['n_admissions']      = n_adm
        feats['days_first_to_last'] = (group['admittime'].iloc[-1]-group['admittime'].iloc[0]).days
        records.append(feats)

    result = pd.DataFrame(records)
    print(f"  ✓ Utilization for {len(result):,} patients")
    return result


def get_dx_features(pg, subject_ids):
    cur = pg.cursor()
    cur.execute(DX_QUERY, {'pids':list(subject_ids)})
    rows = cur.fetchall()
    cur.close()
    df = pd.DataFrame(rows, columns=['subject_id','hadm_id','icd9_code','seq_num','short_title'])

    features = df.groupby(['subject_id','hadm_id']).apply(lambda g: pd.Series({
        'n_diagnoses':       len(g),
        'has_enceph':        int(g['icd9_code'].isin(['5722']).any()),
        'has_hepatorenal':   int(g['icd9_code'].isin(['5724']).any()),
        'has_varices_bleed': int(g['icd9_code'].isin(['45620']).any()),
        'has_sbp':           int(g['icd9_code'].isin(['5671']).any()),
        'has_hcc':           int(g['icd9_code'].isin(['1550','1551','1552']).any()),
        'has_high_acuity':   int(g['icd9_code'].isin(HIGH_ACUITY).any()),
        'has_cirrhosis':     int(g['icd9_code'].isin(['5715','5712','5716']).any()),
        'has_portal_htn':    int(g['icd9_code'].isin(['5723']).any()),
        'has_hep_c':         int(g['icd9_code'].isin(['07054','07070']).any()),
        'has_alcoholic':     int(g['icd9_code'].isin(['5710','5711','5712','5713']).any()),
    })).reset_index()
    return features, df


def get_med_features(pg, subject_ids):
    cur = pg.cursor()
    cur.execute(MED_QUERY, {'pids':list(subject_ids)})
    rows = cur.fetchall()
    cur.close()
    df = pd.DataFrame(rows, columns=['subject_id','hadm_id','drug','drug_name_generic',
                                     'route','startdate','enddate'])
    drug_lower = df['drug'].str.lower().fillna('')
    df['is_lactulose']      = drug_lower.str.contains('lactulose', na=False).astype(int)
    df['is_rifaximin']      = drug_lower.str.contains('rifaximin', na=False).astype(int)
    df['is_spironolactone'] = drug_lower.str.contains('spironolactone', na=False).astype(int)
    df['is_furosemide']     = drug_lower.str.contains('furosemide', na=False).astype(int)
    df['is_albumin']        = drug_lower.str.contains('albumin', na=False).astype(int)
    df['is_nadolol']        = drug_lower.str.contains('nadolol|propranolol', na=False).astype(int)
    df['is_norfloxacin']    = drug_lower.str.contains('norfloxacin|ciprofloxacin', na=False).astype(int)

    features = df.groupby(['subject_id','hadm_id']).agg(
        n_medications     =('drug','count'),
        n_unique_drugs    =('drug','nunique'),
        on_lactulose      =('is_lactulose','max'),
        on_rifaximin      =('is_rifaximin','max'),
        on_spironolactone =('is_spironolactone','max'),
        on_furosemide     =('is_furosemide','max'),
        on_albumin        =('is_albumin','max'),
        on_nadolol        =('is_nadolol','max'),
        on_norfloxacin    =('is_norfloxacin','max'),
    ).reset_index()
    return features, df

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  Liver Disease Progression — Postgres Path")
    print("  Label Strategy: OPTION C (Composite — ICU-appropriate)")
    print("  Ghandian et al. (2022) framework + MELD trajectory")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("""
  Option C Label Criteria (positive if ANY):
    1. Peak MELD-Na >= 25
    2. MELD increased >= 5 points across admissions
    3. Hepatorenal syndrome (ICD-9 5724)
    4. Hepatic failure / encephalopathy (ICD-9 5722)
    5. Bleeding varices (ICD-9 45620)
    6. Died in hospital with peak MELD >= 15
    """)

    pg = psycopg2.connect(**PG_CONFIG)

    # [1] Identify liver patients
    print("\n[1] Identifying liver patients...")
    t0 = time.time()
    cur = pg.cursor()
    cur.execute(COHORT_QUERY, {'codes': LIVER_ICD9})
    rows = cur.fetchall()
    cur.close()
    cols = ['subject_id','hadm_id','admittime','dischtime',
            'admission_type','hospital_expire_flag','gender',
            'dob','dod','expire_flag','age_at_admit','max_icu_los','had_icu_stay']
    df_cohort = pd.DataFrame(rows, columns=cols)
    print(f"  ✓ {df_cohort['subject_id'].nunique():,} patients, "
          f"{len(df_cohort):,} admissions ({time.time()-t0:.1f}s)")
    subject_ids = df_cohort['subject_id'].unique().tolist()

    # [2] All labs
    print("\n[2] Fetching ALL labs (MELD + Ghandian)...")
    t0 = time.time()
    df_labs = get_all_labs(pg, subject_ids)
    print(f"  ✓ {len(df_labs):,} lab results ({time.time()-t0:.1f}s)")

    # [3] MELD per admission
    print("\n[3] Computing MELD + Ghandian labs per admission...")
    t0 = time.time()
    df_meld = compute_meld_per_admission(df_cohort, df_labs)
    print(f"  ✓ {df_meld['meld_na'].notna().sum():,} admissions scored ({time.time()-t0:.1f}s)")

    # [4] Diagnosis features
    print("\n[4] Extracting diagnosis features...")
    df_dx, df_all_dx = get_dx_features(pg, subject_ids)

    # [5] Medication features
    print("\n[5] Extracting medication features...")
    df_meds, df_meds_raw = get_med_features(pg, subject_ids)

    # [6] OPTION C label assignment
    print("\n[6] Assigning OPTION C composite labels...")
    df_meld_with_dx = df_meld.merge(
        df_dx[['subject_id','hadm_id','has_hepatorenal','has_enceph',
               'has_varices_bleed','has_sbp','has_hcc']],
        on=['subject_id','hadm_id'], how='left'
    )
    df_all, df_last = assign_labels_option_c(df_meld_with_dx, df_dx)

    n_pos = (df_last['label']==1).sum()
    n_neg = (df_last['label']==0).sum()
    print(f"  ✓ Positive (transplant-level acuity): {n_pos:,} ({n_pos/(n_pos+n_neg)*100:.1f}%)")
    print(f"  ✓ Negative (lower acuity):            {n_neg:,} ({n_neg/(n_pos+n_neg)*100:.1f}%)")

    # Breakdown of what triggered positive
    print(f"\n  Positive triggers (may overlap):")
    print(f"    Peak MELD >= 25:      {(df_last['peak_meld']>=25).sum():,}")
    print(f"    MELD delta >= 5:      {(df_last['max_meld_delta']>=5).sum():,}")
    print(f"    Hepatorenal:          {(df_last.get('has_hepatorenal',pd.Series([0]*len(df_last)))==1).sum():,}")
    print(f"    Encephalopathy:       {(df_last.get('has_enceph',pd.Series([0]*len(df_last)))==1).sum():,}")
    print(f"    Bleeding varices:     {(df_last.get('has_varices_bleed',pd.Series([0]*len(df_last)))==1).sum():,}")

    # [7] Merge base features
    print("\n[7] Merging base feature sets...")
    df_final = df_last.merge(df_dx,  on=['subject_id','hadm_id'], how='left')
    df_final = df_final.merge(df_meds, on=['subject_id','hadm_id'], how='left')
    df_all_m = df_all.merge(df_dx,  on=['subject_id','hadm_id'], how='left')
    df_all_m = df_all_m.merge(df_meds, on=['subject_id','hadm_id'], how='left')
    bool_cols = [c for c in df_final.columns if c.startswith(('has_','on_','n_'))]
    df_final[bool_cols] = df_final[bool_cols].fillna(0)
    valid_bool = [c for c in bool_cols if c in df_all_m.columns]
    df_all_m[valid_bool] = df_all_m[valid_bool].fillna(0)

    # [8] Ghandian lab summary
    print("\n[8] Ghandian lab summary statistics...")
    df_lab_summary = compute_ghandian_lab_summary(df_labs, df_cohort)

    # [9] MELD trajectory
    print("\n[9] MELD trajectory features...")
    df_meld_traj = compute_meld_trajectory(df_all_m)

    # [10] Medication trajectory
    print("\n[10] Medication trajectory features...")
    df_med_traj = compute_medication_trajectory(df_meds_raw, df_all_m)

    # [11] Disease stage
    print("\n[11] Disease stage features...")
    df_stage = compute_disease_stage(df_all_dx)

    # [12] Utilization
    print("\n[12] Utilization features...")
    df_util = compute_utilization_features(df_all_m)

    # [13] Merge ALL features
    print("\n[13] Merging ALL feature sets...")
    df_complete = df_final.copy()
    for feat_df, name in [
        (df_lab_summary, 'ghandian_labs'),
        (df_meld_traj,   'meld_trajectory'),
        (df_med_traj,    'medication_trajectory'),
        (df_stage,       'disease_stage'),
        (df_util,        'utilization'),
    ]:
        if feat_df is not None and len(feat_df) > 0:
            existing  = set(df_complete.columns)
            new_cols  = ['subject_id'] + [c for c in feat_df.columns
                                           if c != 'subject_id' and c not in existing]
            if len(new_cols) > 1:
                df_complete = df_complete.merge(feat_df[new_cols], on='subject_id', how='left')
                print(f"  + {name}: {len(new_cols)-1} features")
    print(f"  Total features: {len(df_complete.columns)}")

    # [14] Save
    print("\n[14] Saving results...")
    os.makedirs('/project/liver', exist_ok=True)
    df_complete.to_csv('/project/liver/liver_cohort_pg.csv', index=False)
    df_all_m.to_csv('/project/liver/liver_admissions_pg.csv', index=False)
    df_lab_summary.to_csv('/project/liver/liver_lab_summary_pg.csv', index=False)
    df_meld_traj.to_csv('/project/liver/liver_meld_trajectory_pg.csv', index=False)

    print(f"\n{'='*70}")
    print("  Summary")
    print(f"{'='*70}")
    print(f"  Liver patients:   {df_complete['subject_id'].nunique():,}")
    print(f"  Total admissions: {len(df_all_m):,}")
    print(f"  MELD-scoreable:   {df_all_m['meld_na'].notna().sum():,}")
    print(f"  Label=1 (Option C): {n_pos:,} ({n_pos/(n_pos+n_neg)*100:.1f}%)")
    print(f"  Label=0:            {n_neg:,}")
    print(f"  Total features:   {len(df_complete.columns)}")
    print(f"  Mean MELD-Na:     {df_complete['meld_na'].mean():.1f}")
    pg.close()
    print("\n  ✓ Done")

if __name__ == "__main__":
    main()

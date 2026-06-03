"""
label_option_c.py
Option C composite label — shared between liver_pg.py and liver_ydb.py

OPTION C LABEL DEFINITION (ICU-appropriate for MIMIC-III):

Positive (label=1) if ANY criterion met:
  1. Peak MELD-Na >= 25 across all admissions
  2. MELD increased >= 5 points across any consecutive admissions
  3. Hepatorenal syndrome (ICD-9 5724)
  4. Hepatic failure / encephalopathy (ICD-9 5722)
  5. Bleeding varices (ICD-9 45620)
  6. Died in hospital (hospital_expire_flag=1) AND peak MELD >= 15

Negative (label=0) if ALL:
  - Peak MELD-Na < 15 across all admissions
  - No decompensation markers (enceph, hepatorenal, bleeding varices)
  - Did NOT die with significant liver disease

Clinical rationale for MIMIC-III:
  All MIMIC patients are ICU-admitted — they are already sick.
  The original Option A (trajectory only) misses patients who:
    - Present with high MELD on first admission (no trajectory yet)
    - Have severe decompensation markers regardless of MELD trend
    - Died — the most severe outcome, suggesting transplant was needed
  Option C is a superset that captures all clinically relevant paths
  to transplant evaluation in an ICU population.

Expected positive rate: 55-65% (higher than Option A's ~40%)
This is clinically appropriate — in an ICU liver population,
most patients have transplant-level disease severity.
"""

import numpy as np
import pandas as pd


def assign_labels_option_c(df_admissions):
    """
    Apply Option C composite label to admission-level DataFrame.

    Required columns in df_admissions:
      subject_id, admittime, meld_na, hospital_expire_flag,
      has_hepatorenal, has_enceph, has_varices_bleed

    Returns:
      df_all:  admission-level DataFrame with label column
      df_last: patient-level DataFrame (last admission = prediction row)
    """
    df = df_admissions.sort_values(['subject_id','admittime']).copy()

    # ── MELD trajectory features ───────────────────────────────────────────
    df['meld_prev']  = df.groupby('subject_id')['meld_na'].shift(1)
    df['meld_delta'] = df['meld_na'] - df['meld_prev']
    df['days_since_prev'] = (
        df['admittime'] -
        df.groupby('subject_id')['admittime'].shift(1)
    ).dt.days

    # Patient-level aggregates
    patient_max_delta = df.groupby('subject_id')['meld_delta'].max()
    patient_peak_meld = df.groupby('subject_id')['meld_na'].max()
    patient_max_expire= df.groupby('subject_id')['hospital_expire_flag'].max()

    df['peak_meld']      = df['subject_id'].map(patient_peak_meld)
    df['max_meld_delta'] = df['subject_id'].map(patient_max_delta)
    df['ever_expired']   = df['subject_id'].map(patient_max_expire)

    # Patient-level decompensation flags (max across all admissions)
    for flag in ['has_hepatorenal','has_enceph','has_varices_bleed']:
        if flag in df.columns:
            pat_max = df.groupby('subject_id')[flag].max()
            df[f'ever_{flag}'] = df['subject_id'].map(pat_max)
        else:
            df[f'ever_{flag}'] = 0

    # ── OPTION C Composite Label ───────────────────────────────────────────
    #
    # Criterion 1: Peak MELD-Na >= 25
    crit_1 = (df['peak_meld'] >= 25)

    # Criterion 2: MELD increased >= 5 points (trajectory)
    crit_2 = (df['max_meld_delta'] >= 5)

    # Criterion 3: Hepatorenal syndrome ever
    crit_3 = (df['ever_has_hepatorenal'] == 1)

    # Criterion 4: Hepatic failure / encephalopathy ever
    crit_4 = (df['ever_has_enceph'] == 1)

    # Criterion 5: Bleeding varices ever
    crit_5 = (df['ever_has_varices_bleed'] == 1)

    # Criterion 6: In-hospital death with significant liver disease
    crit_6 = ((df['ever_expired'] == 1) & (df['peak_meld'] >= 15))

    df['label'] = (crit_1 | crit_2 | crit_3 | crit_4 | crit_5 | crit_6).astype(int)

    # Last admission per patient = prediction row
    df_last = df.sort_values('admittime').groupby('subject_id').last().reset_index()

    return df, df_last


def print_label_summary(df_last, df_all):
    """Print breakdown of what triggered positive labels."""
    n_pos = (df_last['label']==1).sum()
    n_neg = (df_last['label']==0).sum()
    total = n_pos + n_neg

    print(f"\n  ── Option C Label Summary ────────────────────────────")
    print(f"  Total patients:              {total:,}")
    print(f"  Positive (transplant-level): {n_pos:,} ({n_pos/total*100:.1f}%)")
    print(f"  Negative (lower acuity):     {n_neg:,} ({n_neg/total*100:.1f}%)")

    print(f"\n  Positive triggers (overlap expected):")
    triggers = [
        ('Peak MELD-Na >= 25',      'peak_meld',         lambda s: s>=25),
        ('MELD delta >= 5',          'max_meld_delta',    lambda s: s>=5),
        ('Hepatorenal syndrome',     'ever_has_hepatorenal', lambda s: s==1),
        ('Encephalopathy/failure',   'ever_has_enceph',   lambda s: s==1),
        ('Bleeding varices',         'ever_has_varices_bleed', lambda s: s==1),
        ('Died + peak MELD >= 15',   None,                None),
    ]
    for name, col, fn in triggers:
        if col and col in df_last.columns and fn:
            count = fn(df_last[col]).sum()
            print(f"    {name:<35} {count:>6,} ({count/total*100:.1f}%)")
        elif col is None:
            count = ((df_last.get('ever_expired',pd.Series([0]*len(df_last)))==1) &
                     (df_last['peak_meld']>=15)).sum()
            print(f"    {name:<35} {count:>6,} ({count/total*100:.1f}%)")

    print(f"\n  MELD distribution at last admission:")
    if 'meld_na' in df_last.columns:
        meld_valid = df_last['meld_na'].dropna()
        print(f"    Mean:   {meld_valid.mean():.1f}")
        print(f"    Median: {meld_valid.median():.1f}")
        print(f"    >= 15:  {(meld_valid>=15).sum():,} ({(meld_valid>=15).sum()/len(meld_valid)*100:.1f}%)")
        print(f"    >= 25:  {(meld_valid>=25).sum():,} ({(meld_valid>=25).sum()/len(meld_valid)*100:.1f}%)")
        print(f"    >= 35:  {(meld_valid>=35).sum():,} ({(meld_valid>=35).sum()/len(meld_valid)*100:.1f}%)")

    print(f"\n  Comparison with other label strategies:")
    print(f"    Option A (trajectory only): ~40% positive (MELD delta or peak>=25)")
    print(f"    Option B (30d readmission): ~26.6% positive (from readmission benchmark)")
    print(f"    Option C (composite):       {n_pos/total*100:.1f}% positive (ICU-appropriate)")
    print(f"    → Higher rate expected: ICU liver patients are severely ill")

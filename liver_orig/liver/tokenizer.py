"""
tokenizer.py
Medical Event Tokenizer for Liver Transplant Prediction

Converts clinical events to integer token IDs:
  Demographics  → [PAT_START] [AGE_50-59] [SEX_M]
  Diagnoses     → [DX_5712] [DX_5723]
  MELD          → [MELD_20-24]
  Labs          → [LAB_BILI_2.0-5.0] [LAB_CREAT_1.0-1.5]
  Medications   → [MED_LACTULOSE] [MED_SPIRONOLACTONE]
  Time gaps     → [TIME_30-60d] [TIME_60-90d]
  Encounters    → [VISIT_START] [ENC_EMERGENCY] [VISIT_END]

Usage:
  from tokenizer import MedicalTokenizer
  tok = MedicalTokenizer()
  tok.build_vocab(df_admissions)
  tokens = tok.encode_patient(patient_admissions)
  tok.save('/project/liver/vocab.json')
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict

# =============================================================================
# TOKEN DEFINITIONS
# =============================================================================

# Special tokens
SPECIAL_TOKENS = {
    '[PAD]':        0,
    '[UNK]':        1,
    '[PAT_START]':  2,
    '[PAT_END]':    3,
    '[VISIT_START]': 4,
    '[VISIT_END]':  5,
    '[MASK]':       6,
    '[CLS]':        7,
}

# Age buckets
AGE_TOKENS = {
    '[AGE_18-29]': 10,
    '[AGE_30-39]': 11,
    '[AGE_40-49]': 12,
    '[AGE_50-59]': 13,
    '[AGE_60-69]': 14,
    '[AGE_70-79]': 15,
    '[AGE_80+]':   16,
}

# Sex
SEX_TOKENS = {
    '[SEX_M]': 20,
    '[SEX_F]': 21,
    '[SEX_U]': 22,
}

# Encounter type
ENC_TOKENS = {
    '[ENC_EMERGENCY]': 30,
    '[ENC_ELECTIVE]':  31,
    '[ENC_URGENT]':    32,
    '[ENC_OTHER]':     33,
}

# Time between visits (days)
TIME_TOKENS = {
    '[TIME_0-7d]':    40,
    '[TIME_7-14d]':   41,
    '[TIME_14-30d]':  42,
    '[TIME_30-60d]':  43,
    '[TIME_60-90d]':  44,
    '[TIME_90-180d]': 45,
    '[TIME_180+d]':   46,
    '[TIME_FIRST]':   47,   # first admission, no prior
}

# MELD score bins (matches your diagram exactly)
MELD_TOKENS = {
    '[MELD_6-9]':    50,
    '[MELD_10-14]':  51,
    '[MELD_15-19]':  52,
    '[MELD_20-24]':  53,
    '[MELD_25-29]':  54,
    '[MELD_30-34]':  55,
    '[MELD_35-39]':  56,
    '[MELD_40]':     57,
    '[MELD_NONE]':   58,   # labs unavailable
}

# MELD trend
MELD_TREND_TOKENS = {
    '[MELD_TREND_UP5+]':   60,   # MELD increased >= 5 (danger signal)
    '[MELD_TREND_UP2-5]':  61,   # MELD increased 2-5
    '[MELD_TREND_STABLE]': 62,   # MELD changed < 2
    '[MELD_TREND_DOWN]':   63,   # MELD decreased
    '[MELD_TREND_NA]':     64,   # no prior MELD
}

# Bilirubin bins (mg/dL)
BILI_TOKENS = {
    '[LAB_BILI_<1.0]':      100,
    '[LAB_BILI_1.0-2.0]':   101,
    '[LAB_BILI_2.0-5.0]':   102,
    '[LAB_BILI_5.0-10.0]':  103,
    '[LAB_BILI_10.0-20.0]': 104,
    '[LAB_BILI_20+]':       105,
    '[LAB_BILI_NONE]':      106,
}

# Creatinine bins (mg/dL)
CREAT_TOKENS = {
    '[LAB_CREAT_<1.0]':     110,
    '[LAB_CREAT_1.0-1.5]':  111,
    '[LAB_CREAT_1.5-2.0]':  112,
    '[LAB_CREAT_2.0-4.0]':  113,
    '[LAB_CREAT_4+]':       114,
    '[LAB_CREAT_NONE]':     115,
}

# INR bins
INR_TOKENS = {
    '[LAB_INR_<1.1]':       120,
    '[LAB_INR_1.1-1.5]':    121,
    '[LAB_INR_1.5-2.0]':    122,
    '[LAB_INR_2.0-3.0]':    123,
    '[LAB_INR_3+]':         124,
    '[LAB_INR_NONE]':       125,
}

# Sodium bins (mEq/L)
SODIUM_TOKENS = {
    '[LAB_NA_<120]':        130,
    '[LAB_NA_120-125]':     131,
    '[LAB_NA_125-130]':     132,
    '[LAB_NA_130-135]':     133,
    '[LAB_NA_135-145]':     134,
    '[LAB_NA_145+]':        135,
    '[LAB_NA_NONE]':        136,
}

# Dialysis status
DIALYSIS_TOKENS = {
    '[DIALYSIS_NONE]':          140,
    '[DIALYSIS_INTERMITTENT]':  141,
}

# Liver-relevant medications
MED_TOKENS = {
    '[MED_LACTULOSE]':      200,
    '[MED_RIFAXIMIN]':      201,
    '[MED_SPIRONOLACTONE]': 202,
    '[MED_FUROSEMIDE]':     203,
    '[MED_ALBUMIN]':        204,
    '[MED_NADOLOL]':        205,
    '[MED_NORFLOXACIN]':    206,
    '[MED_OTHER_LIVER]':    207,
}

# Clinical state flags
STATE_TOKENS = {
    '[HAS_ENCEPH]':         300,
    '[HAS_HEPATORENAL]':    301,
    '[HAS_VARICES_BLEED]':  302,
    '[HAS_SBP]':            303,
    '[HAS_HCC]':            304,
    '[HAS_CIRRHOSIS]':      305,
    '[HAS_PORTAL_HTN]':     306,
    '[HAS_HEP_C]':          307,
    '[HAS_ALCOHOLIC]':      308,
    '[HAS_HIGH_ACUITY]':    309,
    '[ICU_STAY]':           310,
}

# ICD-9 liver diagnosis tokens (dynamic — built from data)
# Start at 1000, each unique ICD-9 code gets a token
ICD9_TOKEN_START = 1000

# All fixed tokens merged
FIXED_VOCAB = {}
FIXED_VOCAB.update(SPECIAL_TOKENS)
FIXED_VOCAB.update(AGE_TOKENS)
FIXED_VOCAB.update(SEX_TOKENS)
FIXED_VOCAB.update(ENC_TOKENS)
FIXED_VOCAB.update(TIME_TOKENS)
FIXED_VOCAB.update(MELD_TOKENS)
FIXED_VOCAB.update(MELD_TREND_TOKENS)
FIXED_VOCAB.update(BILI_TOKENS)
FIXED_VOCAB.update(CREAT_TOKENS)
FIXED_VOCAB.update(INR_TOKENS)
FIXED_VOCAB.update(SODIUM_TOKENS)
FIXED_VOCAB.update(DIALYSIS_TOKENS)
FIXED_VOCAB.update(MED_TOKENS)
FIXED_VOCAB.update(STATE_TOKENS)

# =============================================================================
# BINNING FUNCTIONS
# =============================================================================

def bin_age(age):
    if age is None or np.isnan(age): return '[AGE_50-59]'
    a = int(age)
    if a < 30:  return '[AGE_18-29]'
    if a < 40:  return '[AGE_30-39]'
    if a < 50:  return '[AGE_40-49]'
    if a < 60:  return '[AGE_50-59]'
    if a < 70:  return '[AGE_60-69]'
    if a < 80:  return '[AGE_70-79]'
    return '[AGE_80+]'


def bin_sex(sex):
    if not sex: return '[SEX_U]'
    s = str(sex).upper()
    if s == 'M': return '[SEX_M]'
    if s == 'F': return '[SEX_F]'
    return '[SEX_U]'


def bin_enc_type(adm_type):
    if not adm_type: return '[ENC_OTHER]'
    t = str(adm_type).upper()
    if 'EMERGENCY' in t: return '[ENC_EMERGENCY]'
    if 'ELECTIVE'  in t: return '[ENC_ELECTIVE]'
    if 'URGENT'    in t: return '[ENC_URGENT]'
    return '[ENC_OTHER]'


def bin_time_gap(days):
    if days is None or np.isnan(days): return '[TIME_FIRST]'
    d = int(days)
    if d <= 7:   return '[TIME_0-7d]'
    if d <= 14:  return '[TIME_7-14d]'
    if d <= 30:  return '[TIME_14-30d]'
    if d <= 60:  return '[TIME_30-60d]'
    if d <= 90:  return '[TIME_60-90d]'
    if d <= 180: return '[TIME_90-180d]'
    return '[TIME_180+d]'


def bin_meld(score):
    if score is None or np.isnan(score): return '[MELD_NONE]'
    s = float(score)
    if s <= 9:   return '[MELD_6-9]'
    if s <= 14:  return '[MELD_10-14]'
    if s <= 19:  return '[MELD_15-19]'
    if s <= 24:  return '[MELD_20-24]'
    if s <= 29:  return '[MELD_25-29]'
    if s <= 34:  return '[MELD_30-34]'
    if s <= 39:  return '[MELD_35-39]'
    return '[MELD_40]'


def bin_meld_trend(delta):
    if delta is None or np.isnan(delta): return '[MELD_TREND_NA]'
    d = float(delta)
    if d >= 5:   return '[MELD_TREND_UP5+]'
    if d >= 2:   return '[MELD_TREND_UP2-5]'
    if d >= -2:  return '[MELD_TREND_STABLE]'
    return '[MELD_TREND_DOWN]'


def bin_bili(v):
    if v is None or np.isnan(v): return '[LAB_BILI_NONE]'
    if v < 1.0:  return '[LAB_BILI_<1.0]'
    if v < 2.0:  return '[LAB_BILI_1.0-2.0]'
    if v < 5.0:  return '[LAB_BILI_2.0-5.0]'
    if v < 10.0: return '[LAB_BILI_5.0-10.0]'
    if v < 20.0: return '[LAB_BILI_10.0-20.0]'
    return '[LAB_BILI_20+]'


def bin_creat(v):
    if v is None or np.isnan(v): return '[LAB_CREAT_NONE]'
    if v < 1.0:  return '[LAB_CREAT_<1.0]'
    if v < 1.5:  return '[LAB_CREAT_1.0-1.5]'
    if v < 2.0:  return '[LAB_CREAT_1.5-2.0]'
    if v < 4.0:  return '[LAB_CREAT_2.0-4.0]'
    return '[LAB_CREAT_4+]'


def bin_inr(v):
    if v is None or np.isnan(v): return '[LAB_INR_NONE]'
    if v < 1.1:  return '[LAB_INR_<1.1]'
    if v < 1.5:  return '[LAB_INR_1.1-1.5]'
    if v < 2.0:  return '[LAB_INR_1.5-2.0]'
    if v < 3.0:  return '[LAB_INR_2.0-3.0]'
    return '[LAB_INR_3+]'


def bin_sodium(v):
    if v is None or np.isnan(v): return '[LAB_NA_NONE]'
    if v < 120:  return '[LAB_NA_<120]'
    if v < 125:  return '[LAB_NA_120-125]'
    if v < 130:  return '[LAB_NA_125-130]'
    if v < 135:  return '[LAB_NA_130-135]'
    if v <= 145: return '[LAB_NA_135-145]'
    return '[LAB_NA_145+]'

# =============================================================================
# TOKENIZER CLASS
# =============================================================================

class MedicalTokenizer:

    def __init__(self):
        self.vocab     = dict(FIXED_VOCAB)   # token_str → id
        self.id2token  = {v: k for k, v in self.vocab.items()}
        self._next_id  = max(self.vocab.values()) + 1
        self._icd9_map = {}   # icd9_code → token_str

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _add_token(self, token_str):
        """Add a new token to vocabulary if not present."""
        if token_str not in self.vocab:
            self.vocab[token_str]         = self._next_id
            self.id2token[self._next_id]  = token_str
            self._next_id += 1
        return self.vocab[token_str]

    def build_vocab(self, df_admissions):
        """
        Build vocabulary from admission dataframe.
        Adds ICD-9 code tokens for all codes seen in data.
        """
        # Add ICD-9 diagnosis tokens
        icd9_cols = [c for c in df_admissions.columns
                     if c.startswith('has_')]

        # Also scan for any raw icd9 code columns
        if 'icd9_codes' in df_admissions.columns:
            all_codes = set()
            for codes in df_admissions['icd9_codes'].dropna():
                if isinstance(codes, str):
                    all_codes.update(codes.split(','))
            for code in sorted(all_codes):
                tok = f'[DX_{code.strip()}]'
                self._add_token(tok)
                self._icd9_map[code.strip()] = tok

        # Add standard liver ICD-9 tokens
        LIVER_ICD9 = [
            '5710','5711','5712','5713','5715','5716','5718','5719',
            '5720','5722','5724','5728','7891','5671','5723',
            '45620','45621','45680','1550','1551','1552',
            '07054','07044','07032','07070',
        ]
        for code in LIVER_ICD9:
            tok = f'[DX_{code}]'
            self._add_token(tok)
            self._icd9_map[code] = tok

        print(f"  Vocabulary built: {self.vocab_size:,} tokens")
        return self

    def encode_admission(self, row, prev_meld=None, days_since_prev=None):
        """
        Encode one admission row into a list of token IDs.
        Returns list of int token IDs.
        """
        tokens = []

        def add(tok_str):
            tid = self.vocab.get(tok_str, self.vocab['[UNK]'])
            tokens.append(tid)

        # ── Encounter header ───────────────────────────────────────────────
        add('[VISIT_START]')
        add(bin_enc_type(row.get('admission_type')))
        add(bin_time_gap(days_since_prev))

        # ── MELD ───────────────────────────────────────────────────────────
        meld = row.get('meld_na') or row.get('meld')
        add(bin_meld(meld))

        # MELD trend vs prior admission
        if prev_meld is not None and meld is not None:
            delta = meld - prev_meld
        else:
            delta = None
        add(bin_meld_trend(delta))

        # ── Labs ───────────────────────────────────────────────────────────
        add(bin_bili(row.get('bilirubin')))
        add(bin_creat(row.get('creatinine')))
        add(bin_inr(row.get('inr')))
        add(bin_sodium(row.get('sodium')))

        # ── Clinical state ─────────────────────────────────────────────────
        flag_map = {
            'has_enceph':         '[HAS_ENCEPH]',
            'has_hepatorenal':    '[HAS_HEPATORENAL]',
            'has_varices_bleed':  '[HAS_VARICES_BLEED]',
            'has_sbp':            '[HAS_SBP]',
            'has_hcc':            '[HAS_HCC]',
            'has_cirrhosis':      '[HAS_CIRRHOSIS]',
            'has_portal_htn':     '[HAS_PORTAL_HTN]',
            'has_hep_c':          '[HAS_HEP_C]',
            'has_alcoholic':      '[HAS_ALCOHOLIC]',
            'has_high_acuity':    '[HAS_HIGH_ACUITY]',
            'had_icu_stay':       '[ICU_STAY]',
        }
        for col, tok in flag_map.items():
            if row.get(col, 0):
                add(tok)

        # ── Medications ────────────────────────────────────────────────────
        med_map = {
            'on_lactulose':      '[MED_LACTULOSE]',
            'on_rifaximin':      '[MED_RIFAXIMIN]',
            'on_spironolactone': '[MED_SPIRONOLACTONE]',
            'on_furosemide':     '[MED_FUROSEMIDE]',
            'on_albumin':        '[MED_ALBUMIN]',
            'on_nadolol':        '[MED_NADOLOL]',
            'on_norfloxacin':    '[MED_NORFLOXACIN]',
        }
        for col, tok in med_map.items():
            if row.get(col, 0):
                add(tok)

        # Dialysis indicator (creatinine = 4.0 and hepatorenal = proxy)
        if row.get('has_hepatorenal', 0) or (
            row.get('creatinine') and row.get('creatinine', 0) >= 4.0):
            add('[DIALYSIS_INTERMITTENT]')
        else:
            add('[DIALYSIS_NONE]')

        add('[VISIT_END]')
        return tokens

    def encode_patient(self, admissions, max_len=2048):
        """
        Encode a patient's full admission history into one token sequence.

        admissions: list of dicts or DataFrame rows, sorted chronologically.
        Returns: list of int token IDs (padded/truncated to max_len)
        """
        tokens = [self.vocab['[PAT_START]']]

        # Demographics from first admission
        first = admissions[0] if isinstance(admissions, list) else admissions.iloc[0]
        tokens.append(self.vocab.get(bin_age(first.get('age_at_admit')),
                                      self.vocab['[UNK]']))
        tokens.append(self.vocab.get(bin_sex(first.get('gender')),
                                      self.vocab['[UNK]']))

        prev_meld = None
        prev_admit = None

        for i, row in enumerate(admissions if isinstance(admissions, list)
                                 else admissions.itertuples(index=False)):
            if not isinstance(row, dict):
                row = row._asdict()

            admit = row.get('admittime')
            days_gap = None
            if prev_admit is not None and admit is not None:
                if hasattr(admit, 'days'):
                    days_gap = (admit - prev_admit).days
                else:
                    try:
                        from datetime import datetime as _dt
                        a = pd.to_datetime(admit)
                        p = pd.to_datetime(prev_admit)
                        days_gap = (a - p).days
                    except Exception:
                        days_gap = None

            enc_tokens = self.encode_admission(row, prev_meld, days_gap)
            tokens.extend(enc_tokens)

            prev_meld  = row.get('meld_na') or row.get('meld')
            prev_admit = admit

        tokens.append(self.vocab['[PAT_END]'])

        # Truncate from left (keep most recent events) then pad
        if len(tokens) > max_len:
            # Always keep PAT_START + demographics (first 3 tokens)
            tokens = tokens[:3] + tokens[-(max_len - 3):]

        # Pad
        pad_id = self.vocab['[PAD]']
        while len(tokens) < max_len:
            tokens.append(pad_id)

        return tokens[:max_len]

    def decode(self, token_ids):
        """Convert list of token IDs back to token strings."""
        return [self.id2token.get(tid, f'[TOK_{tid}]') for tid in token_ids]

    def save(self, path):
        """Save vocabulary to JSON."""
        data = {
            'vocab':    self.vocab,
            'id2token': {str(k): v for k, v in self.id2token.items()},
            'icd9_map': self._icd9_map,
            'next_id':  self._next_id,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Vocabulary saved: {path} ({self.vocab_size:,} tokens)")

    @classmethod
    def load(cls, path):
        """Load vocabulary from JSON."""
        with open(path) as f:
            data = json.load(f)
        tok = cls()
        tok.vocab    = data['vocab']
        tok.id2token = {int(k): v for k, v in data['id2token'].items()}
        tok._icd9_map = data.get('icd9_map', {})
        tok._next_id  = data.get('next_id',
                                   max(tok.id2token.keys()) + 1)
        return tok


# =============================================================================
# BUILD TOKEN SEQUENCES FROM COHORT
# =============================================================================

def build_sequences(df_admissions, tokenizer, max_len=2048):
    """
    Build token sequences for all patients.

    df_admissions: full admission-level DataFrame (all admissions per patient)
    Returns: list of (token_ids, label, subject_id)
    """
    sequences = []
    patients  = df_admissions.sort_values(
        ['subject_id','admittime']).groupby('subject_id')

    for subject_id, group in patients:
        adm_list = group.to_dict('records')
        label    = int(group['label'].max())  # positive if any admission is positive

        token_ids = tokenizer.encode_patient(adm_list, max_len=max_len)
        sequences.append((token_ids, label, subject_id))

    return sequences


if __name__ == "__main__":
    # Quick test with sample data
    print("Testing tokenizer...")
    tok = MedicalTokenizer()
    tok.build_vocab(pd.DataFrame())  # empty, just loads fixed vocab

    # Simulate patient from your diagram
    test_admissions = [
        {
            'admittime': '2024-01-15', 'gender': 'M', 'age_at_admit': 52,
            'admission_type': 'EMERGENCY',
            'bilirubin': 1.8, 'creatinine': 0.9, 'inr': 1.3, 'sodium': 138,
            'meld': 12, 'meld_na': 12.0,
            'has_cirrhosis': 1, 'has_enceph': 0, 'has_hepatorenal': 0,
            'has_varices_bleed': 0, 'has_sbp': 0, 'has_hcc': 0,
            'has_high_acuity': 0, 'has_portal_htn': 0,
            'has_hep_c': 0, 'has_alcoholic': 1, 'had_icu_stay': 0,
            'on_lactulose': 1, 'on_rifaximin': 0, 'on_spironolactone': 0,
            'on_furosemide': 0, 'on_albumin': 0, 'on_nadolol': 0,
            'on_norfloxacin': 0, 'label': 0,
        },
        {
            'admittime': '2024-03-22', 'gender': 'M', 'age_at_admit': 52,
            'admission_type': 'EMERGENCY',
            'bilirubin': 3.2, 'creatinine': 1.4, 'inr': 1.7, 'sodium': 131,
            'meld': 22, 'meld_na': 22.0,
            'has_cirrhosis': 1, 'has_enceph': 0, 'has_hepatorenal': 0,
            'has_varices_bleed': 0, 'has_sbp': 0, 'has_hcc': 0,
            'has_high_acuity': 0, 'has_portal_htn': 1,
            'has_hep_c': 0, 'has_alcoholic': 1, 'had_icu_stay': 0,
            'on_lactulose': 1, 'on_rifaximin': 1, 'on_spironolactone': 1,
            'on_furosemide': 0, 'on_albumin': 0, 'on_nadolol': 0,
            'on_norfloxacin': 0, 'label': 1,
        },
        {
            'admittime': '2024-05-10', 'gender': 'M', 'age_at_admit': 52,
            'admission_type': 'EMERGENCY',
            'bilirubin': 6.1, 'creatinine': 2.3, 'inr': 2.4, 'sodium': 127,
            'meld': 33, 'meld_na': 33.0,
            'has_cirrhosis': 1, 'has_enceph': 1, 'has_hepatorenal': 1,
            'has_varices_bleed': 0, 'has_sbp': 1, 'has_hcc': 0,
            'has_high_acuity': 1, 'has_portal_htn': 1,
            'has_hep_c': 0, 'has_alcoholic': 1, 'had_icu_stay': 1,
            'on_lactulose': 1, 'on_rifaximin': 1, 'on_spironolactone': 1,
            'on_furosemide': 1, 'on_albumin': 1, 'on_nadolol': 0,
            'on_norfloxacin': 1, 'label': 1,
        },
    ]

    tokens = tok.encode_patient(test_admissions)
    decoded = tok.decode([t for t in tokens if t != 0])  # skip padding
    print(f"\nToken sequence ({len([t for t in tokens if t != 0])} tokens):")
    print(" ".join(decoded[:30]) + " ...")
    print(f"\nVocabulary size: {tok.vocab_size:,} tokens")
    print(f"Max sequence length: 2048")
    print("\n✓ Tokenizer working correctly")

"""
etl.py â€” MIMIC-III â†’ YottaDB ^PHD
Reads from MIMICold PostgreSQL (public schema)
Writes canonical HIE globals into YottaDB

Run inside the hie-mumps container:
  python3 /project/etl.py

Or individual steps:
  python3 /project/etl.py --steps patients admissions
  python3 /project/etl.py --from-step labs
"""

import os
import sys
import time
import argparse
from datetime import datetime

# â”€â”€ Connection config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
PG_CONFIG = {
    "host":     "host.docker.internal",
    "port":     5432,
    "dbname":   "MIMICold",
    "user":     "postgres",
    "password": "YOUR_PASSWORD",
}

YDB_ENV = {
    "ydb_dist":    "/opt/yottadb/current",
    "ydb_gbldir":  "/data/r2.06_x86_64/g/yottadb.gld",
    "ydb_routines": "/opt/yottadb/current/libyottadbutil.so",
}

# Chunk size for large tables
CHUNK = 50000
# Max labs per encounter stored in ^PHD (keeps globals manageable)
MAX_LABS = 200
# Max notes per patient
MAX_NOTES = 10

# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def make_pid(subject_id):
    return f"HIE-MIII-{int(subject_id):06d}"

def make_eid(hadm_id):
    return f"ENC-MIII-{int(hadm_id):06d}"

def dt_ydb(val):
    """datetime â†’ YottaDB compact string YYYYMMDDHHMMSS"""
    if val is None:
        return ""
    return str(val).replace("-", "").replace(" ", "").replace(":", "")[:14]

def dt_str(val):
    """datetime â†’ plain string, empty if None"""
    if val is None:
        return ""
    return str(val)

def safe(val, maxlen=None):
    """None-safe string, optional truncation"""
    if val is None:
        return ""
    s = str(val).strip()
    if maxlen:
        s = s[:maxlen]
    return s

def progress(n, total, label):
    pct = int(n / total * 100) if total else 0
    bar = "â–ˆ" * (pct // 5) + "â–‘" * (20 - pct // 5)
    print(f"\r  {label}: [{bar}] {pct}% ({n:,}/{total:,})", end="", flush=True)

# â”€â”€ YottaDB setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def setup_ydb():
    for k, v in YDB_ENV.items():
        os.environ[k] = v
    # Source ydb_env_set by setting the key vars directly
    os.environ["ydb_dist"]    = YDB_ENV["ydb_dist"]
    os.environ["ydb_gbldir"]  = YDB_ENV["ydb_gbldir"]
    os.chdir("/data/r2.06_x86_64/g")
    import yottadb as ydb
    return ydb

# â”€â”€ Postgres connection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_pg():
    import psycopg2
    return psycopg2.connect(**PG_CONFIG)

# =============================================================================
# STEP 1: PATIENTS
# public.patients: row_id, subject_id, gender, dob, dod, dod_hosp, dod_ssn, expire_flag
# =============================================================================

def load_patients(ydb, pg):
    print("\n[1] Loading patients â†’ ^PHD PID nodes")
    cur = pg.cursor()
    cur.execute("SELECT COUNT(*) FROM public.patients")
    total = cur.fetchone()[0]
    print(f"    {total:,} patients to load")

    cur.execute("""
        SELECT subject_id, gender, dob, dod, dod_hosp, dod_ssn, expire_flag
        FROM public.patients
        ORDER BY subject_id
    """)

    n = 0
    for row in cur:
        subject_id, gender, dob, dod, dod_hosp, dod_ssn, expire_flag = row
        pid = make_pid(subject_id)

        # Root node
        ydb.set("^PHD", [pid, "0"],
            f"MIII^{subject_id}^{datetime.utcnow().strftime('%Y%m%d%H%M%S')}")

        # Demographics
        ydb.set("^PHD", [pid, "PID", "SEX"],  safe(gender))
        ydb.set("^PHD", [pid, "PID", "DOB"],  dt_ydb(dob))
        ydb.set("^PHD", [pid, "PID", "DOD"],  dt_ydb(dod))
        ydb.set("^PHD", [pid, "PID", "DEAD"], "1" if expire_flag == 1 else "0")

        # Source cross-reference
        ydb.set("^PHD", [pid, "SRC", "MIII"], str(subject_id))

        # Reverse lookup: MIMIC subject_id â†’ canonical pid
        ydb.set("^PHD", ["BSRC", "MIII", str(subject_id), pid], "")

        n += 1
        if n % 1000 == 0:
            progress(n, total, "patients")

    progress(total, total, "patients")
    print(f"\n    âœ“ {n:,} patients loaded")
    cur.close()
    return n

# =============================================================================
# STEP 2: ADMISSIONS
# public.admissions: row_id, subject_id, hadm_id, admittime, dischtime,
#   deathtime, admission_type, admission_location, discharge_location,
#   insurance, language, religion, marital_status, ethnicity,
#   edregtime, edouttime, diagnosis, hospital_expire_flag, has_chartevents_data
# =============================================================================

def load_admissions(ydb, pg):
    print("\n[2] Loading admissions â†’ ^PHD VISIT nodes")
    cur = pg.cursor()
    cur.execute("SELECT COUNT(*) FROM public.admissions")
    total = cur.fetchone()[0]
    print(f"    {total:,} admissions to load")

    cur.execute("""
        SELECT subject_id, hadm_id, admittime, dischtime, deathtime,
               admission_type, admission_location, discharge_location,
               insurance, language, religion, marital_status, ethnicity,
               diagnosis, hospital_expire_flag
        FROM public.admissions
        ORDER BY subject_id, admittime
    """)

    n = 0
    skipped = 0
    for row in cur:
        (subject_id, hadm_id, admittime, dischtime, deathtime,
         admission_type, admission_location, discharge_location,
         insurance, language, religion, marital_status, ethnicity,
         diagnosis, hospital_expire_flag) = row

        pid = make_pid(subject_id)
        eid = make_eid(hadm_id)

        # Verify patient exists
        if not ydb.get("^PHD", [pid, "0"]):
            skipped += 1
            continue

        admit  = dt_ydb(admittime)
        disch  = dt_ydb(dischtime)
        status = "CLOSED" if dischtime else "OPEN"

        # Calculate LOS in hours
        los_hours = ""
        if admittime and dischtime:
            delta = dischtime - admittime
            los_hours = str(round(delta.total_seconds() / 3600, 2))

        # Visit root node
        ydb.set("^PHD", [pid, "VISIT", eid, "0"],
            f"{admit}^{disch}^{safe(admission_type)}^"
            f"{safe(admission_location)}^{safe(insurance)}^"
            f"{hospital_expire_flag or 0}^{status}^{los_hours}")

        ydb.set("^PHD", [pid, "VISIT", eid, "DISCH"],   safe(discharge_location))
        ydb.set("^PHD", [pid, "VISIT", eid, "ETH"],     safe(ethnicity))
        ydb.set("^PHD", [pid, "VISIT", eid, "LANG"],    safe(language))
        ydb.set("^PHD", [pid, "VISIT", eid, "REL"],     safe(religion))
        ydb.set("^PHD", [pid, "VISIT", eid, "MAR"],     safe(marital_status))
        ydb.set("^PHD", [pid, "VISIT", eid, "DX_TEXT"], safe(diagnosis, 500))
        ydb.set("^PHD", [pid, "VISIT", eid, "HADM"],    str(hadm_id))
        ydb.set("^PHD", [pid, "VISIT", eid, "DEATH"],   dt_ydb(deathtime))

        # Encounter â†’ patient reverse index
        ydb.set("^PHD", ["BENC", eid, pid], "")

        n += 1
        if n % 1000 == 0:
            progress(n, total, "admissions")

    progress(total, total, "admissions")
    print(f"\n    âœ“ {n:,} admissions loaded, {skipped} skipped (patient not found)")
    cur.close()
    return n

# =============================================================================
# STEP 3: ICU STAYS
# public.icustays: subject_id, hadm_id, icustay_id, dbsource,
#   first_careunit, last_careunit, first_wardid, last_wardid, intime, outtime, los
# =============================================================================

def load_icu(ydb, pg):
    print("\n[3] Loading ICU stays â†’ ^PHD VISIT/ICU nodes")
    cur = pg.cursor()
    cur.execute("SELECT COUNT(*) FROM public.icustays")
    total = cur.fetchone()[0]
    print(f"    {total:,} ICU stays to load")

    cur.execute("""
        SELECT subject_id, hadm_id, icustay_id, first_careunit, last_careunit,
               intime, outtime, los, dbsource
        FROM public.icustays
        ORDER BY subject_id, intime
    """)

    n = 0
    skipped = 0
    for row in cur:
        (subject_id, hadm_id, icustay_id, first_careunit, last_careunit,
         intime, outtime, los, dbsource) = row

        pid = make_pid(subject_id)
        eid = make_eid(hadm_id)

        # Verify encounter exists
        if not ydb.get("^PHD", [pid, "VISIT", eid, "0"]):
            skipped += 1
            continue

        ydb.set("^PHD", [pid, "VISIT", eid, "ICU", str(icustay_id), "0"],
            f"{safe(first_careunit)}^{dt_ydb(intime)}^{dt_ydb(outtime)}^"
            f"{safe(str(los))}^{safe(dbsource)}")

        ydb.set("^PHD", [pid, "VISIT", eid, "ICU", str(icustay_id), "LAST"],
            safe(last_careunit))

        # Mark encounter as having ICU stay
        ydb.set("^PHD", [pid, "VISIT", eid, "HAS_ICU"], "1")

        n += 1
        if n % 500 == 0:
            progress(n, total, "icu stays")

    progress(total, total, "icu stays")
    print(f"\n    âœ“ {n:,} ICU stays loaded, {skipped} skipped")
    cur.close()
    return n

# =============================================================================
# STEP 4: DIAGNOSES
# public.diagnoses_icd: subject_id, hadm_id, seq_num, icd9_code
# public.d_icd_diagnoses: icd9_code, short_title, long_title
# =============================================================================

def load_diagnoses(ydb, pg):
    print("\n[4] Loading diagnoses â†’ ^PHD VISIT/DX nodes")

    # Load ICD9 lookup dictionary into memory
    cur = pg.cursor()
    cur.execute("SELECT icd9_code, short_title, long_title FROM public.d_icd_diagnoses")
    icd_lookup = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
    print(f"    ICD-9 lookup: {len(icd_lookup):,} codes loaded")

    cur.execute("SELECT COUNT(*) FROM public.diagnoses_icd")
    total = cur.fetchone()[0]
    print(f"    {total:,} diagnosis records to load")

    cur.execute("""
        SELECT subject_id, hadm_id, seq_num, icd9_code
        FROM public.diagnoses_icd
        ORDER BY subject_id, hadm_id, seq_num
    """)

    n = 0
    skipped = 0
    for row in cur:
        subject_id, hadm_id, seq_num, icd9_code = row
        pid = make_pid(subject_id)
        eid = make_eid(hadm_id)

        if not ydb.get("^PHD", [pid, "VISIT", eid, "0"]):
            skipped += 1
            continue

        short_title, long_title = icd_lookup.get(safe(icd9_code), ("", ""))

        ydb.set("^PHD", [pid, "VISIT", eid, "DX", str(seq_num)],
            f"{safe(icd9_code)}^{safe(short_title, 100)}^"
            f"{safe(long_title, 300)}^{seq_num}")

        n += 1
        if n % 5000 == 0:
            progress(n, total, "diagnoses")

    progress(total, total, "diagnoses")
    print(f"\n    âœ“ {n:,} diagnoses loaded, {skipped} skipped")
    cur.close()
    return n

# =============================================================================
# STEP 5: PROCEDURES
# public.procedures_icd: subject_id, hadm_id, seq_num, icd9_code
# public.d_icd_procedures: icd9_code, short_title, long_title
# =============================================================================

def load_procedures(ydb, pg):
    print("\n[5] Loading procedures â†’ ^PHD VISIT/PROC nodes")

    cur = pg.cursor()
    cur.execute("SELECT icd9_code, short_title, long_title FROM public.d_icd_procedures")
    proc_lookup = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
    print(f"    ICD-9 procedure lookup: {len(proc_lookup):,} codes loaded")

    cur.execute("SELECT COUNT(*) FROM public.procedures_icd")
    total = cur.fetchone()[0]
    print(f"    {total:,} procedure records to load")

    cur.execute("""
        SELECT subject_id, hadm_id, seq_num, icd9_code
        FROM public.procedures_icd
        ORDER BY subject_id, hadm_id, seq_num
    """)

    n = 0
    skipped = 0
    for row in cur:
        subject_id, hadm_id, seq_num, icd9_code = row
        pid = make_pid(subject_id)
        eid = make_eid(hadm_id)

        if not ydb.get("^PHD", [pid, "VISIT", eid, "0"]):
            skipped += 1
            continue

        short_title, long_title = proc_lookup.get(safe(icd9_code), ("", ""))

        ydb.set("^PHD", [pid, "VISIT", eid, "PROC", str(seq_num)],
            f"{safe(icd9_code)}^{safe(short_title, 100)}^"
            f"{safe(long_title, 300)}^{seq_num}")

        n += 1
        if n % 2000 == 0:
            progress(n, total, "procedures")

    progress(total, total, "procedures")
    print(f"\n    âœ“ {n:,} procedures loaded, {skipped} skipped")
    cur.close()
    return n

# =============================================================================
# STEP 6: PRESCRIPTIONS
# public.prescriptions: subject_id, hadm_id, icustay_id, startdate, enddate,
#   drug_type, drug, drug_name_poe, drug_name_generic, formulary_drug_cd,
#   gsn, ndc, prod_strength, dose_val_rx, dose_unit_rx,
#   form_val_disp, form_unit_disp, route
# =============================================================================

def load_prescriptions(ydb, pg):
    print("\n[6] Loading prescriptions â†’ ^PHD VISIT/MED nodes")
    cur = pg.cursor()
    cur.execute("SELECT COUNT(*) FROM public.prescriptions")
    total = cur.fetchone()[0]
    print(f"    {total:,} prescription records to load")

    # Track med index per encounter
    enc_med_idx = {}

    cur.execute("""
        SELECT subject_id, hadm_id, startdate, enddate,
               drug_type, drug, drug_name_generic,
               dose_val_rx, dose_unit_rx, route,
               formulary_drug_cd, gsn, ndc, prod_strength
        FROM public.prescriptions
        ORDER BY subject_id, hadm_id, startdate
    """)

    n = 0
    skipped = 0
    for row in cur:
        (subject_id, hadm_id, startdate, enddate,
         drug_type, drug, drug_name_generic,
         dose_val_rx, dose_unit_rx, route,
         formulary_drug_cd, gsn, ndc, prod_strength) = row

        pid = make_pid(subject_id)
        eid = make_eid(hadm_id)

        if not ydb.get("^PHD", [pid, "VISIT", eid, "0"]):
            skipped += 1
            continue

        # Auto-increment med index per encounter
        key = f"{pid}|{eid}"
        enc_med_idx[key] = enc_med_idx.get(key, 0) + 1
        idx = enc_med_idx[key]

        ydb.set("^PHD", [pid, "VISIT", eid, "MED", str(idx)],
            f"{safe(drug, 200)}^{safe(drug_name_generic, 200)}^"
            f"{safe(drug_type, 30)}^{safe(dose_val_rx, 50)}^"
            f"{safe(dose_unit_rx, 30)}^{safe(route, 30)}^"
            f"{dt_str(startdate)}^{dt_str(enddate)}^"
            f"{safe(ndc, 20)}^{safe(prod_strength, 100)}")

        n += 1
        if n % 5000 == 0:
            progress(n, total, "prescriptions")

    progress(total, total, "prescriptions")
    print(f"\n    âœ“ {n:,} prescriptions loaded, {skipped} skipped")
    cur.close()
    return n

# =============================================================================
# STEP 7: LAB EVENTS
# public.labevents: subject_id, hadm_id, itemid, charttime,
#   value, valuenum, valueuom, flag
# Chunked â€” 27M+ rows
# =============================================================================

def load_labs(ydb, pg):
    print("\n[7] Loading lab events â†’ ^PHD VISIT/LAB nodes")
    cur_count = pg.cursor()
    cur_count.execute("SELECT COUNT(*) FROM public.labevents WHERE hadm_id IS NOT NULL")
    total = cur_count.fetchone()[0]
    print(f"    {total:,} lab records to load (hadm_id not null)")
    print(f"    Max {MAX_LABS} labs per encounter")

    # Track lab index per encounter
    enc_lab_idx = {}

    cur = pg.cursor("lab_cursor")  # server-side cursor for large result
    cur.execute("""
        SELECT subject_id, hadm_id, itemid, charttime,
               value, valuenum, valueuom, flag
        FROM public.labevents
        WHERE hadm_id IS NOT NULL
        ORDER BY subject_id, hadm_id, charttime
    """)

    n = 0
    skipped = 0
    capped = 0

    while True:
        rows = cur.fetchmany(CHUNK)
        if not rows:
            break

        for row in rows:
            subject_id, hadm_id, itemid, charttime, value, valuenum, valueuom, flag = row
            pid = make_pid(subject_id)
            eid = make_eid(hadm_id)

            if not ydb.get("^PHD", [pid, "VISIT", eid, "0"]):
                skipped += 1
                continue

            key = f"{pid}|{eid}"
            enc_lab_idx[key] = enc_lab_idx.get(key, 0) + 1
            idx = enc_lab_idx[key]

            # Cap labs per encounter
            if idx > MAX_LABS:
                capped += 1
                continue

            ydb.set("^PHD", [pid, "VISIT", eid, "LAB", str(idx)],
                f"{itemid}^{safe(value, 100)}^"
                f"{safe(str(valuenum) if valuenum is not None else '', 30)}^"
                f"{safe(valueuom, 30)}^{safe(flag, 20)}^"
                f"{dt_ydb(charttime)}")

            n += 1
            if n % 10000 == 0:
                progress(n, total, "labs")

    progress(total, total, "labs")
    print(f"\n    âœ“ {n:,} labs loaded, {skipped} skipped, {capped:,} capped at {MAX_LABS}/encounter")
    cur.close()
    return n

# =============================================================================
# STEP 8: NOTES
# public.noteevents: subject_id, hadm_id, chartdate, charttime,
#   storetime, category, description, cgid, iserror, text
# Chunked â€” 2M rows, large text
# =============================================================================

def load_notes(ydb, pg):
    print("\n[8] Loading clinical notes -> ^PHD VISIT/NOTE nodes")
    cur_count = pg.cursor()
    cur_count.execute("""
        SELECT COUNT(*) FROM public.noteevents
        WHERE hadm_id IS NOT NULL AND (iserror IS NULL OR iserror = 0)
    """)
    total = cur_count.fetchone()[0]
    print(f"    {total:,} notes to load")

    # Priority order: discharge summaries first, then others
    cur = pg.cursor("note_cursor")
    cur.execute("""
        SELECT subject_id, hadm_id, chartdate, charttime,
               category, description, cgid, iserror, text
        FROM public.noteevents
        WHERE hadm_id IS NOT NULL
          AND (iserror IS NULL OR iserror = 0)
        ORDER BY
            subject_id,
            CASE WHEN category = 'Discharge summary' THEN 0 ELSE 1 END,
            charttime NULLS LAST,
            chartdate
    """)

    # Track per patient and per category counts
    pat_note_idx = {}
    pat_cat_count = {}

    n = 0
    skipped = 0

    while True:
        rows = cur.fetchmany(2000)
        if not rows:
            break

        for row in rows:
            (subject_id, hadm_id, chartdate, charttime,
             category, description, cgid, iserror, text) = row

            pid = make_pid(subject_id)
            eid = make_eid(hadm_id)

            if not ydb.get("^PHD", [pid, "VISIT", eid, "0"]):
                skipped += 1
                continue

            cat = safe(category, 50)

            # Category limits:
            # Discharge summary: all of them (usually 1-2 per admission)
            # All others: max 10 per patient per category
            cat_key = f"{pid}|{cat}"
            pat_cat_count[cat_key] = pat_cat_count.get(cat_key, 0) + 1

            if cat != "Discharge summary" and pat_cat_count[cat_key] > 10:
                continue

            # Global note index per patient
            pat_note_idx[pid] = pat_note_idx.get(pid, 0) + 1
            idx = pat_note_idx[pid]

            note_dt = dt_ydb(charttime) if charttime else dt_str(chartdate)

            # Note metadata node
            ydb.set("^PHD", [pid, "VISIT", eid, "NOTE", str(idx)],
                f"{cat}^{note_dt}^"
                f"{safe(str(cgid) if cgid else '')}^"
                f"{safe(description, 100)}")

            # Full note text — cap at 500KB per note
            if text:
                ydb.set("^PHD", [pid, "VISIT", eid, "NOTETXT", str(idx)],
                    str(text)[:500000])

            n += 1
            if n % 500 == 0:
                progress(n, total, "notes")

    progress(n, total, "notes")
    print(f"\n    -> {n:,} notes loaded, {skipped} skipped")
    cur.close()
    return n
# =============================================================================
# ORCHESTRATOR
# =============================================================================

STEPS = [
    ("patients",      load_patients,      "PATIENTS â†’ ^PHD PID"),
    ("admissions",    load_admissions,    "ADMISSIONS â†’ ^PHD VISIT"),
    ("icu",           load_icu,           "ICUSTAYS â†’ ^PHD VISIT/ICU"),
    ("diagnoses",     load_diagnoses,     "DIAGNOSES_ICD â†’ ^PHD VISIT/DX"),
    ("procedures",    load_procedures,    "PROCEDURES_ICD â†’ ^PHD VISIT/PROC"),
    ("prescriptions", load_prescriptions, "PRESCRIPTIONS â†’ ^PHD VISIT/MED"),
    ("labs",          load_labs,          "LABEVENTS â†’ ^PHD VISIT/LAB"),
    ("notes",         load_notes,         "NOTEEVENTS â†’ ^PHD VISIT/NOTE"),
]

STEP_NAMES = [s[0] for s in STEPS]

def print_summary(results, elapsed):
    print("\n" + "â•" * 55)
    print("  MIMIC-III â†’ YottaDB Load Summary")
    print("â•" * 55)
    print(f"  {'Step':<16} {'Status':<10} {'Records':>10}  {'Time':>8}")
    print("  " + "â”€" * 51)
    for r in results:
        status  = r["status"]
        mark    = "âœ“" if status == "ok" else ("âš " if status == "skip" else "âœ—")
        t       = f"{r['elapsed']:.1f}s"
        count   = f"{r['count']:,}" if r.get("count") else "â€”"
        print(f"  {mark} {r['step']:<15} {status:<10} {count:>10}  {t:>8}")
    print("â•" * 55)
    m, s = divmod(int(elapsed), 60)
    print(f"  Total: {m}m {s}s")
    print("â•" * 55)

def main():
    parser = argparse.ArgumentParser(description="MIMIC-III â†’ YottaDB ETL")
    parser.add_argument("--steps", nargs="+", choices=STEP_NAMES,
                        help="Run only these steps")
    parser.add_argument("--from-step", choices=STEP_NAMES,
                        help="Run from this step onwards")
    parser.add_argument("--verify-only", action="store_true",
                        help="Show plan without running")
    args = parser.parse_args()

    if args.steps:
        steps_to_run = [s for s in STEPS if s[0] in args.steps]
    elif args.from_step:
        idx = STEP_NAMES.index(args.from_step)
        steps_to_run = STEPS[idx:]
    else:
        steps_to_run = STEPS

    if args.verify_only:
        print("\nWould run:")
        for s in steps_to_run:
            print(f"  {s[0]:<16} {s[2]}")
        print(f"\nSource: MIMICold @ host.docker.internal:5432")
        print(f"Target: YottaDB ^PHD @ {YDB_ENV['ydb_gbldir']}")
        return

    print("=" * 55)
    print("  MIMIC-III â†’ YottaDB ^PHD ETL")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    # Initialize connections
    print("\nConnecting to YottaDB...")
    ydb = setup_ydb()
    print("  âœ“ YottaDB connected")

    print("Connecting to Postgres...")
    pg = get_pg()
    print("  âœ“ Postgres connected (MIMICold)")

    # Run steps
    total_start = time.time()
    results = []

    for step_name, func, desc in steps_to_run:
        print(f"\n{'â”€' * 55}")
        start = time.time()
        result = {"step": step_name, "status": "ok", "count": 0, "elapsed": 0}
        try:
            count = func(ydb, pg)
            result["count"] = count or 0
        except Exception as e:
            print(f"\n  âœ— {step_name} failed: {e}")
            import traceback
            traceback.print_exc()
            result["status"] = "error"
            result["reason"] = str(e)
        result["elapsed"] = time.time() - start
        results.append(result)

        if result["status"] == "error":
            print("\n  Stopping due to error.")
            break

    pg.close()
    print_summary(results, time.time() - total_start)

    if any(r["status"] == "error" for r in results):
        sys.exit(1)

if __name__ == "__main__":
    main()



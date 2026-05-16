"""
sync_engine.py â€” YottaDB ^PHD â†’ Postgres hie schema
Reads canonical patient data from ^PHD globals
Writes to OMOP-aligned hie.* tables in MIMICold

Run inside the hie-mumps container:
  cd /data/r2.06_x86_64/g
  python3 /project/sync_engine.py
  python3 /project/sync_engine.py --steps person visits
  python3 /project/sync_engine.py --limit 1000  # test with 1000 patients first
"""

import os
import sys
import time
import argparse
from datetime import datetime

# â”€â”€ Configuration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
PG_CONFIG = {
    "host":     "host.docker.internal",
    "port":     5432,
    "dbname":   "MIMICold",
    "user":     "postgres",
    "password": "Panjwar4633",
}

YDB_ENV = {
    "ydb_dist":    "/opt/yottadb/current",
    "ydb_gbldir":  "/data/r2.06_x86_64/g/yottadb.gld",
    "ydb_routines": "/opt/yottadb/current/libyottadbutil.so",
}

BATCH_SIZE = 500   # Postgres commit batch size

# â”€â”€ Setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def setup_ydb():
    for k, v in YDB_ENV.items():
        os.environ[k] = v
    os.chdir("/data/r2.06_x86_64/g")
    import yottadb as ydb
    return ydb

def get_pg():
    import psycopg2
    return psycopg2.connect(**PG_CONFIG)

# â”€â”€ YDB helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def yget(ydb, *args):
    """Safe get â€” returns empty string on any error."""
    try:
        val = ydb.get(*args)
        if val is None:
            return ""
        return val.decode() if isinstance(val, bytes) else str(val)
    except Exception:
        return ""

def ynext(ydb, gbl, subs):
    """Safe subscript_next â€” returns None when exhausted."""
    try:
        val = ydb.subscript_next(gbl, subs)
        if val is None:
            return None
        return val.decode() if isinstance(val, bytes) else str(val)
    except Exception:
        return None

def walk_subs(ydb, gbl, path):
    """Yield all subscripts under a path."""
    sub = ""
    while True:
        sub = ynext(ydb, gbl, path + [sub])
        if not sub:
            break
        yield sub

def get_all_pids(ydb):
    """Yield all HIE patient IDs from ^PHD."""
    pid = ""
    while True:
        pid = ynext(ydb, "^PHD", [pid])
        if not pid:
            break
        if pid.startswith("HIE-"):
            yield pid

# â”€â”€ Field parsers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def parse_dob(dob_raw):
    """Convert MIMIC YDB DOB string to date components."""
    if not dob_raw or len(dob_raw) < 8:
        return None, None, None, None
    try:
        # Format: YYYYMMDDHHMMSS
        year  = int(dob_raw[0:4])
        month = int(dob_raw[4:6])
        day   = int(dob_raw[6:8])
        dt_str = f"{dob_raw[0:4]}-{dob_raw[4:6]}-{dob_raw[6:8]}"
        return year, month, day, dt_str
    except Exception:
        return None, None, None, None

def parse_dt(ydb_dt):
    """Convert YDB datetime string YYYYMMDDHHMMSS to ISO string."""
    if not ydb_dt or str(ydb_dt).strip() in ("", "None"):
        return None
    try:
        s = str(ydb_dt).strip()
        # Remove any non-numeric characters
        digits = "".join(c for c in s if c.isdigit())
        if len(digits) < 8:
            return None
        year  = digits[0:4]
        month = digits[4:6].lstrip("0") or "1"
        day   = digits[6:8].lstrip("0") or "1"
        # Validate ranges
        if int(year) < 1800 or int(year) > 2200: return None
        if int(month) < 1 or int(month) > 12: return None
        if int(day) < 1 or int(day) > 31: return None
        if len(digits) >= 14:
            return f"{year}-{month.zfill(2)}-{day.zfill(2)} {digits[8:10]}:{digits[10:12]}:{digits[12:14]}"
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    except Exception:
        return None

def safe(val, maxlen=None):
    if not val:
        return None
    s = str(val).strip()
    if not s:
        return None
    if maxlen:
        s = s[:maxlen]
    return s

def progress(n, total, label):
    if total == 0:
        return
    pct = int(n / total * 100)
    bar = "â–ˆ" * (pct // 5) + "â–‘" * (20 - pct // 5)
    print(f"\r  {label}: [{bar}] {pct}% ({n:,}/{total:,})", end="", flush=True)

# â”€â”€ Source ID â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_source_id(pg):
    with pg.cursor() as cur:
        cur.execute("SELECT source_id FROM hie.source WHERE source_code='MIII'")
        row = cur.fetchone()
        if not row:
            raise RuntimeError("MIII not found in hie.source. Run hie_schema.sql first.")
        return row[0]

# =============================================================================
# STEP 1: PERSON
# =============================================================================

def sync_person(ydb, pg, source_id, limit=None):
    print("\n[1] Syncing person â†’ hie.person")

    # Count patients
    n_total = sum(1 for _ in get_all_pids(ydb))
    if limit:
        n_total = min(n_total, limit)
    print(f"    {n_total:,} patients to sync")

    insert_sql = """
        INSERT INTO hie.person (
            person_id, source_id, src_subject_id,
            gender_concept_code,
            year_of_birth, month_of_birth, day_of_birth,
            birth_datetime, death_datetime,
            dob_raw, dod_raw,
            deceased, expire_flag, ydb_pid, synced_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
        )
        ON CONFLICT (person_id) DO UPDATE SET
            gender_concept_code = EXCLUDED.gender_concept_code,
            birth_datetime      = EXCLUDED.birth_datetime,
            death_datetime      = EXCLUDED.death_datetime,
            deceased            = EXCLUDED.deceased,
            synced_at           = NOW()
    """

    n = 0
    batch = []

    for pid in get_all_pids(ydb):
        if limit and n >= limit:
            break

        sex     = yget(ydb, "^PHD", [pid, "PID", "SEX"])
        dob_raw = yget(ydb, "^PHD", [pid, "PID", "DOB"])
        dod_raw = yget(ydb, "^PHD", [pid, "PID", "DOD"])
        dead    = yget(ydb, "^PHD", [pid, "PID", "DEAD"])
        src_id  = yget(ydb, "^PHD", [pid, "SRC", "MIII"])

        yr, mo, dy, dob_str = parse_dob(dob_raw)
        _, _, _, dod_str    = parse_dob(dod_raw)

        batch.append((
            pid,
            source_id,
            safe(src_id),
            safe(sex, 1),
            yr, mo, dy,
            dob_str,
            dod_str if dod_raw else None,
            safe(dob_raw),
            safe(dod_raw) if dod_raw else None,
            dead == "1",
            1 if dead == "1" else 0,
            pid,
        ))

        n += 1
        if n % BATCH_SIZE == 0:
            with pg.cursor() as cur:
                cur.executemany(insert_sql, batch)
            pg.commit()
            batch = []
            progress(n, n_total, "person")

    if batch:
        with pg.cursor() as cur:
            cur.executemany(insert_sql, batch)
        pg.commit()

    progress(n_total, n_total, "person")
    print(f"\n    âœ“ {n:,} persons synced")
    return n

# =============================================================================
# STEP 2: VISITS
# =============================================================================

def sync_visits(ydb, pg, source_id, limit=None):
    print("\n[2] Syncing visits â†’ hie.visit_occurrence")

    visit_sql = """
        INSERT INTO hie.visit_occurrence (
            visit_occurrence_id, person_id, source_id, src_hadm_id,
            visit_concept_code, visit_start_datetime, visit_end_datetime,
            visit_type, admit_source, discharge_disposition,
            insurance, language, religion, marital_status, ethnicity,
            admit_diagnosis_text, los_hours, los_days,
            hospital_expire_flag, has_icu_stay,
            ydb_eid, synced_at
        ) VALUES (
            %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW()
        )
        ON CONFLICT (visit_occurrence_id) DO UPDATE SET
            visit_end_datetime   = EXCLUDED.visit_end_datetime,
            discharge_disposition= EXCLUDED.discharge_disposition,
            los_hours            = EXCLUDED.los_hours,
            has_icu_stay         = EXCLUDED.has_icu_stay,
            synced_at            = NOW()
    """

    icu_sql = """
        INSERT INTO hie.visit_detail (
            visit_occurrence_id, person_id, source_id,
            src_icustay_id, care_unit, first_care_unit, last_care_unit,
            intime, outtime, los_hours, db_source, synced_at
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
        ON CONFLICT DO NOTHING
    """

    n_visits = 0
    n_icu    = 0
    pat_n    = 0
    visit_batch = []
    icu_batch   = []

    for pid in get_all_pids(ydb):
        if limit and pat_n >= limit:
            break
        pat_n += 1

        for eid in walk_subs(ydb, "^PHD", [pid, "VISIT"]):
            root = yget(ydb, "^PHD", [pid, "VISIT", eid, "0"])
            if not root:
                continue

            parts = (root + "^^^^^^^").split("^")
            admit_dt   = parse_dt(parts[0])
            disch_dt   = parse_dt(parts[1])
            adm_type   = safe(parts[2], 50)
            adm_loc    = safe(parts[3], 100)
            insurance  = safe(parts[4], 50)
            expire     = parts[5] or "0"
            los_hours_s = parts[7] if len(parts) > 7 else ""

            try:
                los_h = float(los_hours_s) if los_hours_s else None
            except Exception:
                los_h = None

            los_d = round(los_h / 24, 2) if los_h else None
            hadm  = yget(ydb, "^PHD", [pid, "VISIT", eid, "HADM"])
            disch_loc = yget(ydb, "^PHD", [pid, "VISIT", eid, "DISCH"])
            eth   = yget(ydb, "^PHD", [pid, "VISIT", eid, "ETH"])
            lang  = yget(ydb, "^PHD", [pid, "VISIT", eid, "LANG"])
            rel   = yget(ydb, "^PHD", [pid, "VISIT", eid, "REL"])
            mar   = yget(ydb, "^PHD", [pid, "VISIT", eid, "MAR"])
            dx_txt = yget(ydb, "^PHD", [pid, "VISIT", eid, "DX_TEXT"])
            has_icu = yget(ydb, "^PHD", [pid, "VISIT", eid, "HAS_ICU"]) == "1"

            visit_batch.append((
                eid, pid, source_id, safe(hadm),
                safe(adm_type, 50), admit_dt, disch_dt,
                safe(adm_type, 50), safe(adm_loc, 100),
                safe(disch_loc, 100),
                safe(insurance, 50), safe(lang, 20),
                safe(rel, 50), safe(mar, 20), safe(eth, 100),
                safe(dx_txt, 500),
                los_h, los_d,
                int(expire) if expire.isdigit() else 0,
                has_icu, eid,
            ))
            n_visits += 1

            # ICU stays
            for icustay_id in walk_subs(ydb, "^PHD", [pid, "VISIT", eid, "ICU"]):
                icu_root = yget(ydb, "^PHD", [pid, "VISIT", eid, "ICU", icustay_id, "0"])
                last_cu  = yget(ydb, "^PHD", [pid, "VISIT", eid, "ICU", icustay_id, "LAST"])
                if not icu_root:
                    continue
                ip = (icu_root + "^^^^").split("^")
                icu_batch.append((
                    eid, pid, source_id,
                    icustay_id, safe(ip[0], 50), safe(ip[0], 50),
                    safe(last_cu, 50),
                    parse_dt(ip[1]), parse_dt(ip[2]),
                    float(ip[3]) if ip[3] and ip[3] != 'None' else None,
                    safe(ip[4], 20),
                ))
                n_icu += 1

            # Commit in batches
            if len(visit_batch) >= BATCH_SIZE:
                with pg.cursor() as cur:
                    cur.executemany(visit_sql, visit_batch)
                    if icu_batch:
                        cur.executemany(icu_sql, icu_batch)
                pg.commit()
                visit_batch = []
                icu_batch   = []
                progress(pat_n, limit or 46520, "visits")

    if visit_batch:
        with pg.cursor() as cur:
            cur.executemany(visit_sql, visit_batch)
            if icu_batch:
                cur.executemany(icu_sql, icu_batch)
        pg.commit()

    print(f"\n    âœ“ {n_visits:,} visits, {n_icu:,} ICU stays synced")
    return n_visits

# =============================================================================
# STEP 3: CONDITIONS (diagnoses)
# =============================================================================

def sync_conditions(ydb, pg, source_id, limit=None):
    print("\n[3] Syncing conditions â†’ hie.condition_occurrence")

    sql = """
        INSERT INTO hie.condition_occurrence (
            visit_occurrence_id, person_id, source_id,
            condition_concept_code, condition_coding_system,
            condition_type, seq_num, short_title, long_title, synced_at
        ) VALUES (%s,%s,%s,%s,'ICD9CM',%s,%s,%s,%s,NOW())
        ON CONFLICT DO NOTHING
    """

    n = 0
    pat_n = 0
    batch = []

    for pid in get_all_pids(ydb):
        if limit and pat_n >= limit:
            break
        pat_n += 1

        for eid in walk_subs(ydb, "^PHD", [pid, "VISIT"]):
            for seq in walk_subs(ydb, "^PHD", [pid, "VISIT", eid, "DX"]):
                val = yget(ydb, "^PHD", [pid, "VISIT", eid, "DX", seq])
                if not val:
                    continue
                parts = (val + "^^^").split("^")
                code  = safe(parts[0], 20)
                short = safe(parts[1], 100)
                long  = safe(parts[2], 300)
                seq_n = int(seq) if seq.isdigit() else None
                dtype = "primary" if seq_n == 1 else "secondary"

                batch.append((
                    eid, pid, source_id,
                    code, dtype, seq_n, short, long,
                ))
                n += 1

                if len(batch) >= BATCH_SIZE * 5:
                    with pg.cursor() as cur:
                        cur.executemany(sql, batch)
                    pg.commit()
                    batch = []
                    progress(pat_n, limit or 46520, "conditions")

    if batch:
        with pg.cursor() as cur:
            cur.executemany(sql, batch)
        pg.commit()

    print(f"\n    âœ“ {n:,} conditions synced")
    return n

# =============================================================================
# STEP 4: PROCEDURES
# =============================================================================

def sync_procedures(ydb, pg, source_id, limit=None):
    print("\n[4] Syncing procedures â†’ hie.procedure_occurrence")

    sql = """
        INSERT INTO hie.procedure_occurrence (
            visit_occurrence_id, person_id, source_id,
            procedure_concept_code, procedure_coding_system,
            seq_num, short_title, long_title, synced_at
        ) VALUES (%s,%s,%s,%s,'ICD9CM',%s,%s,%s,NOW())
        ON CONFLICT DO NOTHING
    """

    n = 0
    pat_n = 0
    batch = []

    for pid in get_all_pids(ydb):
        if limit and pat_n >= limit:
            break
        pat_n += 1

        for eid in walk_subs(ydb, "^PHD", [pid, "VISIT"]):
            for seq in walk_subs(ydb, "^PHD", [pid, "VISIT", eid, "PROC"]):
                val = yget(ydb, "^PHD", [pid, "VISIT", eid, "PROC", seq])
                if not val:
                    continue
                parts = (val + "^^^").split("^")
                batch.append((
                    eid, pid, source_id,
                    safe(parts[0], 20), int(seq) if seq.isdigit() else None,
                    safe(parts[1], 100), safe(parts[2], 300),
                ))
                n += 1

                if len(batch) >= BATCH_SIZE * 5:
                    with pg.cursor() as cur:
                        cur.executemany(sql, batch)
                    pg.commit()
                    batch = []

    if batch:
        with pg.cursor() as cur:
            cur.executemany(sql, batch)
        pg.commit()

    print(f"\n    âœ“ {n:,} procedures synced")
    return n

# =============================================================================
# STEP 5: DRUG EXPOSURE
# =============================================================================

def sync_drugs(ydb, pg, source_id, limit=None):
    print("\n[5] Syncing drugs â†’ hie.drug_exposure")

    sql = """
        INSERT INTO hie.drug_exposure (
            visit_occurrence_id, person_id, source_id,
            drug_name, drug_name_generic, drug_type,
            ndc, prod_strength, dose_val, dose_unit, route,
            drug_exposure_start, drug_exposure_end, synced_at
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
        ON CONFLICT DO NOTHING
    """

    n = 0
    pat_n = 0
    batch = []

    for pid in get_all_pids(ydb):
        if limit and pat_n >= limit:
            break
        pat_n += 1

        for eid in walk_subs(ydb, "^PHD", [pid, "VISIT"]):
            for idx in walk_subs(ydb, "^PHD", [pid, "VISIT", eid, "MED"]):
                val = yget(ydb, "^PHD", [pid, "VISIT", eid, "MED", idx])
                if not val:
                    continue
                p = (val + "^^^^^^^^^").split("^")
                batch.append((
                    eid, pid, source_id,
                    safe(p[0], 200), safe(p[1], 200), safe(p[2], 30),
                    safe(p[8], 20), safe(p[9], 100),
                    safe(p[3], 50), safe(p[4], 30), safe(p[5], 30),
                    parse_dt(p[6]), parse_dt(p[7]),
                ))
                n += 1

                if len(batch) >= BATCH_SIZE * 5:
                    with pg.cursor() as cur:
                        cur.executemany(sql, batch)
                    pg.commit()
                    batch = []
                    progress(pat_n, limit or 46520, "drugs")

    if batch:
        with pg.cursor() as cur:
            cur.executemany(sql, batch)
        pg.commit()

    print(f"\n    âœ“ {n:,} drug exposures synced")
    return n

# =============================================================================
# STEP 6: MEASUREMENTS (labs)
# =============================================================================

def sync_measurements(ydb, pg, source_id, limit=None):
    print("\n[6] Syncing measurements â†’ hie.measurement")

    sql = """
        INSERT INTO hie.measurement (
            visit_occurrence_id, person_id, source_id,
            src_itemid, measurement_concept,
            value_as_string, value_as_number, unit_concept,
            abnormal_flag, measurement_datetime, synced_at
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
        ON CONFLICT DO NOTHING
    """

    n = 0
    pat_n = 0
    batch = []

    for pid in get_all_pids(ydb):
        if limit and pat_n >= limit:
            break
        pat_n += 1

        for eid in walk_subs(ydb, "^PHD", [pid, "VISIT"]):
            for idx in walk_subs(ydb, "^PHD", [pid, "VISIT", eid, "LAB"]):
                val = yget(ydb, "^PHD", [pid, "VISIT", eid, "LAB", idx])
                if not val:
                    continue
                p = (val + "^^^^^").split("^")
                try:
                    num = float(p[2]) if p[2] else None
                except Exception:
                    num = None

                batch.append((
                    eid, pid, source_id,
                    int(p[0]) if p[0].isdigit() else None,
                    None,
                    safe(p[1], 200), num,
                    safe(p[3], 30), safe(p[4], 20),
                    parse_dt(p[5]),
                ))
                n += 1

                if len(batch) >= BATCH_SIZE * 10:
                    with pg.cursor() as cur:
                        cur.executemany(sql, batch)
                    pg.commit()
                    batch = []
                    progress(pat_n, limit or 46520, "labs")

    if batch:
        with pg.cursor() as cur:
            cur.executemany(sql, batch)
        pg.commit()

    print(f"\n    âœ“ {n:,} measurements synced")
    return n

# =============================================================================
# STEP 7: NOTES
# =============================================================================

def sync_notes(ydb, pg, source_id, limit=None):
    print("\n[7] Syncing notes â†’ hie.note")

    sql = """
        INSERT INTO hie.note (
            visit_occurrence_id, person_id, source_id,
            note_category, note_description, note_datetime,
            cgid, note_text, is_error, synced_at
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,false,NOW())
        ON CONFLICT DO NOTHING
    """

    n = 0
    pat_n = 0
    batch = []

    for pid in get_all_pids(ydb):
        if limit and pat_n >= limit:
            break
        pat_n += 1

        for eid in walk_subs(ydb, "^PHD", [pid, "VISIT"]):
            for idx in walk_subs(ydb, "^PHD", [pid, "VISIT", eid, "NOTE"]):
                meta = yget(ydb, "^PHD", [pid, "VISIT", eid, "NOTE", idx])
                text = yget(ydb, "^PHD", [pid, "VISIT", eid, "NOTETXT", idx])
                if not meta:
                    continue
                p = (meta + "^^^").split("^")
                batch.append((
                    eid, pid, source_id,
                    safe(p[0], 50), safe(p[3], 100),
                    parse_dt(p[1]),
                    int(p[2]) if p[2] and p[2].isdigit() else None,
                    text if text else None,
                ))
                n += 1

                if len(batch) >= 200:
                    with pg.cursor() as cur:
                        cur.executemany(sql, batch)
                    pg.commit()
                    batch = []
                    progress(pat_n, limit or 46520, "notes")

    if batch:
        with pg.cursor() as cur:
            cur.executemany(sql, batch)
        pg.commit()

    print(f"\n    âœ“ {n:,} notes synced")
    return n

# =============================================================================
# ORCHESTRATOR
# =============================================================================

STEPS = [
    ("person",     sync_person,       "^PHD PID â†’ hie.person"),
    ("visits",     sync_visits,       "^PHD VISIT â†’ hie.visit_occurrence + visit_detail"),
    ("conditions", sync_conditions,   "^PHD DX â†’ hie.condition_occurrence"),
    ("procedures", sync_procedures,   "^PHD PROC â†’ hie.procedure_occurrence"),
    ("drugs",      sync_drugs,        "^PHD MED â†’ hie.drug_exposure"),
    ("labs",       sync_measurements, "^PHD LAB â†’ hie.measurement"),
    ("notes",      sync_notes,        "^PHD NOTE â†’ hie.note"),
]

STEP_NAMES = [s[0] for s in STEPS]

def print_summary(results, elapsed):
    print("\n" + "â•" * 58)
    print("  ^PHD â†’ Postgres HIE Sync Summary")
    print("â•" * 58)
    print(f"  {'Step':<14} {'Status':<10} {'Records':>12}  {'Time':>8}")
    print("  " + "â”€" * 54)
    for r in results:
        mark  = "âœ“" if r["status"] == "ok" else "âœ—"
        t     = f"{r['elapsed']:.1f}s"
        count = f"{r['count']:,}" if r.get("count") else "â€”"
        print(f"  {mark} {r['step']:<13} {r['status']:<10} {count:>12}  {t:>8}")
    print("â•" * 58)
    m, s = divmod(int(elapsed), 60)
    print(f"  Total: {m}m {s}s")
    print("â•" * 58)

def main():
    parser = argparse.ArgumentParser(description="^PHD â†’ Postgres HIE sync")
    parser.add_argument("--steps", nargs="+", choices=STEP_NAMES)
    parser.add_argument("--from-step", choices=STEP_NAMES)
    parser.add_argument("--limit", type=int, help="Limit to N patients (for testing)")
    parser.add_argument("--verify-only", action="store_true")
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
            print(f"  {s[0]:<14} {s[2]}")
        if args.limit:
            print(f"  Limit: {args.limit:,} patients")
        return

    print("=" * 58)
    print("  ^PHD â†’ Postgres HIE Sync Engine")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.limit:
        print(f"  Patient limit: {args.limit:,}")
    print("=" * 58)

    ydb = setup_ydb()
    print("âœ“ YottaDB connected")

    pg = get_pg()
    print("âœ“ Postgres connected")

    source_id = get_source_id(pg)
    print(f"âœ“ Source ID: {source_id} (MIII)")

    total_start = time.time()
    results = []

    for step_name, func, desc in steps_to_run:
        print(f"\n{'â”€' * 58}")
        start = time.time()
        result = {"step": step_name, "status": "ok", "count": 0, "elapsed": 0}
        try:
            count = func(ydb, pg, source_id, limit=args.limit)
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
            print("  Stopping.")
            break

    pg.close()
    print_summary(results, time.time() - total_start)

if __name__ == "__main__":
    main()



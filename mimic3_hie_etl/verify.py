"""
verify.py — Verify ^PHD globals and print sample LLM context strings

Run inside the hie-mumps container after etl.py:
  python3 /project/verify.py
  python3 /project/verify.py --pid HIE-MIII-010006
  python3 /project/verify.py --context --n 3
"""

import os
import sys
import argparse

YDB_ENV = {
    "ydb_dist":    "/opt/yottadb/current",
    "ydb_gbldir":  "/data/yottadb.gld",
    "ydb_routines": "/opt/yottadb/current/libyottadbutil.so",
}

def setup_ydb():
    for k, v in YDB_ENV.items():
        os.environ[k] = v
    import yottadb as ydb
    return ydb

def count_patients(ydb):
    n = 0
    pid = ""
    while True:
        pid = ydb.subscript_next("^PHD", [pid])
        if not pid:
            break
        if pid.startswith("HIE-"):
            n += 1
    return n

def get_visits(ydb, pid):
    visits = []
    eid = ""
    while True:
        eid = ydb.subscript_next("^PHD", [pid, "VISIT", eid])
        if not eid:
            break
        visits.append(eid)
    return visits

def count_subnodes(ydb, pid, eid, key):
    n = 0
    sub = ""
    while True:
        sub = ydb.subscript_next("^PHD", [pid, "VISIT", eid, key, sub])
        if not sub:
            break
        n += 1
    return n

def build_context(ydb, pid):
    """Build a structured narrative context string for one patient."""
    lines = []

    # Demographics
    sex  = ydb.get("^PHD", [pid, "PID", "SEX"])  or "?"
    dob  = ydb.get("^PHD", [pid, "PID", "DOB"])  or "?"
    dead = ydb.get("^PHD", [pid, "PID", "DEAD"]) or "0"
    src  = ydb.get("^PHD", [pid, "SRC", "MIII"]) or "?"

    lines.append(f"[PATIENT {pid}]")
    lines.append(f"  Source: MIMIC-III subject {src}")
    lines.append(f"  Sex: {sex}  DOB: {dob}  Deceased: {'Yes' if dead=='1' else 'No'}")
    lines.append("")

    # Encounters
    visits = get_visits(ydb, pid)
    for eid in visits[:3]:  # show max 3 encounters
        root = ydb.get("^PHD", [pid, "VISIT", eid, "0"]) or ""
        parts = (root + "^^^^^^^").split("^")
        admit, disch, adm_type, location, insurance, expire, status, los = parts[:8]

        lines.append(f"  [ENCOUNTER {eid}]")
        lines.append(f"    Admit: {admit}  Discharge: {disch}  Status: {status}")
        lines.append(f"    Type: {adm_type}  Location: {location}")
        lines.append(f"    Insurance: {insurance}  LOS: {los}h")

        # Admit diagnosis text
        dx_text = ydb.get("^PHD", [pid, "VISIT", eid, "DX_TEXT"]) or ""
        if dx_text:
            lines.append(f"    Admit Dx: {dx_text[:100]}")

        # Coded diagnoses (first 5)
        dx_lines = []
        seq = ""
        for _ in range(5):
            seq = ydb.subscript_next("^PHD", [pid, "VISIT", eid, "DX", seq])
            if not seq:
                break
            val = ydb.get("^PHD", [pid, "VISIT", eid, "DX", seq]) or ""
            dx_parts = (val + "^^^").split("^")
            code, short = dx_parts[0], dx_parts[1]
            if code:
                dx_lines.append(f"{code} {short}")
        if dx_lines:
            lines.append(f"    Diagnoses: {' | '.join(dx_lines)}")

        # Labs (first 5)
        lab_lines = []
        idx = ""
        for _ in range(5):
            idx = ydb.subscript_next("^PHD", [pid, "VISIT", eid, "LAB", idx])
            if not idx:
                break
            val = ydb.get("^PHD", [pid, "VISIT", eid, "LAB", idx]) or ""
            lparts = (val + "^^^^^").split("^")
            item, value, valuenum, uom, flag = lparts[0], lparts[1], lparts[2], lparts[3], lparts[4]
            v = valuenum or value
            if v:
                lab_lines.append(f"{item}={v}{uom}{'['+flag+']' if flag else ''}")
        if lab_lines:
            lines.append(f"    Labs: {' | '.join(lab_lines)}")

        # Meds (first 5)
        med_lines = []
        idx = ""
        for _ in range(5):
            idx = ydb.subscript_next("^PHD", [pid, "VISIT", eid, "MED", idx])
            if not idx:
                break
            val = ydb.get("^PHD", [pid, "VISIT", eid, "MED", idx]) or ""
            mparts = (val + "^^^^^").split("^")
            drug, generic, dtype, dose, unit, route = mparts[0], mparts[1], mparts[2], mparts[3], mparts[4], mparts[5]
            name = generic or drug
            if name:
                med_lines.append(f"{name} {dose}{unit} {route}".strip())
        if med_lines:
            lines.append(f"    Meds: {' | '.join(med_lines)}")

        # First note snippet
        note_meta = ydb.get("^PHD", [pid, "VISIT", eid, "NOTE", "1"])
        note_text = ydb.get("^PHD", [pid, "VISIT", eid, "NOTETXT", "1"])
        if note_meta and note_text:
            cat = note_meta.split("^")[0]
            lines.append(f"    Note ({cat}): {note_text[:200].strip()}...")

        lines.append("")

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Verify ^PHD globals")
    parser.add_argument("--pid",     help="Specific patient ID to inspect")
    parser.add_argument("--n",       type=int, default=3,
                        help="Number of sample patients")
    parser.add_argument("--context", action="store_true",
                        help="Print full LLM context strings")
    parser.add_argument("--counts",  action="store_true", default=True,
                        help="Print node counts (default)")
    args = parser.parse_args()

    ydb = setup_ydb()

    print("\n" + "═" * 60)
    print("  ^PHD Global Verification")
    print("═" * 60)

    # Patient count
    print("\n[1] Counting patients...")
    n_patients = count_patients(ydb)
    status = "✓" if n_patients > 0 else "✗"
    print(f"    {status} {n_patients:,} patients in ^PHD")

    if n_patients == 0:
        print("\n    No data found. Run etl.py first.")
        sys.exit(1)

    # Sample patients
    sample_pids = []
    if args.pid:
        sample_pids = [args.pid]
    else:
        pid = ""
        for _ in range(args.n):
            pid = ydb.subscript_next("^PHD", [pid])
            if not pid or not pid.startswith("HIE-"):
                break
            sample_pids.append(pid)

    print(f"\n[2] Sampling {len(sample_pids)} patient(s)...")
    for pid in sample_pids:
        print(f"\n  Patient: {pid}")

        sex  = ydb.get("^PHD", [pid, "PID", "SEX"])  or "?"
        dob  = ydb.get("^PHD", [pid, "PID", "DOB"])  or "?"
        dead = ydb.get("^PHD", [pid, "PID", "DEAD"]) or "?"
        src  = ydb.get("^PHD", [pid, "SRC", "MIII"]) or "?"
        print(f"    sex={sex}  dob={dob}  dead={dead}  mimic_id={src}")

        visits = get_visits(ydb, pid)
        print(f"    encounters: {len(visits)}")

        for eid in visits[:2]:
            n_dx   = count_subnodes(ydb, pid, eid, "DX")
            n_lab  = count_subnodes(ydb, pid, eid, "LAB")
            n_med  = count_subnodes(ydb, pid, eid, "MED")
            n_note = count_subnodes(ydb, pid, eid, "NOTE")
            n_icu  = count_subnodes(ydb, pid, eid, "ICU")
            print(f"      {eid}: dx={n_dx} labs={n_lab} meds={n_med} notes={n_note} icu={n_icu}")

    # Reverse index check
    print("\n[3] Checking BSRC reverse index...")
    test_ids = ["10006", "22341"]
    for sid in test_ids:
        pid = ydb.subscript_next("^PHD", ["BSRC", "MIII", sid, ""])
        status = "✓" if pid else "?"
        print(f"    {status} MIMIC {sid:>8} → {pid or 'not found'}")

    # Context strings
    if args.context:
        print(f"\n[4] Sample LLM context strings ({len(sample_pids)} patients)...")
        for pid in sample_pids:
            print("\n" + "─" * 60)
            ctx = build_context(ydb, pid)
            print(ctx)
            print(f"  Token estimate: ~{len(ctx)//4:,} tokens")

    print("\n" + "═" * 60)
    print("  Verification complete")
    print("═" * 60 + "\n")

if __name__ == "__main__":
    main()

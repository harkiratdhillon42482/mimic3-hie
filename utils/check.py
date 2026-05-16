import os
os.environ["ydb_gbldir"] = "/data/r2.06_x86_64/g/yottadb.gld"
os.chdir("/data/r2.06_x86_64/g")
import yottadb as ydb

def s(val):
    if val is None: return ""
    return val.decode() if isinstance(val, bytes) else str(val)

def nxt(gbl, subs):
    try:
        return ydb.subscript_next(gbl, subs)
    except Exception:
        return None

n_pat = 0
pid = ""
while True:
    pid = nxt("^PHD", [pid])
    if not pid: break
    if s(pid).startswith("HIE-"): n_pat += 1
print(f"Patients in ^PHD: {n_pat:,}")

pid = ""
while True:
    pid = nxt("^PHD", [pid])
    if not pid: break
    if s(pid).startswith("HIE-"): break

pid_s = s(pid)
print(f"First patient: {pid_s}")
print(f"  SEX:  {s(ydb.get(chr(94)+'PHD', [pid_s, 'PID', 'SEX']))}")
print(f"  DOB:  {s(ydb.get(chr(94)+'PHD', [pid_s, 'PID', 'DOB']))}")
print(f"  SRC:  {s(ydb.get(chr(94)+'PHD', [pid_s, 'SRC', 'MIII']))}")

n_enc, eid, first_eid = 0, "", None
while True:
    eid = nxt(chr(94)+"PHD", [pid_s, "VISIT", eid])
    if not eid: break
    if first_eid is None: first_eid = s(eid)
    n_enc += 1
print(f"  Encounters: {n_enc}")

if first_eid:
    print(f"  First enc: {first_eid}")
    for key in ["DX","MED","LAB","PROC","ICU"]:
        n, sub = 0, ""
        while True:
            sub = nxt(chr(94)+"PHD", [pid_s, "VISIT", first_eid, key, sub])
            if not sub: break
            n += 1
        print(f"    {key}: {n}")

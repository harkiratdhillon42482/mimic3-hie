# Architecture

## Three Layers

1. YottaDB MUMPS - canonical HIE store (^PHD globals)
2. PostgreSQL - OMOP-aligned analytics layer (hie.* schema)
3. Consumers - ML models, LLM training, dashboards

## ^PHD Global Structure

^PHD(pid, "PID", "SEX")              M/F/U
^PHD(pid, "PID", "DOB")              YYYYMMDDHHMMSS
^PHD(pid, "VISIT", eid, "0")         admit^disch^type^los
^PHD(pid, "VISIT", eid, "DX", n)     icd9^short^long^seq
^PHD(pid, "VISIT", eid, "MED", n)    drug^dose^route^start^stop
^PHD(pid, "VISIT", eid, "LAB", n)    itemid^value^uom^flag^dt
^PHD(pid, "VISIT", eid, "NOTE", n)   category^dt^cgid
^PHD(pid, "VISIT", eid, "NOTETXT",n) full note text
^PHD(pid, "VISIT", eid, "ICU", id)   unit^intime^outtime^los
^PHD("BSRC","MIII",src_id,pid)       reverse lookup

## Why MUMPS

- Zero join cost: all patient data co-located under ^PHD(pid)
- O(log N) point lookups via native B-tree
- Schema-less: multiple sources load without collision
- LLM-ready: one traversal produces complete patient context

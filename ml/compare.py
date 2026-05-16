"""
compare.py — Compare MUMPS vs Postgres ML results
Reads from hie.model_result and prints side-by-side comparison

Run in pgAdmin or inside container:
  python3 /project/compare.py
"""

import psycopg2
import json

PG_CONFIG = {
    "host":     "host.docker.internal",
    "port":     5432,
    "dbname":   "MIMICold",
    "user":     "postgres",
    "password": "Panjwar4633",
}

def main():
    pg = psycopg2.connect(**PG_CONFIG)
    cur = pg.cursor()

    # Get latest run from each source
    cur.execute("""
        SELECT DISTINCT ON (data_source)
            run_id, data_source, model_type, target,
            n_total, n_train, n_test, n_features, positive_rate,
            auc_roc, accuracy, precision_score, recall_score, f1_score,
            feature_time_s, train_time_s, predict_time_s, total_time_s,
            top_features_json, created_at
        FROM hie.model_result
        WHERE target = '30day_readmission'
        ORDER BY data_source, created_at DESC
    """)

    rows = {row[1]: row for row in cur.fetchall()}

    if not rows:
        print("No results found. Run readmission_pg.py and readmission_ydb.py first.")
        return

    pg_row  = rows.get("POSTGRES")
    ydb_row = rows.get("MUMPS")

    print("\n" + "═" * 65)
    print("  30-Day Readmission — MUMPS vs Postgres Comparison")
    print("═" * 65)

    # Header
    print(f"\n  {'Metric':<30} {'Postgres':>12} {'MUMPS':>12} {'Winner':>10}")
    print("  " + "─" * 61)

    def fmt(val, pct=False):
        if val is None: return "—"
        if pct: return f"{float(val):.2%}"
        return f"{float(val):.4f}"

    def winner(pg_val, ydb_val, higher_is_better=True):
        if pg_val is None or ydb_val is None: return "—"
        pg_f  = float(pg_val)
        ydb_f = float(ydb_val)
        diff  = abs(pg_f - ydb_f)
        if diff < 0.001: return "tie"
        if higher_is_better:
            return "Postgres" if pg_f > ydb_f else "MUMPS"
        else:
            return "Postgres" if pg_f < ydb_f else "MUMPS"

    # Dataset stats
    print(f"\n  {'── Dataset ──'}")
    if pg_row:
        print(f"  {'Total admissions':<30} {pg_row[4]:>12,}")
        print(f"  {'Train / Test':<30} {str(pg_row[5])+'/'+str(pg_row[6]):>12}")
        print(f"  {'Features':<30} {pg_row[7]:>12,}")
        print(f"  {'Positive rate (readmit)':<30} {fmt(pg_row[8], pct=True):>12}")

    # Accuracy metrics
    print(f"\n  {'── Accuracy (same model) ──'}")
    metrics = [
        ("AUC-ROC",    9,  True),
        ("Accuracy",   10, True),
        ("Precision",  11, True),
        ("Recall",     12, True),
        ("F1 Score",   13, True),
    ]
    for label, idx, hib in metrics:
        pg_v  = pg_row[idx]  if pg_row  else None
        ydb_v = ydb_row[idx] if ydb_row else None
        w     = winner(pg_v, ydb_v, hib)
        print(f"  {label:<30} {fmt(pg_v):>12} {fmt(ydb_v):>12} {w:>10}")

    # Speed metrics — THE KEY COMPARISON
    print(f"\n  {'── Speed (feature extraction) ──'}")
    speed_metrics = [
        ("Feature extraction (s)", 14, False),
        ("Train time (s)",         15, False),
        ("Predict time (s)",       16, False),
        ("Total time (s)",         17, False),
    ]
    for label, idx, hib in speed_metrics:
        pg_v  = pg_row[idx]  if pg_row  else None
        ydb_v = ydb_row[idx] if ydb_row else None
        w     = winner(pg_v, ydb_v, hib)
        print(f"  {label:<30} {fmt(pg_v):>12} {fmt(ydb_v):>12} {w:>10}")

    # Speedup calculation
    if pg_row and ydb_row and pg_row[14] and ydb_row[14]:
        pg_feat  = float(pg_row[14])
        ydb_feat = float(ydb_row[14])
        if ydb_feat > 0:
            speedup = pg_feat / ydb_feat
            print(f"\n  {'Feature extraction speedup':<30} {'':>12} {speedup:>11.1f}x")

    # Top features
    print(f"\n  {'── Top predictive features ──'}")
    if pg_row and pg_row[18]:
        try:
            feats = json.loads(pg_row[18]) if isinstance(pg_row[18], str) else pg_row[18]
            print(f"  Postgres top 5:")
            for i, (feat, coef) in enumerate(list(feats.items())[:5]):
                direction = "↑ risk" if float(coef) > 0 else "↓ risk"
                print(f"    {i+1}. {feat:<28} {coef:+.4f}  ({direction})")
        except Exception:
            pass

    if ydb_row and ydb_row[18]:
        try:
            feats = json.loads(ydb_row[18]) if isinstance(ydb_row[18], str) else ydb_row[18]
            print(f"\n  MUMPS top 5:")
            for i, (feat, coef) in enumerate(list(feats.items())[:5]):
                direction = "↑ risk" if float(coef) > 0 else "↓ risk"
                print(f"    {i+1}. {feat:<28} {coef:+.4f}  ({direction})")
        except Exception:
            pass

    # Verdict
    print(f"\n  {'── Verdict ──'}")
    if pg_row and ydb_row:
        auc_diff   = abs(float(pg_row[9] or 0) - float(ydb_row[9] or 0))
        feat_pg    = float(pg_row[14] or 0)
        feat_ydb   = float(ydb_row[14] or 0)

        if auc_diff < 0.005 and feat_ydb < feat_pg:
            speedup = feat_pg / feat_ydb if feat_ydb > 0 else 0
            print(f"  ✓ MUMPS: Same accuracy (AUC diff={auc_diff:.4f}), "
                  f"{speedup:.1f}x faster feature extraction")
            print(f"  → Architecture validated: MUMPS is the right data engine")
        elif auc_diff >= 0.005:
            print(f"  ⚠ AUC difference: {auc_diff:.4f} — check feature extraction parity")
        else:
            print(f"  → Results comparable, timing may vary by run")

    print("\n" + "═" * 65 + "\n")

    # Also query the view if it exists
    try:
        cur.execute("SELECT * FROM hie.v_model_comparison")
        rows_view = cur.fetchall()
        if rows_view:
            print("  hie.v_model_comparison:")
            for r in rows_view:
                print(f"    {r}")
    except Exception:
        pass

    pg.close()

if __name__ == "__main__":
    main()

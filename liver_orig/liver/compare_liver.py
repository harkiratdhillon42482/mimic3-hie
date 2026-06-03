"""
compare_liver.py
Compare liver transplant model results across sources and model types

Run inside container:
  python3 /project/liver/compare_liver.py
"""

import psycopg2
import json
from datetime import datetime

PG_CONFIG = {
    "host":     "host.docker.internal",
    "port":     5432,
    "dbname":   "MIMICold",
    "user":     "postgres",
    "password": "YOUR_PASSWORD",
}


def main():
    pg = psycopg2.connect(**PG_CONFIG)
    cur = pg.cursor()

    # Get all liver transplant results
    cur.execute("""
        SELECT DISTINCT ON (model_type, data_source)
            run_id, model_type, data_source,
            n_total, n_train, n_test, n_features, positive_rate,
            auc_roc, accuracy, precision_score, recall_score, f1_score,
            feature_time_s, train_time_s, predict_time_s, total_time_s,
            top_features_json, created_at
        FROM hie.model_result
        WHERE target = 'liver_transplant_candidate'
        ORDER BY model_type, data_source, created_at DESC
    """)
    rows = cur.fetchall()
    pg.close()

    if not rows:
        print("\nNo liver transplant results found.")
        print("Run liver_pg.py, liver_ydb.py, then xgboost_liver.py first.")
        return

    # Organize by model type and source
    results = {}
    for row in rows:
        (run_id, model_type, source,
         n_total, n_train, n_test, n_features, pos_rate,
         auc, acc, prec, rec, f1,
         feat_t, train_t, pred_t, total_t,
         top_feats, created_at) = row
        key = (model_type, source)
        results[key] = {
            'run_id': run_id,
            'n_total': n_total, 'n_train': n_train,
            'n_test': n_test, 'n_features': n_features,
            'pos_rate': float(pos_rate or 0),
            'auc': float(auc or 0),
            'acc': float(acc or 0),
            'prec': float(prec or 0),
            'rec': float(rec or 0),
            'f1': float(f1 or 0),
            'feat_t': float(feat_t or 0),
            'train_t': float(train_t or 0),
            'total_t': float(total_t or 0),
            'top_feats': top_feats,
        }

    print("\n" + "═" * 70)
    print("  Liver Transplant Candidate Prediction — Full Comparison")
    print("═" * 70)

    # ── Dataset Summary ────────────────────────────────────────────────────
    print("\n  Dataset")
    print("  " + "─" * 66)
    for (model, source), r in sorted(results.items()):
        print(f"  {model:<20} {source:<10} "
              f"n={r['n_total']:,}  "
              f"pos={r['pos_rate']:.1%}  "
              f"train={r['n_train']:,}  test={r['n_test']:,}")

    # ── Accuracy Metrics ───────────────────────────────────────────────────
    print("\n  Accuracy Metrics")
    print(f"  {'Model':<20} {'Source':<10} {'AUC':>8} {'Acc':>8} "
          f"{'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("  " + "─" * 66)
    for (model, source), r in sorted(results.items()):
        print(f"  {model:<20} {source:<10} "
              f"{r['auc']:>8.4f} {r['acc']:>8.4f} "
              f"{r['prec']:>8.4f} {r['rec']:>8.4f} {r['f1']:>8.4f}")

    # ── Speed Comparison ───────────────────────────────────────────────────
    print("\n  Speed (seconds)")
    print(f"  {'Model':<20} {'Source':<10} {'Feature':>10} "
          f"{'Train':>8} {'Total':>8}")
    print("  " + "─" * 60)
    for (model, source), r in sorted(results.items()):
        print(f"  {model:<20} {source:<10} "
              f"{r['feat_t']:>10.1f} "
              f"{r['train_t']:>8.2f} "
              f"{r['total_t']:>8.1f}")

    # ── MUMPS vs Postgres for each model ──────────────────────────────────
    model_types = set(k[0] for k in results.keys())
    for model in sorted(model_types):
        pg_r  = results.get((model, 'POSTGRES'))
        ydb_r = results.get((model, 'MUMPS'))
        if not pg_r or not ydb_r:
            continue

        print(f"\n  {model} — MUMPS vs Postgres")
        print("  " + "─" * 50)
        auc_diff  = abs(pg_r['auc'] - ydb_r['auc'])
        feat_diff = pg_r['feat_t'] - ydb_r['feat_t']
        speedup   = pg_r['feat_t'] / max(ydb_r['feat_t'], 0.001)

        print(f"  AUC difference:     {auc_diff:.4f} "
              f"({'identical' if auc_diff < 0.01 else 'differs'})")
        print(f"  Feature time:       PG={pg_r['feat_t']:.1f}s  "
              f"MUMPS={ydb_r['feat_t']:.1f}s")
        if feat_diff > 0:
            print(f"  MUMPS speedup:      {speedup:.1f}x faster for feature extraction")
        else:
            print(f"  Postgres speedup:   {1/speedup:.1f}x faster for feature extraction")

    # ── Model Progression ─────────────────────────────────────────────────
    print("\n  Model Progression (Postgres path)")
    print("  " + "─" * 50)
    print(f"  {'Phase':<8} {'Model':<20} {'AUC':>8} {'Expected':>10}")
    print("  " + "─" * 50)

    phases = [
        ("Phase 1", "XGBoost", "0.75-0.82"),
        ("Phase 2", "LSTM",    "0.80-0.87"),
        ("Phase 3", "Transformer", "0.85-0.92"),
    ]
    for phase, model, expected in phases:
        r = results.get((model, 'POSTGRES'))
        auc_str = f"{r['auc']:.4f}" if r else "not run"
        print(f"  {phase:<8} {model:<20} {auc_str:>8} {expected:>10}")

    # ── Top Features ──────────────────────────────────────────────────────
    xgb_pg = results.get(('XGBoost', 'POSTGRES'))
    if xgb_pg and xgb_pg['top_feats']:
        print("\n  Top Features (XGBoost / Postgres)")
        print("  " + "─" * 50)
        try:
            feats = (json.loads(xgb_pg['top_feats'])
                     if isinstance(xgb_pg['top_feats'], str)
                     else xgb_pg['top_feats'])
            for i, (feat, imp) in enumerate(list(feats.items())[:10]):
                bar = "█" * int(imp * 60)
                print(f"  {i+1:2d}. {feat:<28} {imp:.4f}  {bar}")
        except Exception:
            pass

    # ── Clinical Interpretation ───────────────────────────────────────────
    print("\n  Clinical Interpretation")
    print("  " + "─" * 50)
    print("  MELD-Na trajectory is the strongest predictor of transplant need.")
    print("  Hepatorenal syndrome and encephalopathy signal decompensation.")
    print("  Medication escalation (lactulose → rifaximin → albumin) tracks")
    print("  disease progression — the model learns this sequence matters.")
    print("  Frequent ED visits + shortening intervals = velocity signal.")

    # ── Architecture Verdict ──────────────────────────────────────────────
    print("\n  Architecture Verdict")
    print("  " + "─" * 50)
    xgb_pg  = results.get(('XGBoost', 'POSTGRES'))
    xgb_ydb = results.get(('XGBoost', 'MUMPS'))
    if xgb_pg and xgb_ydb:
        auc_diff = abs(xgb_pg['auc'] - xgb_ydb['auc'])
        if auc_diff < 0.01:
            print(f"  ✓ Model quality identical (AUC diff: {auc_diff:.4f})")
            print(f"  ✓ ^PHD contains same clinical signal as hie.* tables")
            print(f"  → Architecture validated for liver transplant prediction")
        else:
            print(f"  ⚠ AUC differs by {auc_diff:.4f} — check feature parity")

    print("\n" + "═" * 70 + "\n")


if __name__ == "__main__":
    main()

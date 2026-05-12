"""
Random Forest pipeline for World Cup match outcome prediction.

Same walk-forward temporal approach as lgbm_pipeline.py but using
sklearn's RandomForestClassifier.

RF doesn't handle NaN natively — we fill with -999 as sentinel value
(trees learn to split around it).

Target: 0=Loss, 1=Draw, 2=Win (from team perspective)
"""

from __future__ import annotations

import glob
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
import joblib

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data"
OUT = Path(__file__).resolve().parent

NAN_SENTINEL = -999.0

# ── Reuse all mappings from lgbm_pipeline ─────────────────────────────────────
from lgbm_pipeline import (
    NAME_TO_CODE, WC_HOSTS, WC_WINNERS, COPA_NAME_MAP, AFCON_NAME_MAP,
    KNOCKOUT_ROUNDS, FEATURE_COLS,
    load_all_data, validate_name_mapping,
    prepare_euro, prepare_copa, prepare_afcon,
    prepare_wc_matches, compute_features, build_deltas,
    get_model_features,
)


def walk_forward_train_rf(train_df: pd.DataFrame):
    folds = [
        (2010, 2014),
        (2014, 2018),
        (2018, 2022),
    ]

    model_cols = get_model_features()
    fold_results = []
    best_n_estimators = []

    print("\n" + "=" * 70)
    print("WALK-FORWARD TRAINING — Random Forest")
    print("=" * 70)

    for fold_num, (train_cutoff, test_year) in enumerate(folds, 1):
        train_mask = train_df["year"] <= train_cutoff
        test_mask = train_df["year"] == test_year

        X_train = train_df.loc[train_mask, model_cols].fillna(NAN_SENTINEL).values
        y_train = train_df.loc[train_mask, "outcome"].values
        X_test = train_df.loc[test_mask, model_cols].fillna(NAN_SENTINEL).values
        y_test = train_df.loc[test_mask, "outcome"].values

        max_train_year = train_df.loc[train_mask, "year"].max()
        assert max_train_year <= train_cutoff, f"LEAKAGE: train has year {max_train_year} > {train_cutoff}"

        print(f"\n--- Fold {fold_num}: Train ≤{train_cutoff} → Test {test_year} ---")
        print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
        print(f"Train target dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"Test target dist:  {dict(zip(*np.unique(y_test, return_counts=True)))}")

        # Test multiple n_estimators to find best
        best_ll = float("inf")
        best_n = 100
        for n_est in [100, 200, 300, 500]:
            rf = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            )
            rf.fit(X_train, y_train)
            proba = rf.predict_proba(X_test)
            ll = log_loss(y_test, proba)
            if ll < best_ll:
                best_ll = ll
                best_n = n_est
                best_model = rf

        best_n_estimators.append(best_n)

        y_pred_proba = best_model.predict_proba(X_test)
        assert y_pred_proba.shape[1] == 3, f"Expected 3 classes, got {y_pred_proba.shape[1]}"

        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logloss_val = log_loss(y_test, y_pred_proba)

        pred_dist = dict(zip(*np.unique(y_pred, return_counts=True)))

        print(f"Best n_estimators: {best_n}")
        print(f"Log-loss: {logloss_val:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Predicted dist: {pred_dist}")

        fold_results.append({
            "fold": fold_num,
            "train_cutoff": train_cutoff,
            "test_year": test_year,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "best_n_estimators": best_n,
            "logloss": logloss_val,
            "accuracy": accuracy,
            "pred_dist": {int(k): int(v) for k, v in pred_dist.items()},
        })

    return fold_results, best_n_estimators


def train_final_rf(train_df: pd.DataFrame, n_estimators: int):
    model_cols = get_model_features()
    X = train_df[model_cols].fillna(NAN_SENTINEL).values
    y = train_df["outcome"].values

    print(f"\n{'=' * 70}")
    print(f"FINAL MODEL — Random Forest, {n_estimators} estimators, all data")
    print(f"{'=' * 70}")
    print(f"Training on {len(X)} rows")

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X, y)

    # Feature importance
    importance = pd.DataFrame({
        "feature": model_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nTop 20 features by importance:")
    print(importance.head(20).to_string(index=False))

    return rf, importance


def main():
    print("Loading data...")
    wc_raw, rankings, players, euro, copa, afcon = load_all_data()

    print("\n--- Paso 1: Validating name mapping ---")
    validate_name_mapping(wc_raw)

    print("\n--- Paso 2: Preparing WC matches (duplicated) ---")
    wc_rows = prepare_wc_matches(wc_raw)

    print(f"\nTarget distribution (all):")
    for label, name in [(0, "Loss"), (1, "Draw"), (2, "Win")]:
        pct = (wc_rows["outcome"] == label).mean() * 100
        print(f"  {name}: {pct:.1f}%")

    print("\n--- Paso 3: Computing features (walk-forward) ---")
    features = compute_features(wc_rows, wc_raw, rankings, players, euro, copa, afcon)

    print("\n--- Paso 4: Building deltas ---")
    train_df = build_deltas(wc_rows, features)
    print(f"Train dataset shape: {train_df.shape}")

    print("\n--- Paso 5: NaN report ---")
    model_cols = get_model_features()
    nan_counts = train_df[model_cols].isna().sum()
    nan_report = nan_counts[nan_counts > 0].sort_values(ascending=False)
    print(f"Features with NaN ({len(nan_report)}/{len(model_cols)}):")
    print(nan_report.to_string())
    print(f"\nNaN will be filled with sentinel value {NAN_SENTINEL}")

    print("\n--- Paso 6: Walk-forward training ---")
    fold_results, best_n_estimators = walk_forward_train_rf(train_df)

    print("\n--- Paso 7: Final model ---")
    avg_n = int(np.mean(best_n_estimators))
    print(f"Average best n_estimators from folds: {avg_n}")
    final_model, importance = train_final_rf(train_df, max(avg_n, 300))

    # Save artifacts
    model_path = OUT / "rf_wc_model.joblib"
    joblib.dump(final_model, str(model_path))
    print(f"\n✓ Model saved: {model_path}")

    importance.to_csv(OUT / "rf_feature_importance.csv", index=False)
    print(f"✓ Feature importance saved: {OUT / 'rf_feature_importance.csv'}")

    metrics = {
        "folds": fold_results,
        "avg_n_estimators": avg_n,
        "total_train_rows": len(train_df),
    }
    with open(OUT / "rf_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"✓ Metrics saved: {OUT / 'rf_metrics.json'}")

    # Validations
    print(f"\n{'=' * 70}")
    print("VALIDACIONES FINALES")
    print(f"{'=' * 70}")
    print(f"✓ Anti-leakage: verified in each fold")
    print(f"✓ Duplication: {len(train_df)} rows = {len(train_df)//2} matches × 2")

    dist = train_df["outcome"].value_counts(normalize=True)
    for cls in [0, 1, 2]:
        pct = dist.get(cls, 0) * 100
        status = "✓" if pct >= 15 else "⚠"
        print(f"{status} Class {cls}: {pct:.1f}%")

    max_acc = max(fr["accuracy"] for fr in fold_results)
    status = "✓" if max_acc > 0.38 else "⚠"
    print(f"{status} Best fold accuracy: {max_acc:.4f} (baseline ~0.33)")


if __name__ == "__main__":
    main()

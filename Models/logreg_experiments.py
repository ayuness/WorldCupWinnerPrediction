"""Sweep three LogReg variants and compare them against the v2 baseline:

- `v2_drop_market`      : drop `delta_squad_market_value_log` only.
- `v2_calibrated`       : keep all 22 features, wrap in isotonic CalibratedClassifierCV.
- `v2_drop_calibrated`  : both.

Each variant reuses the feature-engineering pipeline from `current_model_pipeline`
and the Monte Carlo simulator from `compare_models`, exactly like `logreg_model`.
Optuna tunes the *uncalibrated* base inside each variant (calibration as a wrapper
doesn't move the optimal hyperparams meaningfully and would 5x the fit count).

Feature ablation is implemented as **value zeroing**, not column dropping:
we set the feature to 0 in `train_df` AND in `team_features_2026`. This keeps
the existing `precompute_neutral_probabilities` working unchanged: it computes
the delta for that feature as 0 everywhere, which is mathematically equivalent
to removing it from the regression (the model learns coef=0 with zero variance).
The benefit is no duplication of the 70-line `make_match_features` helper."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss

try:
    from .current_model_pipeline import (
        PreparedCurrentExperiment,
        prepare_2026_team_features,
        prepare_experiment,
        precompute_neutral_probabilities,
    )
    from .compare_models import multiclass_brier_score, simulate_from_caches
    from .logreg_model import (
        ARTIFACTS_DIR as LOGREG_ARTIFACTS_DIR,
        INNER_YEAR_MAX,
        N_SIMULATIONS,
        N_TRIALS,
        SEED,
        _fit_with_warning_check,
        build_dataset,
        extract_coefficients,
        make_estimator,
    )
    from . import logreg_model as _logreg_module
except ImportError:
    from current_model_pipeline import (  # type: ignore[no-redef]
        PreparedCurrentExperiment,
        prepare_2026_team_features,
        prepare_experiment,
        precompute_neutral_probabilities,
    )
    from compare_models import multiclass_brier_score, simulate_from_caches  # type: ignore[no-redef]
    from logreg_model import (  # type: ignore[no-redef]
        ARTIFACTS_DIR as LOGREG_ARTIFACTS_DIR,
        INNER_YEAR_MAX,
        N_SIMULATIONS,
        N_TRIALS,
        SEED,
        _fit_with_warning_check,
        build_dataset,
        extract_coefficients,
        make_estimator,
    )
    import logreg_model as _logreg_module  # type: ignore[no-redef]


EXPERIMENTS_DIR = LOGREG_ARTIFACTS_DIR / "experiments"
SUMMARY_PATH = LOGREG_ARTIFACTS_DIR / "experiments_summary.csv"
CHAMPION_PATH = LOGREG_ARTIFACTS_DIR / "experiments_champion_pct.csv"

# Maps delta_<feature> name -> key in team_features_2026 (used by precompute_neutral_probabilities)
DELTA_TO_TEAM_KEY = {
    "delta_elo": "elo",
    "delta_form_wp": "form_wp",
    "delta_form_gf": "form_gf",
    "delta_form_gc": "form_gc",
    "delta_wc_win_pct": "wc_win_pct",
    "delta_wc_gc_per_game": "wc_gc_per_game",
    "delta_wc_titles": "wc_titles",
    "delta_wc_experience": "wc_experience",
    "delta_rank_strength": "rank_strength",
    "delta_rank_volatility": "rank_volatility",
    "delta_squad_market_value_log": "squad_market_value_log",
    "delta_squad_avg_age": "squad_avg_age",
    "delta_star_concentration": "star_concentration",
    "delta_top5_avg_value": "top5_avg_value",
    "delta_gk_value": "gk_value",
    "delta_squad_depth_ratio": "squad_depth_ratio",
    "delta_age_weighted_value": "age_weighted_value",
    "delta_top_league_pct": "top_league_pct",
}

VARIANTS = [
    {"name": "v2_drop_market", "drop_features": ["delta_squad_market_value_log"], "calibrate": None},
    {"name": "v2_calibrated", "drop_features": [], "calibrate": "isotonic"},
    {
        "name": "v2_drop_calibrated",
        "drop_features": ["delta_squad_market_value_log"],
        "calibrate": "isotonic",
    },
]


def ablate_features(
    experiment: PreparedCurrentExperiment,
    team_features_2026: dict[str, dict[str, float]],
    drop_features: list[str],
) -> tuple[PreparedCurrentExperiment, dict[str, dict[str, float]]]:
    """Zero out specified delta-features in both train_df and team_features_2026."""
    if not drop_features:
        return experiment, team_features_2026

    new_df = experiment.train_df.copy()
    new_team_features = {team: dict(feats) for team, feats in team_features_2026.items()}
    for delta_col in drop_features:
        if delta_col not in DELTA_TO_TEAM_KEY:
            raise KeyError(f"Unknown delta feature: {delta_col}")
        team_key = DELTA_TO_TEAM_KEY[delta_col]
        new_df[delta_col] = 0.0
        for team in new_team_features:
            new_team_features[team][team_key] = 0.0
    return replace(experiment, train_df=new_df), new_team_features


def tune_uncalibrated(
    experiment: PreparedCurrentExperiment,
    n_trials: int = N_TRIALS,
    inner_year_max: int = INNER_YEAR_MAX,
    seed: int = SEED,
) -> tuple[dict[str, float], optuna.Study]:
    X = experiment.train_df[experiment.feature_cols].to_numpy()
    y = experiment.train_df["outcome"].to_numpy()
    inner_folds = [fold for fold in experiment.folds if fold[2] <= inner_year_max]
    if not inner_folds:
        raise ValueError("No inner folds for tuning.")

    def objective(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", 1e-4, 100.0, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        losses = []
        for train_mask, test_mask, _ in inner_folds:
            est = make_estimator(C=C, l1_ratio=l1_ratio)
            _fit_with_warning_check(est, X[train_mask], y[train_mask])
            y_proba = est.predict_proba(X[test_mask])
            losses.append(log_loss(y[test_mask], y_proba, labels=[0, 1, 2]))
        return float(np.mean(losses))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {
        "C": float(study.best_params["C"]),
        "l1_ratio": float(study.best_params["l1_ratio"]),
    }, study


def _estimator_for(best_params: dict[str, float], calibrate: str | None):
    base = make_estimator(C=best_params["C"], l1_ratio=best_params["l1_ratio"])
    if calibrate is None:
        return base
    return CalibratedClassifierCV(base, method=calibrate, cv=5)


def evaluate_loto(
    experiment: PreparedCurrentExperiment,
    best_params: dict[str, float],
    calibrate: str | None,
) -> pd.DataFrame:
    X = experiment.train_df[experiment.feature_cols].to_numpy()
    y = experiment.train_df["outcome"].to_numpy()
    rows = []
    for train_mask, test_mask, test_year in experiment.folds:
        est = _estimator_for(best_params, calibrate)
        _fit_with_warning_check(est, X[train_mask], y[train_mask])
        y_proba = est.predict_proba(X[test_mask])
        y_pred = est.predict(X[test_mask])
        rows.append(
            {
                "test_year": test_year,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "log_loss": float(log_loss(y[test_mask], y_proba, labels=[0, 1, 2])),
                "accuracy": float(accuracy_score(y[test_mask], y_pred)),
                "brier": multiclass_brier_score(y[test_mask], y_proba),
            }
        )
    return pd.DataFrame(rows)


def _add_mean_row(cv_metrics: pd.DataFrame) -> pd.DataFrame:
    mean_row = pd.DataFrame(
        [
            {
                "test_year": "mean",
                "n_train": float(cv_metrics["n_train"].mean()),
                "n_test": int(cv_metrics["n_test"].sum()),
                "log_loss": float(cv_metrics["log_loss"].mean()),
                "accuracy": float(cv_metrics["accuracy"].mean()),
                "brier": float(cv_metrics["brier"].mean()),
            }
        ]
    )
    return pd.concat([cv_metrics, mean_row], ignore_index=True)


def run_variant(
    name: str,
    base_experiment: PreparedCurrentExperiment,
    base_team_features: dict[str, dict[str, float]],
    drop_features: list[str],
    calibrate: str | None,
    n_trials: int = N_TRIALS,
    n_simulations: int = N_SIMULATIONS,
) -> dict[str, object]:
    out_dir = EXPERIMENTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[VARIANT] {name} (drop={drop_features}, calibrate={calibrate}) ...")
    experiment, team_features_2026 = ablate_features(
        base_experiment, base_team_features, drop_features
    )
    X, y = build_dataset(experiment)

    print(f"  [tune] Optuna {n_trials} trials (uncalibrated base) ...")
    best_params, study = tune_uncalibrated(experiment, n_trials=n_trials)
    print(f"    best_params={best_params}, best_inner_log_loss={study.best_value:.4f}")

    print("  [LOTO] outer CV ...")
    cv_metrics = evaluate_loto(experiment, best_params, calibrate)
    cv_full = _add_mean_row(cv_metrics)
    cv_full.to_csv(out_dir / "cv_metrics.csv", index=False)
    print(
        f"    mean log_loss={cv_metrics['log_loss'].mean():.4f} "
        f"acc={cv_metrics['accuracy'].mean():.4f} brier={cv_metrics['brier'].mean():.4f}"
    )

    print("  [train] final model on full dataset ...")
    pipeline = _estimator_for(best_params, calibrate)
    _fit_with_warning_check(pipeline, X, y)
    joblib.dump(pipeline, out_dir / "pipeline.pkl")

    if calibrate is None:
        coefs = extract_coefficients(pipeline, experiment.feature_cols)
        coefs.to_csv(out_dir / "coefficients.csv", index=False)
    else:
        # CalibratedClassifierCV doesn't expose a single coef_ vector; train
        # an uncalibrated companion just for interpretability.
        companion = make_estimator(C=best_params["C"], l1_ratio=best_params["l1_ratio"])
        _fit_with_warning_check(companion, X, y)
        coefs = extract_coefficients(companion, experiment.feature_cols)
        coefs.to_csv(out_dir / "coefficients.csv", index=False)

    print(f"  [sim] Monte Carlo {n_simulations} sims ...")
    group_cache, knockout_cache = precompute_neutral_probabilities(
        experiment, pipeline, team_features_2026
    )
    sim_results = simulate_from_caches(
        experiment.teams_2026,
        experiment.groups_2026,
        experiment.momios,
        group_cache,
        knockout_cache,
        n_simulations=n_simulations,
        output_path=str(out_dir / "simulation_results.csv"),
        seed=SEED,
    )
    corr_vs_momios = float(sim_results["champion_pct"].corr(sim_results["momios_pct"]))
    print(f"    corr_vs_momios={corr_vs_momios:.4f}")

    (out_dir / "best_params.json").write_text(
        json.dumps(
            {
                **best_params,
                "calibrate": calibrate,
                "drop_features": drop_features,
                "best_inner_log_loss": float(study.best_value),
                "n_trials": n_trials,
                "inner_year_max": INNER_YEAR_MAX,
            },
            indent=2,
        )
    )

    return {
        "variant": name,
        "drop_features": ",".join(drop_features) if drop_features else "",
        "calibrate": calibrate or "none",
        "mean_log_loss": float(cv_metrics["log_loss"].mean()),
        "mean_accuracy": float(cv_metrics["accuracy"].mean()),
        "mean_brier": float(cv_metrics["brier"].mean()),
        "corr_vs_momios": corr_vs_momios,
        "best_inner_log_loss": float(study.best_value),
        "sim_results": sim_results,
    }


def _load_baseline_v2() -> dict[str, object]:
    cv = pd.read_csv(LOGREG_ARTIFACTS_DIR / "cv_metrics.csv")
    sim = pd.read_csv(LOGREG_ARTIFACTS_DIR / "simulation_results_logreg.csv")
    bp = json.loads((LOGREG_ARTIFACTS_DIR / "best_params.json").read_text())
    mean_row = cv[cv["test_year"] == "mean"].iloc[0]
    return {
        "variant": "v2_baseline",
        "drop_features": "",
        "calibrate": "none",
        "mean_log_loss": float(mean_row["log_loss"]),
        "mean_accuracy": float(mean_row["accuracy"]),
        "mean_brier": float(mean_row["brier"]),
        "corr_vs_momios": float(sim["champion_pct"].corr(sim["momios_pct"])),
        "best_inner_log_loss": float(bp.get("best_inner_log_loss", float("nan"))),
        "sim_results": sim,
    }


def main() -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    _logreg_module._convergence_warnings = 0

    print("[prep] Loading dataset and team features (once) ...")
    base_experiment = prepare_experiment()
    base_team_features = prepare_2026_team_features(base_experiment)

    results: list[dict[str, object]] = [_load_baseline_v2()]
    print(
        f"[baseline] v2_baseline loaded from disk: "
        f"log_loss={results[0]['mean_log_loss']:.4f} "
        f"acc={results[0]['mean_accuracy']:.4f} "
        f"brier={results[0]['mean_brier']:.4f} "
        f"corr={results[0]['corr_vs_momios']:.4f}"
    )

    for variant in VARIANTS:
        r = run_variant(
            name=variant["name"],
            base_experiment=base_experiment,
            base_team_features=base_team_features,
            drop_features=variant["drop_features"],
            calibrate=variant["calibrate"],
        )
        results.append(r)

    summary_rows = [{k: v for k, v in r.items() if k != "sim_results"} for r in results]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_PATH, index=False)

    sim_dfs = []
    for r in results:
        sim_df = r["sim_results"][["country_code", "champion_pct"]].rename(
            columns={"champion_pct": r["variant"]}
        )
        sim_dfs.append(sim_df)
    merged = sim_dfs[0]
    for sim_df in sim_dfs[1:]:
        merged = merged.merge(sim_df, on="country_code", how="outer")
    momios = results[0]["sim_results"][["country_code", "momios_pct"]]
    merged = merged.merge(momios, on="country_code", how="left")
    merged = merged.sort_values("v2_baseline", ascending=False).reset_index(drop=True)
    merged.to_csv(CHAMPION_PATH, index=False)

    print("\n=== METRICS SUMMARY (LOTO outer, mean over 6 folds) ===")
    cols = [
        "variant",
        "drop_features",
        "calibrate",
        "mean_log_loss",
        "mean_accuracy",
        "mean_brier",
        "corr_vs_momios",
    ]
    print(summary[cols].to_string(index=False))

    print("\n=== TOP 10 CHAMPION PROBABILITIES (by v2_baseline order) ===")
    print(merged.head(10).to_string(index=False))

    print(
        f"\nArtifacts: {EXPERIMENTS_DIR}\n  summary: {SUMMARY_PATH}\n  champions: {CHAMPION_PATH}"
    )
    print(f"Total sklearn ConvergenceWarnings during run: {_logreg_module._convergence_warnings}")


if __name__ == "__main__":
    main()

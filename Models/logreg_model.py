"""Multinomial logistic regression (elastic-net) alternative to the XGBoost
temporal-master model. Reuses the feature-engineering pipeline of
`current_model_pipeline` and the Monte Carlo simulator of `compare_models`."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings(
    "ignore",
    message=r".*'penalty' was deprecated.*",
    category=FutureWarning,
)

_convergence_warnings = 0


def _fit_with_warning_check(estimator: Pipeline, X: np.ndarray, y: np.ndarray) -> Pipeline:
    """Fit and accumulate sklearn ConvergenceWarning count into a module-level
    counter. Lets us tell L1-zero coefs (real regularization) apart from
    timeout-zero coefs (saga gave up before convergence)."""
    global _convergence_warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(X, y)
    _convergence_warnings += sum(
        1 for w in caught if issubclass(w.category, ConvergenceWarning)
    )
    return estimator

try:
    from .current_model_pipeline import (
        PreparedCurrentExperiment,
        prepare_2026_team_features,
        prepare_experiment,
        precompute_neutral_probabilities,
    )
    from .compare_models import multiclass_brier_score, simulate_from_caches
except ImportError:
    from current_model_pipeline import (  # type: ignore[no-redef]
        PreparedCurrentExperiment,
        prepare_2026_team_features,
        prepare_experiment,
        precompute_neutral_probabilities,
    )
    from compare_models import multiclass_brier_score, simulate_from_caches  # type: ignore[no-redef]


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
ARTIFACTS_DIR = THIS_DIR / "artifacts" / "logreg"
XGB_EVAL_PATH = THIS_DIR / "artifacts" / "current_temporal_master_evaluation_folds.csv"
XGB_SIM_PATH = THIS_DIR / "artifacts" / "current_temporal_master_simulation_results.csv"

SEED = 42
N_TRIALS = 100
N_SIMULATIONS = 50_000
INNER_YEAR_MAX = 2014
MAX_ITER = 20000
TOL = 1e-5
CLASS_LABELS = {0: "away_win", 1: "draw", 2: "home_win"}


def make_estimator(C: float, l1_ratio: float) -> Pipeline:
    """Build a fresh sklearn Pipeline: StandardScaler -> multinomial LogReg (elastic net).

    StandardScaler lives inside the Pipeline so it is refit on train-only data
    every fold, preventing scale leakage from test years. `class_weight` is
    intentionally fixed to None: the audit found that `balanced` was canceling
    the home-advantage prior and degrading accuracy by ~5pp."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    C=C,
                    l1_ratio=l1_ratio,
                    class_weight=None,
                    max_iter=MAX_ITER,
                    tol=TOL,
                    random_state=SEED,
                ),
            ),
        ]
    )


def _params_for_estimator(params: dict[str, object]) -> dict[str, object]:
    return {
        "C": float(params["C"]),
        "l1_ratio": float(params["l1_ratio"]),
    }


def tune_logreg(
    experiment: PreparedCurrentExperiment,
    n_trials: int = N_TRIALS,
    inner_year_max: int = INNER_YEAR_MAX,
    seed: int = SEED,
) -> tuple[dict[str, object], optuna.Study]:
    X = experiment.train_df[experiment.feature_cols].to_numpy()
    y = experiment.train_df["outcome"].to_numpy()
    inner_folds = [fold for fold in experiment.folds if fold[2] <= inner_year_max]
    if not inner_folds:
        raise ValueError("No inner folds available for tuning (test_year <= inner_year_max).")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }
        losses = []
        for train_mask, test_mask, _ in inner_folds:
            est = make_estimator(**_params_for_estimator(params))
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
    return dict(study.best_params), study


def evaluate_loto(
    experiment: PreparedCurrentExperiment,
    best_params: dict[str, object],
) -> pd.DataFrame:
    X = experiment.train_df[experiment.feature_cols].to_numpy()
    y = experiment.train_df["outcome"].to_numpy()
    rows = []
    for train_mask, test_mask, test_year in experiment.folds:
        est = make_estimator(**_params_for_estimator(best_params))
        _fit_with_warning_check(est, X[train_mask], y[train_mask])
        y_proba = est.predict_proba(X[test_mask])
        y_pred = est.predict(X[test_mask])
        rows.append(
            {
                "model": "logreg",
                "test_year": test_year,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "log_loss": float(log_loss(y[test_mask], y_proba, labels=[0, 1, 2])),
                "accuracy": float(accuracy_score(y[test_mask], y_pred)),
                "brier": multiclass_brier_score(y[test_mask], y_proba),
            }
        )
    return pd.DataFrame(rows)


def train_final(
    experiment: PreparedCurrentExperiment,
    best_params: dict[str, object],
) -> Pipeline:
    X = experiment.train_df[experiment.feature_cols].to_numpy()
    y = experiment.train_df["outcome"].to_numpy()
    pipeline = make_estimator(**_params_for_estimator(best_params))
    _fit_with_warning_check(pipeline, X, y)
    return pipeline


def extract_coefficients(pipeline: Pipeline, feature_cols: list[str]) -> pd.DataFrame:
    """Returns one row per (class, feature) with both standardized and
    original-units coefficients. Original-units coefficients are obtained by
    dividing by the scaler's std, which lets you read effect sizes in the
    natural units of each delta (e.g. delta_elo points)."""
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    clf: LogisticRegression = pipeline.named_steps["clf"]
    coef_std = clf.coef_  # (n_classes, n_features)
    intercept_std = clf.intercept_  # (n_classes,)
    coef_orig = coef_std / scaler.scale_
    intercept_orig = intercept_std - coef_std @ (scaler.mean_ / scaler.scale_)

    rows: list[dict[str, object]] = []
    classes = list(clf.classes_)
    for class_idx, class_value in enumerate(classes):
        class_name = CLASS_LABELS.get(int(class_value), str(class_value))
        for feat_idx, feat_name in enumerate(feature_cols):
            rows.append(
                {
                    "class": class_name,
                    "feature": feat_name,
                    "coef_standardized": float(coef_std[class_idx, feat_idx]),
                    "coef_original": float(coef_orig[class_idx, feat_idx]),
                }
            )
        rows.append(
            {
                "class": class_name,
                "feature": "_intercept_",
                "coef_standardized": float(intercept_std[class_idx]),
                "coef_original": float(intercept_orig[class_idx]),
            }
        )
    return pd.DataFrame(rows)


def build_dataset(experiment: PreparedCurrentExperiment) -> tuple[np.ndarray, np.ndarray]:
    """Materialize (X, y) and run integrity checks (shape, no NaN, dtypes)."""
    X = experiment.train_df[experiment.feature_cols].to_numpy(dtype=float)
    y = experiment.train_df["outcome"].to_numpy(dtype=int)
    n_features_expected = 22
    if X.shape[1] != n_features_expected:
        raise AssertionError(f"Expected {n_features_expected} features, got {X.shape[1]}")
    if np.isnan(X).any():
        nan_cols = [
            experiment.feature_cols[c]
            for c in range(X.shape[1])
            if np.isnan(X[:, c]).any()
        ]
        raise AssertionError(f"NaN found in features: {nan_cols}")
    if not set(np.unique(y)).issubset({0, 1, 2}):
        raise AssertionError(f"Unexpected target values: {np.unique(y)}")
    return X, y


def build_comparison_vs_xgboost(
    cv_metrics: pd.DataFrame,
    sim_results: pd.DataFrame,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if not XGB_EVAL_PATH.exists() or not XGB_SIM_PATH.exists():
        print(f"  XGBoost artifacts not found at {XGB_EVAL_PATH.parent} -- skipping comparison")
        return None, None

    xgb_eval = pd.read_csv(XGB_EVAL_PATH)
    xgb_sim = pd.read_csv(XGB_SIM_PATH)

    corr_logreg = float(sim_results["champion_pct"].corr(sim_results["momios_pct"]))
    corr_xgb = float(xgb_sim["champion_pct"].corr(xgb_sim["momios_pct"]))

    cv_compare = pd.DataFrame(
        [
            {
                "model": "logreg_elasticnet",
                "mean_log_loss": float(cv_metrics["log_loss"].mean()),
                "mean_accuracy": float(cv_metrics["accuracy"].mean()),
                "mean_brier": float(cv_metrics["brier"].mean()),
                "corr_vs_momios": corr_logreg,
            },
            {
                "model": "xgboost_current_temporal_master",
                "mean_log_loss": float(xgb_eval["log_loss"].mean()),
                "mean_accuracy": float(xgb_eval["accuracy"].mean()),
                "mean_brier": float(xgb_eval["brier"].mean()),
                "corr_vs_momios": corr_xgb,
            },
        ]
    )

    logreg_sim = sim_results[["country_code", "champion_pct"]].rename(
        columns={"champion_pct": "logreg_champion_pct"}
    )
    xgb_renamed = xgb_sim[["country_code", "champion_pct", "momios_pct"]].rename(
        columns={"champion_pct": "xgboost_champion_pct"}
    )
    champion_compare = logreg_sim.merge(xgb_renamed, on="country_code", how="inner")
    champion_compare["delta_logreg_minus_xgb"] = (
        champion_compare["logreg_champion_pct"] - champion_compare["xgboost_champion_pct"]
    )
    champion_compare = champion_compare.sort_values(
        "logreg_champion_pct", ascending=False
    ).reset_index(drop=True)
    return cv_compare, champion_compare


def main() -> None:
    global _convergence_warnings
    _convergence_warnings = 0
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/7] Preparing experiment (matches + features) ...")
    experiment = prepare_experiment()
    X, y = build_dataset(experiment)
    print(
        f"  matches={X.shape[0]} features={X.shape[1]} class_distribution={np.bincount(y).tolist()}"
    )

    print(f"[2/7] Tuning logreg with Optuna ({N_TRIALS} trials, inner folds <= {INNER_YEAR_MAX}) ...")
    best_params, study = tune_logreg(experiment, n_trials=N_TRIALS)
    print(f"  best_params={best_params}")
    print(f"  best_inner_log_loss={study.best_value:.4f}")
    (ARTIFACTS_DIR / "best_params.json").write_text(
        json.dumps(
            {
                **best_params,
                "class_weight": None,
                "best_inner_log_loss": float(study.best_value),
                "n_trials": N_TRIALS,
                "inner_year_max": INNER_YEAR_MAX,
                "max_iter": MAX_ITER,
                "tol": TOL,
                "seed": SEED,
            },
            indent=2,
        )
    )

    print("[3/7] LOTO outer CV ...")
    cv_metrics = evaluate_loto(experiment, best_params)
    mean_row = pd.DataFrame(
        [
            {
                "model": "logreg",
                "test_year": "mean",
                "n_train": float(cv_metrics["n_train"].mean()),
                "n_test": int(cv_metrics["n_test"].sum()),
                "log_loss": float(cv_metrics["log_loss"].mean()),
                "accuracy": float(cv_metrics["accuracy"].mean()),
                "brier": float(cv_metrics["brier"].mean()),
            }
        ]
    )
    cv_full = pd.concat([cv_metrics, mean_row], ignore_index=True)
    cv_full.to_csv(ARTIFACTS_DIR / "cv_metrics.csv", index=False)
    print(cv_full.to_string(index=False))

    print("[4/7] Training final model on full dataset ...")
    pipeline = train_final(experiment, best_params)
    joblib.dump(pipeline, ARTIFACTS_DIR / "pipeline.pkl")

    print("[5/7] Extracting coefficients ...")
    coefs = extract_coefficients(pipeline, experiment.feature_cols)
    coefs.to_csv(ARTIFACTS_DIR / "coefficients.csv", index=False)

    strength_features = [
        "delta_elo",
        "delta_rank_strength",
        "delta_squad_market_value_log",
        "delta_form_wp",
    ]
    home_total = float(
        coefs[
            (coefs["class"] == "home_win")
            & (coefs["feature"].isin(strength_features))
        ]["coef_standardized"].sum()
    )
    away_total = float(
        coefs[
            (coefs["class"] == "away_win")
            & (coefs["feature"].isin(strength_features))
        ]["coef_standardized"].sum()
    )
    print(
        f"  sanity (sum of standardized strength-proxy coefs): home_win={home_total:+.4f} "
        f"away_win={away_total:+.4f}  -- expect home_win > 0 and away_win < 0"
    )
    if home_total <= 0 or away_total >= 0:
        print(
            "  WARNING: strength-proxy sign sanity check failed -- inspect coefficients.csv"
        )

    print(f"[6/7] Running Monte Carlo simulation ({N_SIMULATIONS} sims) ...")
    team_features_2026 = prepare_2026_team_features(experiment)
    group_cache, knockout_cache = precompute_neutral_probabilities(
        experiment, pipeline, team_features_2026
    )
    sim_results = simulate_from_caches(
        experiment.teams_2026,
        experiment.groups_2026,
        experiment.momios,
        group_cache,
        knockout_cache,
        n_simulations=N_SIMULATIONS,
        output_path=str(ARTIFACTS_DIR / "simulation_results_logreg.csv"),
        seed=SEED,
    )
    total_pct = float(sim_results["champion_pct"].sum())
    print(f"  champion_pct sum = {total_pct:.2f} (should be ~100)")
    print("  top-10 champion probabilities:")
    print(sim_results.head(10).to_string(index=False))

    print("[7/7] Building comparison vs XGBoost ...")
    cv_compare, champion_compare = build_comparison_vs_xgboost(cv_metrics, sim_results)
    if cv_compare is not None and champion_compare is not None:
        cv_compare.to_csv(ARTIFACTS_DIR / "comparison_vs_xgboost.csv", index=False)
        champion_compare.to_csv(ARTIFACTS_DIR / "champion_pct_comparison.csv", index=False)
        corr_xgb = float(
            champion_compare["logreg_champion_pct"].corr(champion_compare["xgboost_champion_pct"])
        )
        print(f"  corr(logreg vs xgboost) champion_pct = {corr_xgb:.3f}")
        print(cv_compare.to_string(index=False))
        print("  top-10 side-by-side:")
        print(champion_compare.head(10).to_string(index=False))

    print(f"\nArtifacts saved to: {ARTIFACTS_DIR}")
    print(f"Total sklearn ConvergenceWarnings during run: {_convergence_warnings}")
    if _convergence_warnings > 0:
        print(
            f"  -> {_convergence_warnings} fits did not converge in MAX_ITER={MAX_ITER}; "
            "consider raising further or relaxing TOL."
        )


if __name__ == "__main__":
    main()

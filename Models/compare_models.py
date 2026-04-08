from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss

try:
    from .current_model_pipeline import (
        PreparedCurrentExperiment,
        fit_final_model as fit_current_model,
        save_trained_model,
        prepare_2026_team_features,
        prepare_experiment as prepare_current_experiment,
        precompute_neutral_probabilities as precompute_current_probabilities,
        tune_model as tune_current_model,
    )
    from .master48_alt_model import (
        build_knockout_bracket,
        fit_final_model as fit_master48_model,
        get_advancing_teams,
        precompute_neutral_probabilities as precompute_master48_probabilities,
        prepare_experiment as prepare_master48_experiment,
        simulate_group,
        simulate_knockout_rounds,
        tune_model as tune_master48_model,
    )
except ImportError:
    from current_model_pipeline import (
        PreparedCurrentExperiment,
        fit_final_model as fit_current_model,
        save_trained_model,
        prepare_2026_team_features,
        prepare_experiment as prepare_current_experiment,
        precompute_neutral_probabilities as precompute_current_probabilities,
        tune_model as tune_current_model,
    )
    from master48_alt_model import (
        build_knockout_bracket,
        fit_final_model as fit_master48_model,
        get_advancing_teams,
        precompute_neutral_probabilities as precompute_master48_probabilities,
        prepare_experiment as prepare_master48_experiment,
        simulate_group,
        simulate_knockout_rounds,
        tune_model as tune_master48_model,
    )


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent if THIS_DIR.name == "Models" else THIS_DIR
ARTIFACTS_DIR = THIS_DIR / "artifacts"
CURRENT_EVAL_FOLDS_PATH = ARTIFACTS_DIR / "current_temporal_master_evaluation_folds.csv"
CURRENT_EVAL_SUMMARY_PATH = ARTIFACTS_DIR / "current_temporal_master_evaluation_summary.csv"
CURRENT_SIMULATION_PATH = ARTIFACTS_DIR / "current_temporal_master_simulation_results.csv"
CURRENT_BUNDLE_MANIFEST_PATH = ARTIFACTS_DIR / "current_temporal_master_artifact_bundle.json"
SEED = 42
N_TRIALS = 60
N_SIMULATIONS = 50_000
OUTPUT_DIR = ROOT_DIR / "Data"


def multiclass_brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    one_hot = np.eye(y_proba.shape[1])[y_true]
    return float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))


def rebuild_folds(train_df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, int]]:
    years = train_df["year"].to_numpy()
    tournaments = train_df["tournament"].to_numpy()
    folds = []
    for test_year in [2002, 2006, 2010, 2014, 2018, 2022]:
        train_mask = years < test_year
        test_mask = (years == test_year) & (tournaments == "WC")
        if train_mask.sum() > 0 and test_mask.sum() > 0:
            folds.append((train_mask, test_mask, test_year))
    return folds


def subset_current_experiment(
    experiment: PreparedCurrentExperiment,
    subset_mask: np.ndarray,
) -> PreparedCurrentExperiment:
    subset_df = experiment.train_df.loc[subset_mask].reset_index(drop=True)
    return replace(experiment, train_df=subset_df, folds=rebuild_folds(subset_df))


def evaluate_cv(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    folds: list[tuple[np.ndarray, np.ndarray, int]],
    params: dict[str, object],
    model_name: str,
    benchmark_name: str,
    train_extra_mask: np.ndarray | None = None,
    test_extra_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    X = train_df[feature_cols].to_numpy()
    y = train_df["outcome"].to_numpy()
    if train_extra_mask is None:
        train_extra_mask = np.ones(len(train_df), dtype=bool)
    if test_extra_mask is None:
        test_extra_mask = np.ones(len(train_df), dtype=bool)

    rows = []
    for base_train_mask, base_test_mask, test_year in folds:
        train_mask = base_train_mask & train_extra_mask
        test_mask = base_test_mask & test_extra_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        model = xgb.XGBClassifier(**params)
        model.fit(X[train_mask], y[train_mask], eval_set=[(X[test_mask], y[test_mask])], verbose=False)
        y_proba = model.predict_proba(X[test_mask])
        y_pred = model.predict(X[test_mask])

        rows.append(
            {
                "model": model_name,
                "benchmark": benchmark_name,
                "test_year": test_year,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "log_loss": float(log_loss(y[test_mask], y_proba)),
                "accuracy": float(accuracy_score(y[test_mask], y_pred)),
                "brier": multiclass_brier_score(y[test_mask], y_proba),
            }
        )

    return pd.DataFrame(rows)


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for (model_name, benchmark_name), group in metrics_df.groupby(["model", "benchmark"], sort=False):
        summary_rows.append(
            {
                "model": model_name,
                "benchmark": benchmark_name,
                "folds": int(len(group)),
                "total_test_matches": int(group["n_test"].sum()),
                "mean_log_loss": float(group["log_loss"].mean()),
                "mean_accuracy": float(group["accuracy"].mean()),
                "mean_brier": float(group["brier"].mean()),
            }
        )
    return pd.DataFrame(summary_rows).sort_values(["benchmark", "mean_log_loss"]).reset_index(drop=True)


def simulate_from_caches(
    teams_2026: list[str],
    groups_2026: dict[str, list[str]],
    momios: pd.DataFrame,
    group_cache: dict[tuple[str, str], np.ndarray],
    knockout_cache: dict[tuple[str, str], np.ndarray],
    n_simulations: int,
    output_path: str,
    seed: int = SEED,
    extra_team_fields: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    champion_counts: dict[str, int] = defaultdict(int)
    finalist_counts: dict[str, int] = defaultdict(int)
    semifinal_counts: dict[str, int] = defaultdict(int)
    group_exit_counts: dict[str, int] = defaultdict(int)

    valid_sims = 0
    for _ in range(n_simulations):
        group_results = {}
        for group_letter, group_teams in groups_2026.items():
            group_results[group_letter] = simulate_group(group_teams, group_cache, rng)

        top2, best_thirds = get_advancing_teams(group_results)
        r32_matches = build_knockout_bracket(groups_2026, top2, best_thirds)
        advancing = set(top2 + [team for team, _ in best_thirds])
        for team in teams_2026:
            if team not in advancing:
                group_exit_counts[team] += 1

        champion, finalist, round_results = simulate_knockout_rounds(r32_matches, knockout_cache, rng)
        champion_counts[champion] += 1
        finalist_counts[finalist] += 1
        for team in round_results.get("semifinal", set()):
            semifinal_counts[team] += 1
        valid_sims += 1

    momios_lookup = dict(zip(momios["country_code"], momios["champion_prob"] * 100))
    rows = []
    for team in teams_2026:
        row = {
            "country_code": team,
            "champion_pct": champion_counts[team] / valid_sims * 100 if valid_sims else 0.0,
            "finalist_pct": (champion_counts[team] + finalist_counts[team]) / valid_sims * 100 if valid_sims else 0.0,
            "semifinal_pct": semifinal_counts[team] / valid_sims * 100 if valid_sims else 0.0,
            "group_exit_pct": group_exit_counts[team] / valid_sims * 100 if valid_sims else 0.0,
            "momios_pct": float(momios_lookup.get(team, np.nan)),
        }
        if extra_team_fields and team in extra_team_fields:
            row.update(extra_team_fields[team])
        rows.append(row)

    results_df = pd.DataFrame(rows).sort_values("champion_pct", ascending=False).reset_index(drop=True)
    results_df.to_csv(output_path, index=False)
    return results_df


def correlation_with_momios(results_df: pd.DataFrame) -> float:
    return float(results_df["champion_pct"].corr(results_df["momios_pct"]))


def build_simulation_comparison(
    current_results: pd.DataFrame,
    master48_results: pd.DataFrame,
) -> pd.DataFrame:
    current_df = current_results[["country_code", "champion_pct", "momios_pct"]].rename(
        columns={"champion_pct": "current_champion_pct", "momios_pct": "momios_pct"}
    )
    master48_df = master48_results[["country_code", "champion_pct"]].rename(
        columns={"champion_pct": "master48_champion_pct"}
    )
    comparison = current_df.merge(master48_df, on="country_code", how="inner")
    comparison["delta_current_minus_master48"] = (
        comparison["current_champion_pct"] - comparison["master48_champion_pct"]
    )
    return comparison.sort_values("current_champion_pct", ascending=False).reset_index(drop=True)


def save_current_artifact_bundle(
    folds_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    simulation_df: pd.DataFrame,
    correlation: float,
    n_trials: int,
    n_simulations: int,
    model_path: Path,
    metadata_path: Path,
) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    folds_df.to_csv(CURRENT_EVAL_FOLDS_PATH, index=False)

    summary_row = summary_df[
        (summary_df["model"] == "current") & (summary_df["benchmark"] == "full_coverage")
    ].copy()
    if summary_row.empty:
        raise ValueError("Current full_coverage summary row not found.")
    summary_row["corr_vs_momios"] = correlation
    summary_row["n_trials"] = n_trials
    summary_row["n_simulations"] = n_simulations
    summary_row.to_csv(CURRENT_EVAL_SUMMARY_PATH, index=False)

    simulation_df.to_csv(CURRENT_SIMULATION_PATH, index=False)

    manifest = {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "evaluation_folds_path": str(CURRENT_EVAL_FOLDS_PATH),
        "evaluation_summary_path": str(CURRENT_EVAL_SUMMARY_PATH),
        "simulation_results_path": str(CURRENT_SIMULATION_PATH),
        "n_trials": n_trials,
        "n_simulations": n_simulations,
        "corr_vs_momios": correlation,
    }
    CURRENT_BUNDLE_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Preparing experiments...")
    current_experiment = prepare_current_experiment()
    master48_experiment = prepare_master48_experiment()

    usable_master48_teams = set(master48_experiment.team_feature_table.index)
    current_common_mask = (
        current_experiment.train_df["home_team"].isin(usable_master48_teams)
        & current_experiment.train_df["away_team"].isin(usable_master48_teams)
    ).to_numpy()
    current_common_experiment = subset_current_experiment(current_experiment, current_common_mask)

    print(
        f"Current model matches: {len(current_experiment.train_df)} | "
        f"Current common-subset matches: {len(current_common_experiment.train_df)} | "
        f"Master48 matches: {len(master48_experiment.train_df)}"
    )

    print("Tuning current full model...")
    current_full_params, _ = tune_current_model(current_experiment, n_trials=N_TRIALS)
    print("Tuning current common-subset model...")
    current_common_params, _ = tune_current_model(current_common_experiment, n_trials=N_TRIALS)
    print("Tuning master48 model...")
    master48_params, _ = tune_master48_model(master48_experiment, n_trials=N_TRIALS)

    print("Evaluating CV benchmarks...")
    current_full_cv = evaluate_cv(
        current_experiment.train_df,
        current_experiment.feature_cols,
        current_experiment.folds,
        current_full_params,
        model_name="current",
        benchmark_name="full_coverage",
    )
    current_shared_test_cv = evaluate_cv(
        current_experiment.train_df,
        current_experiment.feature_cols,
        current_experiment.folds,
        current_full_params,
        model_name="current",
        benchmark_name="shared_test_full_training",
        test_extra_mask=current_common_mask,
    )
    current_common_cv = evaluate_cv(
        current_common_experiment.train_df,
        current_common_experiment.feature_cols,
        current_common_experiment.folds,
        current_common_params,
        model_name="current",
        benchmark_name="shared_train_and_test",
    )
    master48_cv = evaluate_cv(
        master48_experiment.train_df,
        master48_experiment.feature_cols,
        master48_experiment.folds,
        master48_params,
        model_name="master48",
        benchmark_name="shared_train_and_test",
    )

    metrics_df = pd.concat(
        [current_full_cv, current_shared_test_cv, current_common_cv, master48_cv],
        ignore_index=True,
    )
    summary_df = summarize_metrics(metrics_df)
    metrics_df.to_csv(OUTPUT_DIR / "model_comparison_cv.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "model_comparison_summary.csv", index=False)

    print("Training final models...")
    current_full_model = fit_current_model(current_experiment, current_full_params)
    master48_model = fit_master48_model(master48_experiment, master48_params)
    saved_model_path, saved_metadata_path = save_trained_model(
        current_full_model,
        current_experiment,
        current_full_params,
        extra_metadata={
            "benchmark": "full_coverage",
            "source_script": "Models/compare_models.py",
        },
    )
    print(f"Saved current model artifact: {saved_model_path}")
    print(f"Saved current model metadata: {saved_metadata_path}")

    print("Preparing 2026 features and probabilities...")
    current_team_features = prepare_2026_team_features(current_experiment)
    current_group_cache, current_knockout_cache = precompute_current_probabilities(
        current_experiment,
        current_full_model,
        current_team_features,
    )
    master48_group_cache, master48_knockout_cache = precompute_master48_probabilities(
        master48_experiment,
        master48_model,
    )

    print("Running 2026 Monte Carlo simulations...")
    current_results = simulate_from_caches(
        current_experiment.teams_2026,
        current_experiment.groups_2026,
        current_experiment.momios,
        current_group_cache,
        current_knockout_cache,
        n_simulations=N_SIMULATIONS,
        output_path=str(OUTPUT_DIR / "simulation_results_current_compare.csv"),
        extra_team_fields={team: {"elo": values["elo"]} for team, values in current_team_features.items()},
    )
    master48_results = simulate_from_caches(
        master48_experiment.teams_2026,
        master48_experiment.groups_2026,
        master48_experiment.momios,
        master48_group_cache,
        master48_knockout_cache,
        n_simulations=N_SIMULATIONS,
        output_path=str(OUTPUT_DIR / "simulation_results_master48_compare.csv"),
    )

    simulation_comparison_df = build_simulation_comparison(current_results, master48_results)
    simulation_comparison_df.to_csv(OUTPUT_DIR / "model_simulation_comparison.csv", index=False)
    save_current_artifact_bundle(
        current_full_cv,
        summary_df,
        current_results,
        correlation_with_momios(current_results),
        N_TRIALS,
        N_SIMULATIONS,
        saved_model_path,
        saved_metadata_path,
    )

    print("\n=== CV Summary ===")
    print(summary_df.to_string(index=False))
    print("\n=== Simulation Correlation vs Momios ===")
    print(f"Current model:  r = {correlation_with_momios(current_results):.3f}")
    print(f"Master48 model: r = {correlation_with_momios(master48_results):.3f}")
    print("\n=== Top 10 Champion Probabilities (Current vs Master48) ===")
    print(simulation_comparison_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

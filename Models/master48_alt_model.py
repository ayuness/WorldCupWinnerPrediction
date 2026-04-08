from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import glob

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent if THIS_DIR.name == "Models" else THIS_DIR
SEED = 42
COUNTRY_CODE_NORMALIZATION = {
    "CGO": "COD",
}


def _repo_path(*parts: str) -> Path:
    return ROOT_DIR.joinpath(*parts)


@dataclass
class PreparedMaster48Experiment:
    master48_raw: pd.DataFrame
    team_feature_table: pd.DataFrame
    team_feature_cols: list[str]
    all_matches: pd.DataFrame
    train_df: pd.DataFrame
    feature_cols: list[str]
    folds: list[tuple[np.ndarray, np.ndarray, int]]
    teams_2026: list[str]
    groups_2026: dict[str, list[str]]
    momios: pd.DataFrame
    retained_match_stats: dict[str, Any]


def _normalize_country_code(code: Any) -> str | Any:
    if pd.isna(code):
        return code
    return COUNTRY_CODE_NORMALIZATION.get(str(code), str(code))


def _build_name_to_code(rankings: pd.DataFrame) -> dict[str, str]:
    name_to_code = dict(zip(rankings["team_name"], rankings["country_code"]))
    name_to_code.update(
        {
            "West Germany": "GER",
            "Germany DR": "GER_DR",
            "Yugoslavia": "YUG",
            "FR Yugoslavia": "SRB",
            "Serbia and Montenegro": "SRB",
            "Soviet Union": "URS",
            "Czechoslovakia": "CZE",
            "Czech Republic": "CZE",
            "Zaire": "COD",
            "Dutch East Indies": "IDN",
            "China PR": "CHN",
            "Korea DPR": "PRK",
            "United States": "USA",
            "Trinidad and Tobago": "TTO",
            "United Arab Emirates": "ARE",
            "Costa Rica": "CRC",
            "El Salvador": "SLV",
            "Serbia": "SRB",
            "Russia": "RUS",
            "Togo": "TGO",
            "Honduras": "HON",
            "Cuba": "CUB",
            "Iceland": "ISL",
            "Slovenia": "SVN",
            "Greece": "GRE",
            "Angola": "ANG",
            "Cameroon": "CMR",
            "Nigeria": "NGA",
            "Bulgaria": "BUL",
            "Hungary": "HUN",
            "Peru": "PER",
            "Chile": "CHI",
            "Bolivia": "BOL",
            "Israel": "ISR",
            "Kuwait": "KUW",
            "Denmark": "DEN",
            "Brasil": "BRA",
            "Equador": "ECU",
            "Paraguai": "PAR",
            "Uruguai": "URU",
            "EUA": "USA",
            "Mexico": "MEX",
            "Japao": "JPN",
            "Catar": "QAT",
            "Haiti": "HAI",
            "Venezuela": "VEN",
            "Jamaica": "JAM",
            "Panama": "PAN",
            "Georgia": "GEO",
            "North Macedonia": "MKD",
        }
    )
    return name_to_code


def _unify_matches(
    wc_matches: pd.DataFrame,
    euro: pd.DataFrame,
    copa: pd.DataFrame,
    name_to_code: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, match in wc_matches.iterrows():
        home_code = name_to_code.get(match["home_team"])
        away_code = name_to_code.get(match["away_team"])
        if home_code and away_code:
            rows.append(
                {
                    "date": match["Date"],
                    "home_code": _normalize_country_code(home_code),
                    "away_code": _normalize_country_code(away_code),
                    "home_score": match["home_score"],
                    "away_score": match["away_score"],
                    "tournament": "WC",
                    "year": int(match["Year"]),
                    "stage": match["Round"],
                    "tournament_weight": 1.0,
                }
            )

    for _, match in euro.iterrows():
        home_code = match.get("home_team_code")
        away_code = match.get("away_team_code")
        if pd.isna(home_code) or pd.isna(away_code):
            home_code = name_to_code.get(match.get("home_team"))
            away_code = name_to_code.get(match.get("away_team"))
        if pd.notna(home_code) and pd.notna(away_code):
            rows.append(
                {
                    "date": match.get("date", f"{int(match['year'])}-06-15"),
                    "home_code": _normalize_country_code(home_code),
                    "away_code": _normalize_country_code(away_code),
                    "home_score": match["home_score"],
                    "away_score": match["away_score"],
                    "tournament": "EURO",
                    "year": int(match["year"]),
                    "stage": match.get("round", ""),
                    "tournament_weight": 0.85,
                }
            )

    for _, match in copa.iterrows():
        home_code = name_to_code.get(match["Casa"])
        away_code = name_to_code.get(match["Fora"])
        if home_code and away_code:
            rows.append(
                {
                    "date": match["Data"],
                    "home_code": _normalize_country_code(home_code),
                    "away_code": _normalize_country_code(away_code),
                    "home_score": match["Gols Casa"],
                    "away_score": match["Gols Fora"],
                    "tournament": "COPA",
                    "year": int(match["Edição"]),
                    "stage": match["Fase"],
                    "tournament_weight": 0.80,
                }
            )

    all_matches = pd.DataFrame(rows)
    all_matches = all_matches.dropna(subset=["home_score", "away_score"]).copy()
    all_matches["date"] = pd.to_datetime(all_matches["date"], errors="coerce")
    all_matches = all_matches.sort_values("date").reset_index(drop=True)
    all_matches["outcome"] = np.where(
        all_matches["home_score"] > all_matches["away_score"],
        2,
        np.where(all_matches["home_score"] < all_matches["away_score"], 0, 1),
    )

    knockout_keywords = {
        "Final",
        "Semi",
        "Quarter",
        "Round of 16",
        "Round of 32",
        "FINAL",
        "SEMIFINAL",
        "QUARTER",
        "ROUND_OF_16",
        "Semifinal",
        "Cuartos",
        "Octavos",
        "Third-place",
        "Second round",
        "Final stage",
    }
    all_matches["is_knockout"] = all_matches["stage"].apply(
        lambda stage: int(any(keyword.lower() in str(stage).lower() for keyword in knockout_keywords))
    )
    return all_matches


def _encode_master48_features(master48: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    categorical_cols = [col for col in ["confederation", "status"] if col in master48.columns]
    numeric_cols = [col for col in master48.columns if col not in ["country_code", *categorical_cols]]

    encoded_parts = [master48[["country_code"]], master48[numeric_cols].astype(float)]
    if categorical_cols:
        encoded_parts.append(
            pd.get_dummies(master48[categorical_cols], prefix=categorical_cols, dtype=float)
        )

    team_feature_table = pd.concat(encoded_parts, axis=1).set_index("country_code")
    return team_feature_table, team_feature_table.columns.tolist()


def _build_training_df(
    all_matches: pd.DataFrame,
    team_feature_table: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    usable_teams = set(team_feature_table.index)
    team_feature_cols = team_feature_table.columns.tolist()
    delta_feature_cols = [f"delta_{col}" for col in team_feature_cols]
    feature_cols = [*delta_feature_cols, "is_knockout", "tournament_weight"]

    rows: list[dict[str, Any]] = []
    for _, match in all_matches.iterrows():
        home_team = match["home_code"]
        away_team = match["away_code"]
        if home_team not in usable_teams or away_team not in usable_teams:
            continue

        deltas = team_feature_table.loc[home_team] - team_feature_table.loc[away_team]
        row = {f"delta_{col}": float(deltas[col]) for col in team_feature_cols}
        row.update(
            {
                "year": int(match["year"]),
                "tournament": match["tournament"],
                "home_team": home_team,
                "away_team": away_team,
                "is_knockout": int(match["is_knockout"]),
                "tournament_weight": float(match["tournament_weight"]),
                "outcome": int(match["outcome"]),
            }
        )
        rows.append(row)

    train_df = pd.DataFrame(rows)
    retained = len(train_df)
    retained_by_tournament = train_df.groupby("tournament").size().to_dict()
    stats = {
        "all_matches": len(all_matches),
        "retained_matches": retained,
        "retained_ratio": retained / len(all_matches) if len(all_matches) else 0.0,
        "retained_by_tournament": retained_by_tournament,
    }
    return train_df, feature_cols, stats


def _get_loto_folds(
    train_df: pd.DataFrame,
    test_wc_years: list[int] | None = None,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    if test_wc_years is None:
        test_wc_years = [2002, 2006, 2010, 2014, 2018, 2022]

    years = train_df["year"].to_numpy()
    tournaments = train_df["tournament"].to_numpy()
    folds: list[tuple[np.ndarray, np.ndarray, int]] = []
    for test_year in test_wc_years:
        train_mask = years < test_year
        test_mask = (years == test_year) & (tournaments == "WC")
        if train_mask.sum() > 0 and test_mask.sum() > 0:
            folds.append((train_mask, test_mask, test_year))
    return folds


def prepare_experiment(
    master_path: str = "Data/master_dataset_48.csv",
    momios_path: str = "Data/momios_mundial2026.csv",
) -> PreparedMaster48Experiment:
    wc_matches = pd.read_csv(_repo_path("Data", "WorldCupHistory", "matches_1930_2022.csv"))
    rankings = pd.read_csv(_repo_path("Data", "ranking_mundial_2026_v2.csv"))
    copa = pd.read_csv(_repo_path("Data", "Copa_America.csv"))
    momios = pd.read_csv(_repo_path(*Path(momios_path).parts))

    euro_files = sorted(glob.glob(str(_repo_path("Data", "euro_histoia", "*.csv"))))
    euro_frames = []
    for path in euro_files:
        euro_frame = pd.read_csv(
            path,
            usecols=[
                "home_team",
                "away_team",
                "home_team_code",
                "away_team_code",
                "home_score",
                "away_score",
                "year",
                "date",
                "round",
            ],
        )
        euro_frames.append(euro_frame.dropna(subset=["home_score", "away_score"]))
    euro = pd.concat(euro_frames, ignore_index=True)

    master48_raw = pd.read_csv(_repo_path(*Path(master_path).parts))
    master48_raw["country_code"] = master48_raw["country_code"].map(_normalize_country_code)
    momios["country_code"] = momios["country_code"].map(_normalize_country_code)
    team_feature_table, team_feature_cols = _encode_master48_features(master48_raw)
    name_to_code = _build_name_to_code(rankings)
    all_matches = _unify_matches(wc_matches, euro, copa, name_to_code)
    train_df, feature_cols, retained_match_stats = _build_training_df(all_matches, team_feature_table)
    folds = _get_loto_folds(train_df)

    teams_2026 = momios["country_code"].tolist()
    groups_2026 = {
        group_letter: momios[momios["group_letter"] == group_letter]["country_code"].tolist()
        for group_letter in momios["group_letter"].unique()
    }

    return PreparedMaster48Experiment(
        master48_raw=master48_raw,
        team_feature_table=team_feature_table,
        team_feature_cols=team_feature_cols,
        all_matches=all_matches,
        train_df=train_df,
        feature_cols=feature_cols,
        folds=folds,
        teams_2026=teams_2026,
        groups_2026=groups_2026,
        momios=momios,
        retained_match_stats=retained_match_stats,
    )


def _base_xgb_params(seed: int = SEED) -> dict[str, Any]:
    return {
        "objective": "multi:softprob",
        "num_class": 3,
        "random_state": seed,
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
        "verbosity": 0,
    }


def tune_model(
    experiment: PreparedMaster48Experiment,
    n_trials: int = 60,
    inner_year_max: int = 2014,
    seed: int = SEED,
) -> tuple[dict[str, Any], optuna.Study]:
    X = experiment.train_df[experiment.feature_cols].to_numpy()
    y = experiment.train_df["outcome"].to_numpy()
    inner_folds = [fold for fold in experiment.folds if fold[2] <= inner_year_max]
    if not inner_folds:
        raise ValueError("No inner folds available for tuning.")

    def objective(trial: optuna.Trial) -> float:
        params = {
            **_base_xgb_params(seed),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }

        losses = []
        for train_mask, test_mask, _ in inner_folds:
            model = xgb.XGBClassifier(**params)
            model.fit(X[train_mask], y[train_mask], eval_set=[(X[test_mask], y[test_mask])], verbose=False)
            proba = model.predict_proba(X[test_mask])
            losses.append(log_loss(y[test_mask], proba))
        return float(np.mean(losses))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {**_base_xgb_params(seed), **study.best_params}, study


def evaluate_model(
    experiment: PreparedMaster48Experiment,
    params: dict[str, Any],
) -> pd.DataFrame:
    X = experiment.train_df[experiment.feature_cols].to_numpy()
    y = experiment.train_df["outcome"].to_numpy()

    rows = []
    for train_mask, test_mask, test_year in experiment.folds:
        model = xgb.XGBClassifier(**params)
        model.fit(X[train_mask], y[train_mask], eval_set=[(X[test_mask], y[test_mask])], verbose=False)
        proba = model.predict_proba(X[test_mask])
        pred = model.predict(X[test_mask])
        rows.append(
            {
                "test_year": test_year,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "log_loss": float(log_loss(y[test_mask], proba)),
                "accuracy": float(accuracy_score(y[test_mask], pred)),
            }
        )
    return pd.DataFrame(rows)


def fit_final_model(
    experiment: PreparedMaster48Experiment,
    params: dict[str, Any],
) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(**params)
    X = experiment.train_df[experiment.feature_cols].to_numpy()
    y = experiment.train_df["outcome"].to_numpy()
    model.fit(X, y)
    return model


def _pairwise_vector(
    experiment: PreparedMaster48Experiment,
    team_a: str,
    team_b: str,
    is_knockout: int,
) -> np.ndarray:
    deltas = experiment.team_feature_table.loc[team_a] - experiment.team_feature_table.loc[team_b]
    return np.array(
        [*deltas.to_numpy(dtype=float), float(is_knockout), 1.0],
        dtype=float,
    ).reshape(1, -1)


def precompute_neutral_probabilities(
    experiment: PreparedMaster48Experiment,
    model: xgb.XGBClassifier,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], np.ndarray]]:
    pairs = [(a, b) for a in experiment.teams_2026 for b in experiment.teams_2026 if a != b]
    group_vectors = np.vstack([_pairwise_vector(experiment, a, b, 0) for a, b in pairs])
    knockout_vectors = np.vstack([_pairwise_vector(experiment, a, b, 1) for a, b in pairs])

    group_raw = model.predict_proba(group_vectors)
    knockout_raw = model.predict_proba(knockout_vectors)
    raw_group = {pair: group_raw[idx] for idx, pair in enumerate(pairs)}
    raw_knockout = {pair: knockout_raw[idx] for idx, pair in enumerate(pairs)}

    group_cache: dict[tuple[str, str], np.ndarray] = {}
    knockout_cache: dict[tuple[str, str], np.ndarray] = {}
    for team_a in experiment.teams_2026:
        for team_b in experiment.teams_2026:
            if team_a == team_b:
                continue

            group_ab = raw_group[(team_a, team_b)]
            group_ba = raw_group[(team_b, team_a)]
            avg_a_win = (group_ab[2] + group_ba[0]) / 2
            avg_draw = (group_ab[1] + group_ba[1]) / 2
            avg_b_win = (group_ab[0] + group_ba[2]) / 2
            total = avg_a_win + avg_draw + avg_b_win
            group_cache[(team_a, team_b)] = np.array(
                [avg_b_win / total, avg_draw / total, avg_a_win / total]
            )

            knockout_ab = raw_knockout[(team_a, team_b)]
            knockout_ba = raw_knockout[(team_b, team_a)]
            avg_a_win = (knockout_ab[2] + knockout_ba[0]) / 2
            avg_draw = (knockout_ab[1] + knockout_ba[1]) / 2
            avg_b_win = (knockout_ab[0] + knockout_ba[2]) / 2
            total = avg_a_win + avg_draw + avg_b_win
            knockout_cache[(team_a, team_b)] = np.array(
                [avg_b_win / total, avg_draw / total, avg_a_win / total]
            )

    return group_cache, knockout_cache


def sample_goals(outcome: int, rng: np.random.Generator) -> tuple[int, int]:
    if outcome == 2:
        goals_for = max(1, int(rng.poisson(1.8)))
        goals_against = min(goals_for - 1, max(0, int(rng.poisson(0.7))))
    elif outcome == 0:
        goals_against = max(1, int(rng.poisson(1.8)))
        goals_for = min(goals_against - 1, max(0, int(rng.poisson(0.7))))
    else:
        draw_goals = max(0, int(rng.poisson(0.9)))
        goals_for, goals_against = draw_goals, draw_goals
    return goals_for, goals_against


def simulate_group(
    group_teams: list[str],
    group_cache: dict[tuple[str, str], np.ndarray],
    rng: np.random.Generator,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    standings = {team: {"pts": 0, "gd": 0, "gf": 0} for team in group_teams}
    for home_idx in range(len(group_teams)):
        for away_idx in range(home_idx + 1, len(group_teams)):
            home_team = group_teams[home_idx]
            away_team = group_teams[away_idx]
            proba = group_cache[(home_team, away_team)]
            outcome = int(rng.choice([0, 1, 2], p=proba))
            goals_for, goals_against = sample_goals(outcome, rng)

            if outcome == 2:
                standings[home_team]["pts"] += 3
            elif outcome == 0:
                standings[away_team]["pts"] += 3
            else:
                standings[home_team]["pts"] += 1
                standings[away_team]["pts"] += 1

            standings[home_team]["gd"] += goals_for - goals_against
            standings[home_team]["gf"] += goals_for
            standings[away_team]["gd"] += goals_against - goals_for
            standings[away_team]["gf"] += goals_against

    ordered_teams = sorted(
        group_teams,
        key=lambda team: (standings[team]["pts"], standings[team]["gd"], standings[team]["gf"]),
        reverse=True,
    )
    return ordered_teams, standings


def get_advancing_teams(
    group_results: dict[str, tuple[list[str], dict[str, dict[str, int]]]],
) -> tuple[list[str], list[tuple[str, str]]]:
    top2: list[str] = []
    thirds: list[tuple[str, dict[str, int], str]] = []
    for group_letter in sorted(group_results):
        ordered_teams, standings = group_results[group_letter]
        top2.extend(ordered_teams[:2])
        thirds.append((ordered_teams[2], standings[ordered_teams[2]], group_letter))

    thirds.sort(key=lambda item: (item[1]["pts"], item[1]["gd"], item[1]["gf"]), reverse=True)
    best_thirds = [(team, group_letter) for team, _, group_letter in thirds[:8]]
    return top2, best_thirds


def simulate_knockout_match(
    team_a: str,
    team_b: str,
    knockout_cache: dict[tuple[str, str], np.ndarray],
    rng: np.random.Generator,
) -> str:
    proba = knockout_cache[(team_a, team_b)]
    outcome = int(rng.choice([0, 1, 2], p=proba))
    if outcome == 2:
        return team_a
    if outcome == 0:
        return team_b

    win_total = proba[0] + proba[2]
    p_team_a = proba[2] / win_total if win_total > 0 else 0.5
    return team_a if rng.random() < p_team_a else team_b


def build_knockout_bracket(
    groups_2026: dict[str, list[str]],
    top2: list[str],
    best_thirds: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    group_letters = sorted(groups_2026)
    winners = {group_letters[idx]: top2[idx * 2] for idx in range(12)}
    runners = {group_letters[idx]: top2[idx * 2 + 1] for idx in range(12)}
    thirds_by_group = {group_letter: team for team, group_letter in best_thirds}
    qualified_third_groups = set(thirds_by_group)

    third_slot_allowed = {
        "A": {"C", "E", "F", "H", "I"},
        "B": {"E", "F", "G", "I", "J"},
        "D": {"B", "E", "F", "I", "J"},
        "E": {"A", "B", "C", "D", "F"},
        "G": {"A", "E", "H", "I", "J"},
        "I": {"C", "D", "F", "G", "H"},
        "K": {"D", "E", "I", "J", "L"},
        "L": {"E", "H", "I", "J", "K"},
    }

    slot_candidates = {
        slot: sorted(groups & qualified_third_groups)
        for slot, groups in third_slot_allowed.items()
    }
    if any(not candidates for candidates in slot_candidates.values()):
        raise ValueError("No valid third-place candidates for one or more round-of-32 slots.")

    slot_order = sorted(slot_candidates, key=lambda slot: (len(slot_candidates[slot]), slot))
    used_groups: set[str] = set()
    slot_assignment: dict[str, str] = {}

    def backtrack(slot_idx: int = 0) -> bool:
        if slot_idx == len(slot_order):
            return True

        current_slot = slot_order[slot_idx]
        for candidate_group in slot_candidates[current_slot]:
            if candidate_group in used_groups:
                continue

            used_groups.add(candidate_group)
            slot_assignment[current_slot] = candidate_group
            feasible = True
            for future_slot in slot_order[slot_idx + 1 :]:
                if not any(group not in used_groups for group in slot_candidates[future_slot]):
                    feasible = False
                    break

            if feasible and backtrack(slot_idx + 1):
                return True

            used_groups.remove(candidate_group)
            slot_assignment.pop(current_slot, None)
        return False

    if not backtrack():
        raise ValueError(f"Unable to assign third-place groups: {sorted(qualified_third_groups)}")

    pair_templates = [
        ("2A", "2B"),
        ("1C", "2F"),
        ("1E", ("3slot", "E")),
        ("1F", "2C"),
        ("2E", "2I"),
        ("1I", ("3slot", "I")),
        ("1A", ("3slot", "A")),
        ("1L", ("3slot", "L")),
        ("1G", ("3slot", "G")),
        ("1D", ("3slot", "D")),
        ("1H", "2J"),
        ("2K", "2L"),
        ("1B", ("3slot", "B")),
        ("2D", "2G"),
        ("1J", "2H"),
        ("1K", ("3slot", "K")),
    ]

    def resolve_slot(slot: str | tuple[str, str]) -> str:
        if isinstance(slot, tuple):
            assigned_group = slot_assignment[slot[1]]
            return thirds_by_group[assigned_group]

        place, group_letter = slot[0], slot[1]
        return winners[group_letter] if place == "1" else runners[group_letter]

    bracket = [(resolve_slot(slot_a), resolve_slot(slot_b)) for slot_a, slot_b in pair_templates]
    flat_teams = [team for match in bracket for team in match]
    if len(bracket) != 16 or len(set(flat_teams)) != 32:
        raise ValueError("Invalid round-of-32 bracket composition.")
    return bracket


def simulate_knockout_rounds(
    r32_matches: list[tuple[str, str]],
    knockout_cache: dict[tuple[str, str], np.ndarray],
    rng: np.random.Generator,
) -> tuple[str, str, dict[str, set[str]]]:
    current_round = r32_matches
    round_results: dict[str, set[str]] = defaultdict(set)
    while len(current_round) > 1:
        round_winners = [
            simulate_knockout_match(team_a, team_b, knockout_cache, rng)
            for team_a, team_b in current_round
        ]
        current_round = [
            (round_winners[idx], round_winners[idx + 1])
            for idx in range(0, len(round_winners), 2)
        ]
        if len(current_round) == 2:
            for team_a, team_b in current_round:
                round_results["semifinal"].add(team_a)
                round_results["semifinal"].add(team_b)
        elif len(current_round) == 1:
            round_results["final"].add(current_round[0][0])
            round_results["final"].add(current_round[0][1])

    champion = simulate_knockout_match(
        current_round[0][0], current_round[0][1], knockout_cache, rng
    )
    finalist = current_round[0][1] if champion == current_round[0][0] else current_round[0][0]
    return champion, finalist, round_results


def simulate_tournament(
    experiment: PreparedMaster48Experiment,
    model: xgb.XGBClassifier,
    n_simulations: int = 50_000,
    seed: int = SEED,
    output_path: str | None = "Data/simulation_results_master48.csv",
) -> pd.DataFrame:
    group_cache, knockout_cache = precompute_neutral_probabilities(experiment, model)
    rng = np.random.default_rng(seed)

    champion_counts: dict[str, int] = defaultdict(int)
    finalist_counts: dict[str, int] = defaultdict(int)
    semifinal_counts: dict[str, int] = defaultdict(int)
    group_exit_counts: dict[str, int] = defaultdict(int)

    valid_sims = 0
    for _ in range(n_simulations):
        group_results = {}
        for group_letter, group_teams in experiment.groups_2026.items():
            group_results[group_letter] = simulate_group(group_teams, group_cache, rng)

        top2, best_thirds = get_advancing_teams(group_results)
        r32_matches = build_knockout_bracket(experiment.groups_2026, top2, best_thirds)
        advancing = set(top2 + [team for team, _ in best_thirds])
        for team in experiment.teams_2026:
            if team not in advancing:
                group_exit_counts[team] += 1

        champion, finalist, round_results = simulate_knockout_rounds(r32_matches, knockout_cache, rng)
        champion_counts[champion] += 1
        finalist_counts[finalist] += 1
        for team in round_results.get("semifinal", set()):
            semifinal_counts[team] += 1
        valid_sims += 1

    results = []
    momios_lookup = dict(zip(experiment.momios["country_code"], experiment.momios["champion_prob"] * 100))
    for team in experiment.teams_2026:
        results.append(
            {
                "country_code": team,
                "champion_pct": champion_counts[team] / valid_sims * 100 if valid_sims else 0.0,
                "finalist_pct": (champion_counts[team] + finalist_counts[team]) / valid_sims * 100
                if valid_sims
                else 0.0,
                "semifinal_pct": semifinal_counts[team] / valid_sims * 100 if valid_sims else 0.0,
                "group_exit_pct": group_exit_counts[team] / valid_sims * 100 if valid_sims else 0.0,
                "momios_pct": momios_lookup.get(team, np.nan),
            }
        )

    results_df = pd.DataFrame(results).sort_values("champion_pct", ascending=False).reset_index(drop=True)
    results_df.index += 1
    if output_path:
        resolved_output = _repo_path(*Path(output_path).parts)
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(resolved_output, index=False)
    return results_df


def compare_with_existing_results(
    results_df: pd.DataFrame,
    existing_path: str = "Data/simulation_results.csv",
) -> pd.DataFrame | None:
    existing_file = _repo_path(*Path(existing_path).parts)
    if not existing_file.exists():
        return None

    current_df = pd.read_csv(existing_file)[["country_code", "champion_pct"]].rename(
        columns={"champion_pct": "current_model_pct"}
    )
    merged = results_df.merge(current_df, on="country_code", how="left")
    merged["delta_vs_current"] = merged["champion_pct"] - merged["current_model_pct"]
    return merged.sort_values("champion_pct", ascending=False).reset_index(drop=True)


def run_pipeline(
    n_trials: int = 60,
    n_simulations: int = 50_000,
    simulation_output_path: str = "Data/simulation_results_master48.csv",
) -> dict[str, Any]:
    experiment = prepare_experiment()
    best_params, study = tune_model(experiment, n_trials=n_trials)
    cv_df = evaluate_model(experiment, best_params)
    final_model = fit_final_model(experiment, best_params)
    results_df = simulate_tournament(
        experiment,
        final_model,
        n_simulations=n_simulations,
        output_path=simulation_output_path,
    )
    comparison_df = compare_with_existing_results(results_df)
    return {
        "experiment": experiment,
        "best_params": best_params,
        "study": study,
        "cv_df": cv_df,
        "final_model": final_model,
        "results_df": results_df,
        "comparison_df": comparison_df,
    }


if __name__ == "__main__":
    artifacts = run_pipeline()
    experiment = artifacts["experiment"]
    cv_df = artifacts["cv_df"]
    results_df = artifacts["results_df"]

    print("Master48 alternative experiment")
    print(
        f"Retained matches: {experiment.retained_match_stats['retained_matches']} / "
        f"{experiment.retained_match_stats['all_matches']} "
        f"({experiment.retained_match_stats['retained_ratio']:.1%})"
    )
    print(cv_df.to_string(index=False))
    print("\nTop 10 simulation results")
    print(results_df.head(10).to_string(index=False))

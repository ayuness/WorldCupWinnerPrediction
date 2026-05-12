from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import glob

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent if THIS_DIR.name == "Models" else THIS_DIR
ARTIFACTS_DIR = THIS_DIR / "artifacts"
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "current_temporal_master_model.xgb.json"
DEFAULT_METADATA_PATH = ARTIFACTS_DIR / "current_temporal_master_model.metadata.json"
SEED = 42
K_FACTOR = 40
ELO_INIT = 1500
COUNTRY_CODE_NORMALIZATION = {
    "CGO": "COD",
}


def _repo_path(*parts: str) -> Path:
    return ROOT_DIR.joinpath(*parts)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass
class PreparedCurrentExperiment:
    master64_raw: pd.DataFrame
    momios: pd.DataFrame
    worldcups: pd.DataFrame
    all_matches: pd.DataFrame
    train_df: pd.DataFrame
    feature_cols: list[str]
    folds: list[tuple[np.ndarray, np.ndarray, int]]
    teams_2026: list[str]
    groups_2026: dict[str, list[str]]
    elo_final: dict[str, float]
    team_histories: dict[str, list[tuple[pd.Timestamp, str, int, int]]]
    player_features: dict[tuple[str, int], dict[str, float]]
    player_years: list[int]
    temporal_master_features: dict[tuple[str, int], dict[str, float]]
    temporal_master_years: list[int]
    temporal_master_defaults: dict[int, dict[str, float]]


def _normalize_country_code(code: Any) -> str | Any:
    if pd.isna(code):
        return code
    return COUNTRY_CODE_NORMALIZATION.get(str(code), str(code))


def _normalize_country_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = df[col].map(_normalize_country_code)
    return df


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
    return {name: _normalize_country_code(code) for name, code in name_to_code.items()}


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


def compute_elo_history(matches_df: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    elo_current = defaultdict(lambda: ELO_INIT)
    elo_snapshots = []

    for _, match in matches_df.iterrows():
        home = match["home_code"]
        away = match["away_code"]
        home_elo = elo_current[home]
        away_elo = elo_current[away]

        exp_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        exp_away = 1 - exp_home

        outcome = int(match["outcome"])
        if outcome == 2:
            actual_home, actual_away = 1.0, 0.0
        elif outcome == 0:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5

        elo_snapshots.append(
            {
                "date": match["date"],
                "year": int(match["year"]),
                "home_code": home,
                "away_code": away,
                "home_elo_before": home_elo,
                "away_elo_before": away_elo,
                "tournament": match["tournament"],
                "outcome": outcome,
            }
        )

        k = K_FACTOR * float(match.get("tournament_weight", 1.0))
        elo_current[home] += k * (actual_home - exp_home)
        elo_current[away] += k * (actual_away - exp_away)

    return dict(elo_current), pd.DataFrame(elo_snapshots)


def _compute_wc_cumulative(
    wc_matches: pd.DataFrame,
    worldcups: pd.DataFrame,
    winner_name_to_code: dict[str, str],
    name_to_code: dict[str, str],
) -> tuple[list[int], dict[tuple[str, int], dict[str, float]], dict[tuple[str, int], int]]:
    wc_years = sorted(wc_matches["Year"].unique())
    wc_team_accum = defaultdict(lambda: {"wins": 0, "draws": 0, "losses": 0, "gf": 0, "gc": 0, "games": 0})
    wc_cumulative: dict[tuple[str, int], dict[str, float]] = {}
    wc_titles_cumulative: dict[tuple[str, int], int] = {}
    titles_by_team = defaultdict(int)
    all_wc_teams_ever: set[str] = set()

    for year in wc_years:
        year_teams = set(wc_matches[wc_matches["Year"] == year]["home_code"].dropna()) | set(
            wc_matches[wc_matches["Year"] == year]["away_code"].dropna()
        )
        all_wc_teams_ever |= year_teams

        for team in all_wc_teams_ever:
            acc = wc_team_accum[team]
            games = acc["games"]
            wc_cumulative[(team, int(year))] = {
                "wc_win_pct": acc["wins"] / games if games > 0 else 0.0,
                "wc_gc_per_game": acc["gc"] / games if games > 0 else 0.0,
                "wc_experience": games,
            }
            wc_titles_cumulative[(team, int(year))] = titles_by_team[team]

        for _, row in wc_matches[wc_matches["Year"] == year].iterrows():
            home = row.get("home_code")
            away = row.get("away_code")
            if pd.isna(home) or pd.isna(away):
                continue
            home_score = int(row["home_score"])
            away_score = int(row["away_score"])
            wc_team_accum[home]["games"] += 1
            wc_team_accum[away]["games"] += 1
            wc_team_accum[home]["gf"] += home_score
            wc_team_accum[home]["gc"] += away_score
            wc_team_accum[away]["gf"] += away_score
            wc_team_accum[away]["gc"] += home_score
            if home_score > away_score:
                wc_team_accum[home]["wins"] += 1
                wc_team_accum[away]["losses"] += 1
            elif home_score < away_score:
                wc_team_accum[away]["wins"] += 1
                wc_team_accum[home]["losses"] += 1
            else:
                wc_team_accum[home]["draws"] += 1
                wc_team_accum[away]["draws"] += 1

        winner_row = worldcups[worldcups["year"] == year]
        if not winner_row.empty:
            winner_name = winner_row.iloc[0]["winner"]
            winner_code = winner_name_to_code.get(winner_name, name_to_code.get(winner_name))
            if winner_code:
                titles_by_team[_normalize_country_code(winner_code)] += 1

    return wc_years, wc_cumulative, wc_titles_cumulative


def compute_recent_form(
    matches_df: pd.DataFrame,
    window: int = 10,
) -> tuple[pd.DataFrame, dict[str, list[tuple[pd.Timestamp, str, int, int]]]]:
    team_match_history: dict[str, list[tuple[pd.Timestamp, str, int, int]]] = defaultdict(list)
    form_data = []

    for _, match in matches_df.iterrows():
        home = match["home_code"]
        away = match["away_code"]

        home_recent = team_match_history[home][-window:]
        if len(home_recent) >= 3:
            home_form_wp = sum(1 for _, outcome, _, _ in home_recent if outcome == "W") / len(home_recent)
            home_form_gf = float(np.mean([gf for _, _, gf, _ in home_recent]))
            home_form_gc = float(np.mean([gc for _, _, _, gc in home_recent]))
        else:
            home_form_wp, home_form_gf, home_form_gc = 0.33, 1.0, 1.0

        away_recent = team_match_history[away][-window:]
        if len(away_recent) >= 3:
            away_form_wp = sum(1 for _, outcome, _, _ in away_recent if outcome == "W") / len(away_recent)
            away_form_gf = float(np.mean([gf for _, _, gf, _ in away_recent]))
            away_form_gc = float(np.mean([gc for _, _, _, gc in away_recent]))
        else:
            away_form_wp, away_form_gf, away_form_gc = 0.33, 1.0, 1.0

        form_data.append(
            {
                "home_form_wp": home_form_wp,
                "home_form_gf": home_form_gf,
                "home_form_gc": home_form_gc,
                "away_form_wp": away_form_wp,
                "away_form_gf": away_form_gf,
                "away_form_gc": away_form_gc,
            }
        )

        home_score = int(match["home_score"])
        away_score = int(match["away_score"])
        if home_score > away_score:
            team_match_history[home].append((match["date"], "W", home_score, away_score))
            team_match_history[away].append((match["date"], "L", away_score, home_score))
        elif home_score < away_score:
            team_match_history[home].append((match["date"], "L", home_score, away_score))
            team_match_history[away].append((match["date"], "W", away_score, home_score))
        else:
            team_match_history[home].append((match["date"], "D", home_score, away_score))
            team_match_history[away].append((match["date"], "D", away_score, home_score))

    return pd.DataFrame(form_data), dict(team_match_history)


def compute_player_features_all(df: pd.DataFrame) -> dict[tuple[str, int], dict[str, float]]:
    features: dict[tuple[str, int], dict[str, float]] = {}
    for (code, year), group in df.groupby(["country_code", "year"]):
        values = group["market_value_eur_m"].dropna().sort_values(ascending=False)
        if len(values) < 3:
            continue
        total_value = float(values.sum())
        star_concentration = float(values.head(3).sum() / total_value) if total_value > 0 else 0.0
        top5_avg_value = float(values.head(5).mean()) if len(values) > 0 else 0.0
        gk_vals = group[group["position"] == "GK"]["market_value_eur_m"].dropna()
        gk_value = float(gk_vals.max()) if len(gk_vals) > 0 else 0.0
        squad_depth_ratio = float(values.median() / values.mean()) if values.mean() > 0 else 0.0
        age_weighted = 0.0
        for _, player in group.dropna(subset=["market_value_eur_m", "age"]).iterrows():
            weight = max(0.0, (30 - float(player["age"])) / 10)
            age_weighted += float(player["market_value_eur_m"]) * weight
        top_league_pct = float(group["is_top_league"].mean()) if "is_top_league" in group.columns else 0.0
        features[(_normalize_country_code(code), int(year))] = {
            "star_concentration": star_concentration,
            "top5_avg_value": top5_avg_value,
            "gk_value": gk_value,
            "squad_depth_ratio": squad_depth_ratio,
            "age_weighted_value": age_weighted,
            "top_league_pct": top_league_pct,
        }
    return features


def compute_temporal_master_features(
    rankings: pd.DataFrame,
    players_df: pd.DataFrame,
) -> tuple[dict[tuple[str, int], dict[str, float]], list[int], dict[int, dict[str, float]]]:
    ranking_df = rankings[["country_code", "year", "ranking"]].dropna().copy()
    ranking_df["country_code"] = ranking_df["country_code"].map(_normalize_country_code)
    ranking_df["year"] = ranking_df["year"].astype(int)
    ranking_df["ranking"] = ranking_df["ranking"].astype(float)
    max_rank = max(int(ranking_df["ranking"].max()), 2)
    ranking_df = ranking_df.sort_values(["country_code", "year"]).reset_index(drop=True)
    ranking_df["rank_strength"] = 1.0 - (ranking_df["ranking"] - 1.0) / (max_rank - 1)
    ranking_df["rank_volatility"] = (
        ranking_df.groupby("country_code")["ranking"]
        .transform(lambda series: series.rolling(window=4, min_periods=2).std().fillna(0.0))
        .astype(float)
    )

    player_yearly = (
        players_df.groupby(["country_code", "year"], as_index=False)
        .agg(
            squad_market_value_eur_m=("market_value_eur_m", "sum"),
            squad_avg_age=("age", "mean"),
        )
        .copy()
    )
    player_yearly["country_code"] = player_yearly["country_code"].map(_normalize_country_code)
    player_yearly["year"] = player_yearly["year"].astype(int)
    player_yearly["squad_market_value_log"] = np.log1p(
        player_yearly["squad_market_value_eur_m"].fillna(0.0)
    )
    player_yearly["squad_avg_age"] = player_yearly["squad_avg_age"].astype(float)

    years = sorted(set(ranking_df["year"].tolist()) | set(player_yearly["year"].tolist()))
    rank_vol_global = float(ranking_df["rank_volatility"].median()) if len(ranking_df) else 0.0
    squad_value_global = (
        float(player_yearly["squad_market_value_log"].median()) if len(player_yearly) else 0.0
    )
    squad_age_global = float(player_yearly["squad_avg_age"].mean()) if len(player_yearly) else 27.0

    defaults_by_year: dict[int, dict[str, float]] = {}
    for year in years:
        rank_year = ranking_df[ranking_df["year"] == year]
        player_year = player_yearly[player_yearly["year"] == year]
        defaults_by_year[int(year)] = {
            "rank_strength": 0.5,
            "rank_volatility": float(rank_year["rank_volatility"].median()) if len(rank_year) else rank_vol_global,
            "squad_market_value_log": float(player_year["squad_market_value_log"].median())
            if len(player_year)
            else squad_value_global,
            "squad_avg_age": float(player_year["squad_avg_age"].mean()) if len(player_year) else squad_age_global,
        }

    features: dict[tuple[str, int], dict[str, float]] = {}
    for _, row in ranking_df.iterrows():
        key = (row["country_code"], int(row["year"]))
        features.setdefault(key, {}).update(
            {
                "rank_strength": float(row["rank_strength"]),
                "rank_volatility": float(row["rank_volatility"]),
                "rank_available": 1.0,
            }
        )
    for _, row in player_yearly.iterrows():
        key = (row["country_code"], int(row["year"]))
        features.setdefault(key, {}).update(
            {
                "squad_market_value_log": float(row["squad_market_value_log"]),
                "squad_avg_age": float(row["squad_avg_age"]),
                "squad_available": 1.0,
            }
        )

    return features, years, defaults_by_year


def get_player_features(
    player_features: dict[tuple[str, int], dict[str, float]],
    player_years: list[int],
    team_code: str,
    year: int,
) -> dict[str, float]:
    for candidate_year in reversed(player_years):
        if candidate_year <= year and (team_code, candidate_year) in player_features:
            return player_features[(team_code, candidate_year)]
    return {
        "star_concentration": 0.0,
        "top5_avg_value": 0.0,
        "gk_value": 0.0,
        "squad_depth_ratio": 0.0,
        "age_weighted_value": 0.0,
        "top_league_pct": 0.0,
    }


def get_temporal_master_features(
    temporal_master_features: dict[tuple[str, int], dict[str, float]],
    temporal_master_years: list[int],
    temporal_master_defaults: dict[int, dict[str, float]],
    team_code: str,
    year: int,
) -> dict[str, float]:
    default_year = None
    result = None
    found_rank = False
    found_squad = False
    for candidate_year in reversed(temporal_master_years):
        if candidate_year <= year:
            if default_year is None:
                default_year = candidate_year
                result = {
                    **temporal_master_defaults[candidate_year],
                    "rank_available": 0.0,
                    "squad_available": 0.0,
                }
            row = temporal_master_features.get((team_code, candidate_year))
            if not row:
                continue
            if not found_rank and row.get("rank_available", 0.0):
                result["rank_strength"] = row["rank_strength"]
                result["rank_volatility"] = row["rank_volatility"]
                result["rank_available"] = 1.0
                found_rank = True
            if not found_squad and row.get("squad_available", 0.0):
                result["squad_market_value_log"] = row["squad_market_value_log"]
                result["squad_avg_age"] = row["squad_avg_age"]
                result["squad_available"] = 1.0
                found_squad = True
            if found_rank and found_squad:
                return result
    if default_year is None:
        default_year = temporal_master_years[-1]
        result = {
            **temporal_master_defaults[default_year],
            "rank_available": 0.0,
            "squad_available": 0.0,
        }
    return result


def _get_loto_folds(train_df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, int]]:
    years = train_df["year"].to_numpy()
    tournaments = train_df["tournament"].to_numpy()
    test_wc_years = [2002, 2006, 2010, 2014, 2018, 2022]
    folds: list[tuple[np.ndarray, np.ndarray, int]] = []
    for test_year in test_wc_years:
        train_mask = years < test_year
        test_mask = (years == test_year) & (tournaments == "WC")
        if train_mask.sum() > 0 and test_mask.sum() > 0:
            folds.append((train_mask, test_mask, test_year))
    return folds


def prepare_experiment(
    master_path: str = "Data/master_dataset_64.csv",
    momios_path: str = "Data/momios_mundial2026.csv",
) -> PreparedCurrentExperiment:
    wc_matches = pd.read_csv(_repo_path("Data", "WorldCupHistory", "matches_1930_2022.csv"))
    worldcups = pd.read_csv(_repo_path("Data", "WorldCupHistory", "worldcups.csv"))
    rankings = pd.read_csv(_repo_path("Data", "ranking_mundial_2026_v2.csv"))
    master = pd.read_csv(_repo_path(*Path(master_path).parts))
    momios = pd.read_csv(_repo_path(*Path(momios_path).parts))
    copa_path = _repo_path("Data", "Copa_America.csv")
    if not copa_path.exists():
        copa_path = _repo_path("Data", "Copa_America", "Copa_America.csv")
    copa = pd.read_csv(copa_path)
    players_df = pd.read_csv(_repo_path("Data", "transfermarkt_players.csv"))

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

    rankings = _normalize_country_column(rankings, "country_code")
    master = _normalize_country_column(master, "country_code")
    momios = _normalize_country_column(momios, "country_code")
    players_df = _normalize_country_column(players_df, "country_code")

    name_to_code = _build_name_to_code(rankings)
    winner_name_to_code = {
        "Uruguay": "URU",
        "Italy": "ITA",
        "Brazil": "BRA",
        "West Germany": "GER",
        "England": "ENG",
        "Argentina": "ARG",
        "Germany": "GER",
        "France": "FRA",
        "Spain": "ESP",
    }

    wc_matches["home_code"] = wc_matches["home_team"].map(name_to_code)
    wc_matches["away_code"] = wc_matches["away_team"].map(name_to_code)

    all_matches = _unify_matches(wc_matches, euro, copa, name_to_code)
    elo_final, elo_history = compute_elo_history(all_matches)
    wc_years, wc_cumulative, wc_titles_cumulative = _compute_wc_cumulative(
        wc_matches,
        worldcups,
        winner_name_to_code,
        name_to_code,
    )
    form_df, team_histories = compute_recent_form(all_matches, window=10)

    player_features = compute_player_features_all(players_df)
    player_years = sorted({year for _, year in player_features})
    temporal_master_features, temporal_master_years, temporal_master_defaults = compute_temporal_master_features(
        rankings,
        players_df,
    )

    feature_cols = [
        "delta_elo",
        "delta_form_wp",
        "delta_form_gf",
        "delta_form_gc",
        "delta_wc_win_pct",
        "delta_wc_gc_per_game",
        "delta_wc_titles",
        "delta_wc_experience",
        "is_knockout",
        "tournament_weight",
        "delta_rank_strength",
        "delta_rank_volatility",
        "delta_squad_market_value_log",
        "delta_squad_avg_age",
        "both_rank_available",
        "both_squad_available",
        "delta_star_concentration",
        "delta_top5_avg_value",
        "delta_gk_value",
        "delta_squad_depth_ratio",
        "delta_age_weighted_value",
        "delta_top_league_pct",
    ]

    def get_wc_stats(team: str, year: int) -> tuple[dict[str, float], int]:
        for candidate_year in reversed(wc_years):
            if candidate_year <= year and (team, candidate_year) in wc_cumulative:
                return wc_cumulative[(team, candidate_year)], wc_titles_cumulative.get((team, candidate_year), 0)
        return {"wc_win_pct": 0.0, "wc_gc_per_game": 0.0, "wc_experience": 0}, 0

    rows = []
    for idx, (_, match) in enumerate(all_matches.iterrows()):
        year = int(match["year"])
        home = match["home_code"]
        away = match["away_code"]

        elo_row = elo_history.iloc[idx]
        form_row = form_df.iloc[idx]
        home_wc, home_titles = get_wc_stats(home, year)
        away_wc, away_titles = get_wc_stats(away, year)
        home_pf = get_player_features(player_features, player_years, home, year)
        away_pf = get_player_features(player_features, player_years, away, year)
        home_tm = get_temporal_master_features(
            temporal_master_features,
            temporal_master_years,
            temporal_master_defaults,
            home,
            year,
        )
        away_tm = get_temporal_master_features(
            temporal_master_features,
            temporal_master_years,
            temporal_master_defaults,
            away,
            year,
        )

        rows.append(
            {
                "year": year,
                "home_team": home,
                "away_team": away,
                "tournament": match["tournament"],
                "delta_elo": float(elo_row["home_elo_before"] - elo_row["away_elo_before"]),
                "delta_form_wp": float(form_row["home_form_wp"] - form_row["away_form_wp"]),
                "delta_form_gf": float(form_row["home_form_gf"] - form_row["away_form_gf"]),
                "delta_form_gc": float(form_row["home_form_gc"] - form_row["away_form_gc"]),
                "delta_wc_win_pct": float(home_wc["wc_win_pct"] - away_wc["wc_win_pct"]),
                "delta_wc_gc_per_game": float(home_wc["wc_gc_per_game"] - away_wc["wc_gc_per_game"]),
                "delta_wc_titles": int(home_titles - away_titles),
                "delta_wc_experience": int(home_wc["wc_experience"] - away_wc["wc_experience"]),
                "is_knockout": int(match["is_knockout"]),
                "tournament_weight": float(match["tournament_weight"]),
                "delta_rank_strength": float(home_tm["rank_strength"] - away_tm["rank_strength"]),
                "delta_rank_volatility": float(home_tm["rank_volatility"] - away_tm["rank_volatility"]),
                "delta_squad_market_value_log": float(
                    home_tm["squad_market_value_log"] - away_tm["squad_market_value_log"]
                ),
                "delta_squad_avg_age": float(home_tm["squad_avg_age"] - away_tm["squad_avg_age"]),
                "both_rank_available": int(home_tm["rank_available"] and away_tm["rank_available"]),
                "both_squad_available": int(home_tm["squad_available"] and away_tm["squad_available"]),
                "delta_star_concentration": float(home_pf["star_concentration"] - away_pf["star_concentration"]),
                "delta_top5_avg_value": float(home_pf["top5_avg_value"] - away_pf["top5_avg_value"]),
                "delta_gk_value": float(home_pf["gk_value"] - away_pf["gk_value"]),
                "delta_squad_depth_ratio": float(home_pf["squad_depth_ratio"] - away_pf["squad_depth_ratio"]),
                "delta_age_weighted_value": float(home_pf["age_weighted_value"] - away_pf["age_weighted_value"]),
                "delta_top_league_pct": float(home_pf["top_league_pct"] - away_pf["top_league_pct"]),
                "outcome": int(match["outcome"]),
            }
        )

    train_df = pd.DataFrame(rows)
    folds = _get_loto_folds(train_df)
    teams_2026 = momios["country_code"].tolist()
    groups_2026 = {
        group_letter: momios[momios["group_letter"] == group_letter]["country_code"].tolist()
        for group_letter in momios["group_letter"].unique()
    }

    return PreparedCurrentExperiment(
        master64_raw=master,
        momios=momios,
        worldcups=worldcups,
        all_matches=all_matches,
        train_df=train_df,
        feature_cols=feature_cols,
        folds=folds,
        teams_2026=teams_2026,
        groups_2026=groups_2026,
        elo_final=elo_final,
        team_histories=team_histories,
        player_features=player_features,
        player_years=player_years,
        temporal_master_features=temporal_master_features,
        temporal_master_years=temporal_master_years,
        temporal_master_defaults=temporal_master_defaults,
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
    experiment: PreparedCurrentExperiment,
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
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }

        losses = []
        for train_mask, test_mask, _ in inner_folds:
            model = xgb.XGBClassifier(**params)
            model.fit(X[train_mask], y[train_mask], eval_set=[(X[test_mask], y[test_mask])], verbose=False)
            y_proba = model.predict_proba(X[test_mask])
            from sklearn.metrics import log_loss  # local import to keep module load lean

            losses.append(log_loss(y[test_mask], y_proba))
        return float(np.mean(losses))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {**_base_xgb_params(seed), **study.best_params}, study


def fit_final_model(
    experiment: PreparedCurrentExperiment,
    params: dict[str, Any],
) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(**params)
    X = experiment.train_df[experiment.feature_cols].to_numpy()
    y = experiment.train_df["outcome"].to_numpy()
    model.fit(X, y)
    return model


def save_trained_model(
    model: xgb.XGBClassifier,
    experiment: PreparedCurrentExperiment,
    params: dict[str, Any],
    model_path: Path | str = DEFAULT_MODEL_PATH,
    metadata_path: Path | str = DEFAULT_METADATA_PATH,
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    resolved_model_path = Path(model_path)
    resolved_metadata_path = Path(metadata_path)
    resolved_model_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    model.save_model(resolved_model_path)
    metadata = {
        "model_path": str(resolved_model_path),
        "feature_cols": experiment.feature_cols,
        "n_train_rows": int(len(experiment.train_df)),
        "n_folds": int(len(experiment.folds)),
        "teams_2026": int(len(experiment.teams_2026)),
        "params": _json_safe(params),
    }
    if extra_metadata:
        metadata.update(_json_safe(extra_metadata))
    resolved_metadata_path.write_text(json.dumps(metadata, indent=2))
    return resolved_model_path, resolved_metadata_path


def load_trained_model(model_path: Path | str = DEFAULT_MODEL_PATH) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.load_model(Path(model_path))
    return model


def prepare_2026_team_features(
    experiment: PreparedCurrentExperiment,
) -> dict[str, dict[str, float]]:
    hist_wc = pd.read_csv(_repo_path("Data", "WorldCupHistory", "historial_mundialista.csv"))
    qual_agg = pd.read_csv(_repo_path("Data", "Eliminatorias", "eliminatorias_conjuntas.csv"))
    hist_wc = _normalize_country_column(hist_wc, "country_code")
    qual_agg = _normalize_country_column(qual_agg, "country_code")
    hist_wc_dict = dict(zip(hist_wc["country_code"], hist_wc["pj"]))
    qual_lookup = qual_agg.set_index("country_code")

    team_features_2026: dict[str, dict[str, float]] = {}
    for team_code in experiment.teams_2026:
        master_row = experiment.master64_raw[experiment.master64_raw["country_code"] == team_code]
        if master_row.empty:
            continue
        master = master_row.iloc[0]

        qual_form_wp = float(np.clip(float(master["qual_points_per_game"]) / 3.0, 0.0, 1.0))
        if team_code in qual_lookup.index and float(qual_lookup.loc[team_code, "pj"]) > 0:
            qual_form_wp = float(qual_lookup.loc[team_code, "win_pct"])
        qual_form_gf = float(master["qual_gf_per_game"])
        qual_form_gc = float(master["qual_gc_per_game"])

        recent = experiment.team_histories.get(team_code, [])[-10:]
        if len(recent) >= 3:
            major_wp = sum(1 for _, outcome, _, _ in recent if outcome == "W") / len(recent)
            major_gf = float(np.mean([gf for _, _, gf, _ in recent]))
            major_gc = float(np.mean([gc for _, _, _, gc in recent]))
        else:
            major_wp, major_gf, major_gc = qual_form_wp, qual_form_gf, qual_form_gc

        player_features = get_player_features(
            experiment.player_features,
            experiment.player_years,
            team_code,
            2025,
        )
        temporal_master = get_temporal_master_features(
            experiment.temporal_master_features,
            experiment.temporal_master_years,
            experiment.temporal_master_defaults,
            team_code,
            2026,
        )
        team_features_2026[team_code] = {
            "elo": float(experiment.elo_final.get(team_code, ELO_INIT)),
            "form_wp": float(0.70 * qual_form_wp + 0.30 * major_wp),
            "form_gf": float(0.70 * qual_form_gf + 0.30 * major_gf),
            "form_gc": float(0.70 * qual_form_gc + 0.30 * major_gc),
            "wc_win_pct": float(master["wc_win_pct"]),
            "wc_gc_per_game": float(master["wc_gc_per_game"]),
            "wc_titles": float(master["wc_titles"]),
            "wc_experience": float(hist_wc_dict.get(team_code, 0 if int(master["wc_debut_flag"]) == 1 else 10)),
            "rank_strength": float(temporal_master["rank_strength"]),
            "rank_volatility": float(temporal_master["rank_volatility"]),
            "squad_market_value_log": float(temporal_master["squad_market_value_log"]),
            "squad_avg_age": float(temporal_master["squad_avg_age"]),
            "rank_available": float(temporal_master["rank_available"]),
            "squad_available": float(temporal_master["squad_available"]),
            "star_concentration": float(player_features["star_concentration"]),
            "top5_avg_value": float(player_features["top5_avg_value"]),
            "gk_value": float(player_features["gk_value"]),
            "squad_depth_ratio": float(player_features["squad_depth_ratio"]),
            "age_weighted_value": float(player_features["age_weighted_value"]),
            "top_league_pct": float(player_features["top_league_pct"]),
        }

    return team_features_2026


def precompute_neutral_probabilities(
    experiment: PreparedCurrentExperiment,
    model: xgb.XGBClassifier,
    team_features_2026: dict[str, dict[str, float]],
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], np.ndarray]]:
    def make_match_features(team_a: str, team_b: str, is_knockout: int = 0) -> np.ndarray:
        features_a = team_features_2026[team_a]
        features_b = team_features_2026[team_b]
        return np.array(
            [
                features_a["elo"] - features_b["elo"],
                features_a["form_wp"] - features_b["form_wp"],
                features_a["form_gf"] - features_b["form_gf"],
                features_a["form_gc"] - features_b["form_gc"],
                features_a["wc_win_pct"] - features_b["wc_win_pct"],
                features_a["wc_gc_per_game"] - features_b["wc_gc_per_game"],
                features_a["wc_titles"] - features_b["wc_titles"],
                features_a["wc_experience"] - features_b["wc_experience"],
                float(is_knockout),
                1.0,
                features_a["rank_strength"] - features_b["rank_strength"],
                features_a["rank_volatility"] - features_b["rank_volatility"],
                features_a["squad_market_value_log"] - features_b["squad_market_value_log"],
                features_a["squad_avg_age"] - features_b["squad_avg_age"],
                float(features_a["rank_available"] and features_b["rank_available"]),
                float(features_a["squad_available"] and features_b["squad_available"]),
                features_a["star_concentration"] - features_b["star_concentration"],
                features_a["top5_avg_value"] - features_b["top5_avg_value"],
                features_a["gk_value"] - features_b["gk_value"],
                features_a["squad_depth_ratio"] - features_b["squad_depth_ratio"],
                features_a["age_weighted_value"] - features_b["age_weighted_value"],
                features_a["top_league_pct"] - features_b["top_league_pct"],
            ],
            dtype=float,
        ).reshape(1, -1)

    pairs = [(a, b) for a in experiment.teams_2026 for b in experiment.teams_2026 if a != b]
    X_group = np.vstack([make_match_features(a, b, 0) for a, b in pairs])
    X_knockout = np.vstack([make_match_features(a, b, 1) for a, b in pairs])
    group_raw = model.predict_proba(X_group)
    knockout_raw = model.predict_proba(X_knockout)

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

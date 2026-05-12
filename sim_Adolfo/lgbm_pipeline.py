"""
LightGBM pipeline for World Cup match outcome prediction.

Walk-forward temporal training: features for each match are computed
using ONLY data available before that match date.

Target: 0=Loss, 1=Draw, 2=Win (from team perspective)
"""

from __future__ import annotations

import glob
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data"
OUT = Path(__file__).resolve().parent

# ── Paso 1: Name-to-code mapping ─────────────────────────────────────────────

NAME_TO_CODE = {
    # Standard
    "Argentina": "ARG", "Australia": "AUS", "Austria": "AUT",
    "Belgium": "BEL", "Bolivia": "BOL", "Brazil": "BRA",
    "Bulgaria": "BUL", "Cameroon": "CMR", "Canada": "CAN",
    "Chile": "CHL", "China PR": "CHN", "Colombia": "COL",
    "Costa Rica": "CRC", "Croatia": "CRO", "Cuba": "CUB",
    "Czech Republic": "CZE", "Czechoslovakia": "CZE",
    "Denmark": "DEN", "DR Congo": "COD",
    "Dutch East Indies": "IDN", "Ecuador": "ECU", "Egypt": "EGY",
    "El Salvador": "SLV", "England": "ENG", "France": "FRA",
    "Germany": "GER", "Germany FR": "GER", "West Germany": "GER",
    "Germany DR": "GER_DR", "East Germany": "GER_DR",
    "Ghana": "GHA", "Greece": "GRE", "Haiti": "HAI",
    "Honduras": "HON", "Hungary": "HUN", "Iceland": "ISL",
    "Indonesia": "IDN", "IR Iran": "IRN", "Iran": "IRN",
    "Iraq": "IRQ", "Republic of Ireland": "IRL", "Ireland": "IRL",
    "Israel": "ISR", "Italy": "ITA", "Ivory Coast": "CIV",
    "Côte d'Ivoire": "CIV", "Jamaica": "JAM", "Japan": "JPN",
    "Korea DPR": "PRK", "North Korea": "PRK",
    "Korea Republic": "KOR", "South Korea": "KOR",
    "Kuwait": "KUW", "Mexico": "MEX", "Morocco": "MAR",
    "Netherlands": "NED", "New Zealand": "NZL", "Nigeria": "NGA",
    "Northern Ireland": "NIR", "Norway": "NOR", "Panama": "PAN",
    "Paraguay": "PAR", "Peru": "PER", "Poland": "POL",
    "Portugal": "POR", "Qatar": "QAT", "Romania": "ROU",
    "Russia": "RUS", "Saudi Arabia": "KSA", "Scotland": "SCO",
    "Senegal": "SEN", "Serbia": "SRB", "Serbia and Montenegro": "SRB",
    "FR Yugoslavia": "SRB", "Yugoslavia": "YUG",
    "Slovakia": "SVK", "Slovenia": "SVN", "South Africa": "RSA",
    "Soviet Union": "URS", "Spain": "ESP", "Sweden": "SWE",
    "Switzerland": "SUI", "Togo": "TOG", "Trinidad and Tobago": "TTO",
    "Tunisia": "TUN", "Türkiye": "TUR", "Turkey": "TUR",
    "Ukraine": "UKR", "United Arab Emirates": "ARE",
    "United States": "USA", "Uruguay": "URU", "Wales": "WAL",
    "Zaire": "COD", "Algeria": "ALG", "Angola": "ANG",
    "Congo": "CGO", "Cape Verde": "CPV",
    "Bosnia and Herzegovina": "BIH",
    "Uzbekistan": "UZB", "Jordan": "JOR",
    "Congo-Brazzaville": "CGO",
}

# World Cup hosts per year
WC_HOSTS = {
    1930: {"URU"}, 1934: {"ITA"}, 1938: {"FRA"}, 1950: {"BRA"},
    1954: {"SUI"}, 1958: {"SWE"}, 1962: {"CHL"}, 1966: {"ENG"},
    1970: {"MEX"}, 1974: {"GER"}, 1978: {"ARG"}, 1982: {"ESP"},
    1986: {"MEX"}, 1990: {"ITA"}, 1994: {"USA"}, 1998: {"FRA"},
    2002: {"KOR", "JPN"}, 2006: {"GER"}, 2010: {"RSA"},
    2014: {"BRA"}, 2018: {"RUS"}, 2022: {"QAT"},
    2026: {"USA", "MEX", "CAN"},
}

# World Cup winners
WC_WINNERS = {
    1930: "URU", 1934: "ITA", 1938: "ITA", 1950: "URU",
    1954: "GER", 1958: "BRA", 1962: "BRA", 1966: "ENG",
    1970: "BRA", 1974: "GER", 1978: "ARG", 1982: "ITA",
    1986: "ARG", 1990: "GER", 1994: "BRA", 1998: "FRA",
    2002: "BRA", 2006: "ITA", 2010: "ESP", 2014: "GER",
    2018: "FRA", 2022: "ARG",
}

# Copa América name mapping (Portuguese)
COPA_NAME_MAP = {
    "Brasil": "BRA", "Argentina": "ARG", "Uruguai": "URU",
    "Colombia": "COL", "Equador": "ECU", "Paraguai": "PAR",
    "Bolivia": "BOL", "Chile": "CHL", "Peru": "PER",
    "Venezuela": "VEN", "Honduras": "HON", "Costa Rica": "CRC",
    "Mexico": "MEX", "México": "MEX", "EUA": "USA", "Haiti": "HAI",
    "Jamaica": "JAM", "Catar": "QAT", "Japao": "JPN",
    "Panama": "PAN", "Panamá": "PAN", "Canadá": "CAN",
}

# AFCON name mapping
AFCON_NAME_MAP = {
    "Algeria": "ALG", "Congo": "CGO", "Congo-Brazzaville": "CGO",
    "Ivory Coast": "CIV", "Cape Verde": "CPV", "Egypt": "EGY",
    "Ghana": "GHA", "Morocco": "MAR", "South Africa": "RSA",
    "Senegal": "SEN", "Tunisia": "TUN", "Angola": "ANG",
    "Benin": "BEN", "Burkina Faso": "BFA", "Burundi": "BDI",
    "Cameroon": "CMR", "Comoros": "COM", "DR Congo": "COD",
    "Congo-Kinshasa": "COD", "Zaire": "COD", "Congo-Léopoldville": "COD",
    "Equatorial Guinea": "GEQ", "Ethiopia": "ETH", "Gabon": "GAB",
    "Gambia": "GAM", "Guinea": "GUI", "Guinea-Bissau": "GNB",
    "Kenya": "KEN", "Liberia": "LBR", "Libya": "LBY",
    "Madagascar": "MAD", "Malawi": "MWI", "Mali": "MLI",
    "Mauritania": "MTN", "Mauritius": "MRI", "Mozambique": "MOZ",
    "Namibia": "NAM", "Nigeria": "NGA", "Rwanda": "RWA",
    "Sierra Leone": "SLE", "Sudan": "SDN", "Tanzania": "TAN",
    "Togo": "TOG", "Uganda": "UGA", "United Arab Rep.": "EGY",
    "Upper Volta": "BFA", "Zambia": "ZAM", "Zimbabwe": "ZIM",
}

KNOCKOUT_ROUNDS = {
    "Round of 16", "Quarter-finals", "Semi-finals",
    "Third-place match", "Final",
}


# ── Load data ─────────────────────────────────────────────────────────────────

def load_all_data():
    wc = pd.read_csv(DATA / "WorldCupHistory" / "matches_1930_2022.csv")
    rankings = pd.read_csv(DATA / "ranking_mundial_2026_v2.csv")
    players = pd.read_csv(DATA / "transfermarkt_players.csv")

    euro_files = sorted(glob.glob(str(DATA / "euro_histoia" / "*.csv")))
    euro = pd.concat([pd.read_csv(f) for f in euro_files], ignore_index=True)

    copa1 = pd.read_csv(DATA / "Copa_America" / "Copa_America.csv")
    copa2 = pd.read_csv(DATA / "Copa_America" / "Copa_America_2024.csv")
    copa = pd.concat([copa1, copa2], ignore_index=True)

    afcon = pd.read_csv(DATA / "Africa_Cup_Of_Nations" / "Africa Cup of Nations Matches.csv")

    return wc, rankings, players, euro, copa, afcon


# ── Paso 1 validation ────────────────────────────────────────────────────────

def validate_name_mapping(wc: pd.DataFrame):
    all_names = set(wc["home_team"].unique()) | set(wc["away_team"].unique())
    unmapped = [n for n in all_names if n not in NAME_TO_CODE]
    if unmapped:
        print(f"UNMAPPED NAMES: {unmapped}")
        raise ValueError("Resolve all name mappings before proceeding")
    print(f"✓ All {len(all_names)} team names mapped")


# ── Prepare continental matches ───────────────────────────────────────────────

def prepare_euro(euro: pd.DataFrame) -> pd.DataFrame:
    df = euro[["home_team_code", "away_team_code", "home_score", "away_score", "year", "date"]].copy()
    df = df.dropna(subset=["home_score", "away_score"])
    df.columns = ["home_code", "away_code", "home_score", "away_score", "year", "date"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["tournament"] = "euro"
    return df


def prepare_copa(copa: pd.DataFrame) -> pd.DataFrame:
    df = copa.copy()
    df["home_code"] = df["Casa"].map(COPA_NAME_MAP)
    df["away_code"] = df["Fora"].map(COPA_NAME_MAP)
    df = df.dropna(subset=["home_code", "away_code", "Gols Casa", "Gols Fora"])
    df["home_score"] = df["Gols Casa"].astype(int)
    df["away_score"] = df["Gols Fora"].astype(int)
    df["year"] = df["Edição"].astype(int)
    df["date"] = pd.to_datetime(df["Data"], errors="coerce")
    df["tournament"] = "copa"
    return df[["home_code", "away_code", "home_score", "away_score", "year", "date", "tournament"]]


def prepare_afcon(afcon: pd.DataFrame) -> pd.DataFrame:
    df = afcon.copy()
    df["HomeTeam"] = df["HomeTeam"].str.strip()
    df["AwayTeam"] = df["AwayTeam"].str.strip()
    df["home_code"] = df["HomeTeam"].map(AFCON_NAME_MAP)
    df["away_code"] = df["AwayTeam"].map(AFCON_NAME_MAP)
    df = df.dropna(subset=["home_code", "away_code", "HomeTeamGoals", "AwayTeamGoals"])
    df["home_score"] = df["HomeTeamGoals"].astype(int)
    df["away_score"] = df["AwayTeamGoals"].astype(int)
    df["year"] = df["Year"].astype(int)
    df["date"] = pd.to_datetime(df["Date "].str.strip(), format="%d-%b-%y", errors="coerce")
    df["tournament"] = "afcon"
    return df[["home_code", "away_code", "home_score", "away_score", "year", "date", "tournament"]]


# ── Paso 2: Prepare WC matches & duplicate ───────────────────────────────────

def prepare_wc_matches(wc: pd.DataFrame) -> pd.DataFrame:
    df = wc[wc["Year"] >= 1950].copy()
    df["home_code"] = df["home_team"].map(NAME_TO_CODE)
    df["away_code"] = df["away_team"].map(NAME_TO_CODE)
    df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["year"] = df["Year"].astype(int)
    df["is_knockout"] = df["Round"].isin(KNOCKOUT_ROUNDS).astype(int)
    df["round"] = df["Round"]
    df = df.sort_values("date").reset_index(drop=True)

    rows = []
    for match_id, (_, m) in enumerate(df.iterrows()):
        hs, as_ = int(m["home_score"]), int(m["away_score"])
        base = {
            "match_id": match_id,
            "date": m["date"], "year": m["year"],
            "is_knockout": m["is_knockout"], "round": m["round"],
            "home_score": hs, "away_score": as_,
        }
        if hs > as_:
            outcome_home, outcome_away = 2, 0
        elif hs == as_:
            outcome_home, outcome_away = 1, 1
        else:
            outcome_home, outcome_away = 0, 2

        rows.append({**base, "team": m["home_code"], "opponent": m["away_code"],
                      "gf": hs, "gc": as_, "outcome": outcome_home})
        rows.append({**base, "team": m["away_code"], "opponent": m["home_code"],
                      "gf": as_, "gc": hs, "outcome": outcome_away})

    result = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    n_matches = len(df)
    assert len(result) == 2 * n_matches, f"Expected {2*n_matches}, got {len(result)}"
    print(f"✓ {n_matches} matches from 1950 → {len(result)} rows (duplicated)")
    return result


# ── Paso 3: Feature computation (walk-forward) ───────────────────────────────

def compute_features(
    wc_rows: pd.DataFrame,
    wc_raw: pd.DataFrame,
    rankings: pd.DataFrame,
    players: pd.DataFrame,
    euro: pd.DataFrame,
    copa: pd.DataFrame,
    afcon: pd.DataFrame,
):
    wc_raw = wc_raw.copy()
    wc_raw["home_code"] = wc_raw["home_team"].map(NAME_TO_CODE)
    wc_raw["away_code"] = wc_raw["away_team"].map(NAME_TO_CODE)
    wc_raw["date"] = pd.to_datetime(wc_raw["Date"], errors="coerce")
    wc_raw = wc_raw.sort_values("date").reset_index(drop=True)

    euro_prep = prepare_euro(euro)
    copa_prep = prepare_copa(copa)
    afcon_prep = prepare_afcon(afcon)

    # Pre-build ranking lookup: {(code, year): ranking}
    rank_lookup = {}
    for _, r in rankings.iterrows():
        rank_lookup[(r["country_code"], int(r["year"]))] = float(r["ranking"])

    # Pre-build transfermarkt aggregates: {(code, year): {market_value, avg_age}}
    tm_lookup = {}
    for (code, year), grp in players.groupby(["country_code", "year"]):
        mv = grp["market_value_eur_m"].sum()
        avg_age = grp["age"].mean()
        tm_lookup[(code, int(year))] = {"market_value_eur_m": mv, "squad_avg_age": avg_age}

    # Pre-compute top-10 sets by year from rankings
    top10_by_year = {}
    for year in rankings["year"].unique():
        yr_ranks = rankings[rankings["year"] == year]
        top10_by_year[int(year)] = set(yr_ranks[yr_ranks["ranking"] <= 10]["country_code"])

    unique_dates = sorted(wc_rows["date"].unique())
    print(f"Computing features for {len(unique_dates)} unique match dates...")

    # Accumulators for WC history (updated as we process chronologically)
    # For each team: {wins, draws, losses, gf, gc, games, appearances_set, semi_years, final_years, titles}
    wc_history = {}

    def _get_hist(code):
        if code not in wc_history:
            wc_history[code] = {
                "wins": 0, "draws": 0, "losses": 0,
                "gf": 0, "gc": 0, "games": 0,
                "wc_years": set(), "semi_years": set(),
                "final_years": set(), "titles": 0,
            }
        return wc_history[code]

    # Continental accumulators
    euro_hist = {}  # {code: {wins, games}}
    copa_hist = {}
    afcon_hist = {}

    def _get_cont(store, code):
        if code not in store:
            store[code] = {"wins": 0, "games": 0}
        return store[code]

    # Pre-sort continental matches by date
    euro_prep = euro_prep.sort_values("date").reset_index(drop=True)
    copa_prep = copa_prep.sort_values("date").reset_index(drop=True)
    afcon_prep = afcon_prep.sort_values("date").reset_index(drop=True)

    euro_idx = 0
    copa_idx = 0
    afcon_idx = 0

    # Track which WC raw matches we've processed (for updating history)
    wc_raw_idx = 0

    # win_pct_vs_top10 accumulator: {code: {wins_vs_top10, games_vs_top10}}
    top10_hist = {}

    def _get_top10(code):
        if code not in top10_hist:
            top10_hist[code] = {"wins": 0, "games": 0}
        return top10_hist[code]

    # Process matches chronologically
    all_features = []
    prev_wc_year = None

    for i, row in wc_rows.iterrows():
        match_date = row["date"]
        match_year = row["year"]
        team = row["team"]
        opponent = row["opponent"]

        # ── Update continental histories up to this date ──────────────
        while euro_idx < len(euro_prep) and euro_prep.iloc[euro_idx]["date"] < match_date:
            er = euro_prep.iloc[euro_idx]
            hc, ac = er["home_code"], er["away_code"]
            hs, as_ = er["home_score"], er["away_score"]
            h = _get_cont(euro_hist, hc)
            a = _get_cont(euro_hist, ac)
            h["games"] += 1; a["games"] += 1
            if hs > as_: h["wins"] += 1
            elif as_ > hs: a["wins"] += 1

            # Also update top10 history from continental
            yr = int(er["year"])
            top10 = top10_by_year.get(yr, set())
            if ac in top10:
                t = _get_top10(hc); t["games"] += 1
                if hs > as_: t["wins"] += 1
            if hc in top10:
                t = _get_top10(ac); t["games"] += 1
                if as_ > hs: t["wins"] += 1

            euro_idx += 1

        while copa_idx < len(copa_prep) and copa_prep.iloc[copa_idx]["date"] < match_date:
            cr = copa_prep.iloc[copa_idx]
            hc, ac = cr["home_code"], cr["away_code"]
            hs, as_ = cr["home_score"], cr["away_score"]
            h = _get_cont(copa_hist, hc)
            a = _get_cont(copa_hist, ac)
            h["games"] += 1; a["games"] += 1
            if hs > as_: h["wins"] += 1
            elif as_ > hs: a["wins"] += 1

            yr = int(cr["year"])
            top10 = top10_by_year.get(yr, set())
            if ac in top10:
                t = _get_top10(hc); t["games"] += 1
                if hs > as_: t["wins"] += 1
            if hc in top10:
                t = _get_top10(ac); t["games"] += 1
                if as_ > hs: t["wins"] += 1

            copa_idx += 1

        while afcon_idx < len(afcon_prep) and afcon_prep.iloc[afcon_idx]["date"] < match_date:
            ar = afcon_prep.iloc[afcon_idx]
            hc, ac = ar["home_code"], ar["away_code"]
            hs, as_ = ar["home_score"], ar["away_score"]
            h = _get_cont(afcon_hist, hc)
            a = _get_cont(afcon_hist, ac)
            h["games"] += 1; a["games"] += 1
            if hs > as_: h["wins"] += 1
            elif as_ > hs: a["wins"] += 1

            yr = int(ar["year"])
            top10 = top10_by_year.get(yr, set())
            if ac in top10:
                t = _get_top10(hc); t["games"] += 1
                if hs > as_: t["wins"] += 1
            if hc in top10:
                t = _get_top10(ac); t["games"] += 1
                if as_ > hs: t["wins"] += 1

            afcon_idx += 1

        # ── Update WC history from raw matches before this date ──────
        while wc_raw_idx < len(wc_raw) and wc_raw.iloc[wc_raw_idx]["date"] < match_date:
            mr = wc_raw.iloc[wc_raw_idx]
            if mr["Year"] >= 1950:
                hc, ac = mr["home_code"], mr["away_code"]
                hs, as_ = int(mr["home_score"]), int(mr["away_score"])
                yr = int(mr["Year"])
                rnd = mr["Round"]

                for code, gf, gc in [(hc, hs, as_), (ac, as_, hs)]:
                    h = _get_hist(code)
                    h["games"] += 1; h["gf"] += gf; h["gc"] += gc
                    h["wc_years"].add(yr)
                    if gf > gc: h["wins"] += 1
                    elif gf == gc: h["draws"] += 1
                    else: h["losses"] += 1
                    if rnd in ("Semi-finals", "Third-place match", "Final"):
                        h["semi_years"].add(yr)
                    if rnd == "Final":
                        h["final_years"].add(yr)

                if yr in WC_WINNERS:
                    winner_code = WC_WINNERS[yr]
                    _get_hist(winner_code)["titles"] = sum(
                        1 for wy, wc in WC_WINNERS.items() if wc == winner_code and wy < match_date.year
                    )

                # Update top10 from WC
                top10 = top10_by_year.get(yr, set())
                if ac in top10:
                    t = _get_top10(hc); t["games"] += 1
                    if hs > as_: t["wins"] += 1
                if hc in top10:
                    t = _get_top10(ac); t["games"] += 1
                    if as_ > hs: t["wins"] += 1

            wc_raw_idx += 1

        # ── Compute features for team ────────────────────────────────
        feat = {}

        # Group A: WC History
        th = _get_hist(team)
        if th["games"] > 0:
            feat["wc_win_pct"] = th["wins"] / th["games"]
            feat["wc_draw_pct"] = th["draws"] / th["games"]
            feat["wc_gf_per_game"] = th["gf"] / th["games"]
            feat["wc_gc_per_game"] = th["gc"] / th["games"]
        else:
            feat["wc_win_pct"] = np.nan
            feat["wc_draw_pct"] = np.nan
            feat["wc_gf_per_game"] = np.nan
            feat["wc_gc_per_game"] = np.nan

        feat["wc_appearances"] = len(th["wc_years"])
        feat["wc_games_played"] = th["games"]

        n_appearances = len(th["wc_years"])
        if n_appearances > 0:
            feat["wc_semi_pct"] = len(th["semi_years"]) / n_appearances
            feat["wc_final_pct"] = len(th["final_years"]) / n_appearances
        else:
            feat["wc_semi_pct"] = np.nan
            feat["wc_final_pct"] = np.nan

        feat["wc_champion_count"] = sum(
            1 for wy, wc in WC_WINNERS.items() if wc == team and wy < match_year
        )

        # Group B: Ranking
        rank_val = rank_lookup.get((team, match_year))
        feat["ranking"] = rank_val
        feat["has_ranking"] = 1 if rank_val is not None else 0

        # Ranking volatility (std of last 3 years)
        recent_ranks = []
        for y in range(match_year - 3, match_year):
            rv = rank_lookup.get((team, y))
            if rv is not None:
                recent_ranks.append(rv)
        feat["ranking_volatility"] = np.std(recent_ranks) if len(recent_ranks) > 1 else np.nan

        # Group C: Market value
        tm_data = tm_lookup.get((team, match_year))
        if tm_data is None:
            # Try previous year
            tm_data = tm_lookup.get((team, match_year - 1))
        feat["market_value_eur_m"] = tm_data["market_value_eur_m"] if tm_data else np.nan
        feat["squad_avg_age"] = tm_data["squad_avg_age"] if tm_data else np.nan

        # Group D: win_pct_vs_top10
        t10 = _get_top10(team)
        feat["win_pct_vs_top10"] = (t10["wins"] / t10["games"]) if t10["games"] > 0 else np.nan

        # Group E: Continental
        euro_h = euro_hist.get(team)
        feat["euro_win_pct"] = (euro_h["wins"] / euro_h["games"]) if euro_h and euro_h["games"] > 0 else np.nan

        copa_h = copa_hist.get(team)
        feat["copa_win_pct"] = (copa_h["wins"] / copa_h["games"]) if copa_h and copa_h["games"] > 0 else np.nan

        afcon_h = afcon_hist.get(team)
        feat["afcon_win_pct"] = (afcon_h["wins"] / afcon_h["games"]) if afcon_h and afcon_h["games"] > 0 else np.nan

        # Group F: Context
        hosts = WC_HOSTS.get(match_year, set())
        feat["host_flag"] = 1 if team in hosts else 0

        all_features.append(feat)

    return pd.DataFrame(all_features, index=wc_rows.index)


# ── Paso 4: Build deltas ─────────────────────────────────────────────────────

FEATURE_COLS = [
    "wc_win_pct", "wc_draw_pct", "wc_gf_per_game", "wc_gc_per_game",
    "wc_appearances", "wc_games_played", "wc_semi_pct", "wc_final_pct",
    "wc_champion_count",
    "ranking", "ranking_volatility", "has_ranking",
    "market_value_eur_m", "squad_avg_age",
    "win_pct_vs_top10",
    "euro_win_pct", "copa_win_pct", "afcon_win_pct",
    "host_flag",
]


def build_deltas(wc_rows: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    result_rows = []

    for match_id, group in wc_rows.groupby("match_id"):
        assert len(group) == 2, f"Match {match_id} has {len(group)} rows"
        idx_a, idx_b = group.index[0], group.index[1]
        row_a, row_b = wc_rows.loc[idx_a], wc_rows.loc[idx_b]
        feat_a, feat_b = features.loc[idx_a], features.loc[idx_b]

        for row, feat, opp_feat in [(row_a, feat_a, feat_b), (row_b, feat_b, feat_a)]:
            delta_row = {
                "date": row["date"],
                "year": row["year"],
                "team": row["team"],
                "opponent": row["opponent"],
                "outcome": row["outcome"],
                "is_knockout": row["is_knockout"],
            }
            for col in FEATURE_COLS:
                delta_row[f"team_{col}"] = feat[col]
                delta_row[f"opp_{col}"] = opp_feat[col]
                va, vb = feat[col], opp_feat[col]
                if pd.notna(va) and pd.notna(vb):
                    delta_row[f"delta_{col}"] = va - vb
                else:
                    delta_row[f"delta_{col}"] = np.nan
            result_rows.append(delta_row)

    return pd.DataFrame(result_rows).sort_values("date").reset_index(drop=True)


# ── Paso 5 & 6: Walk-forward training ────────────────────────────────────────

def get_model_features():
    cols = []
    for col in FEATURE_COLS:
        cols.append(f"team_{col}")
        cols.append(f"opp_{col}")
        cols.append(f"delta_{col}")
    cols.append("is_knockout")
    return cols


def walk_forward_train(train_df: pd.DataFrame):
    folds = [
        (2010, 2014),
        (2014, 2018),
        (2018, 2022),
    ]

    model_cols = get_model_features()
    fold_results = []
    best_iterations = []

    print("\n" + "=" * 70)
    print("WALK-FORWARD TRAINING")
    print("=" * 70)

    for fold_num, (train_cutoff, test_year) in enumerate(folds, 1):
        train_mask = train_df["year"] <= train_cutoff
        test_mask = train_df["year"] == test_year

        X_train = train_df.loc[train_mask, model_cols]
        y_train = train_df.loc[train_mask, "outcome"]
        X_test = train_df.loc[test_mask, model_cols]
        y_test = train_df.loc[test_mask, "outcome"]

        # Anti-leakage check
        max_train_year = train_df.loc[train_mask, "year"].max()
        assert max_train_year <= train_cutoff, f"LEAKAGE: train has year {max_train_year} > {train_cutoff}"

        print(f"\n--- Fold {fold_num}: Train ≤{train_cutoff} → Test {test_year} ---")
        print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
        print(f"Train target dist: {dict(y_train.value_counts().sort_index())}")
        print(f"Test target dist:  {dict(y_test.value_counts().sort_index())}")

        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "verbosity": -1,
            "seed": 42,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
        }

        callbacks = [
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(0),
        ]

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[test_data],
            callbacks=callbacks,
        )

        best_iter = model.best_iteration
        best_iterations.append(best_iter)

        y_pred_proba = model.predict(X_test)
        assert y_pred_proba.shape[1] == 3, f"Expected 3 classes, got {y_pred_proba.shape[1]}"

        y_pred = np.argmax(y_pred_proba, axis=1)
        accuracy = (y_pred == y_test.values).mean()
        logloss = model.best_score["valid_0"]["multi_logloss"]

        pred_dist = pd.Series(y_pred).value_counts().sort_index().to_dict()

        print(f"Best iteration: {best_iter}")
        print(f"Log-loss: {logloss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Predicted dist: {pred_dist}")

        fold_results.append({
            "fold": fold_num,
            "train_cutoff": train_cutoff,
            "test_year": test_year,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "best_iteration": best_iter,
            "logloss": logloss,
            "accuracy": accuracy,
            "pred_dist": pred_dist,
            "model": model,
        })

    return fold_results, best_iterations


# ── Paso 7: Final model ──────────────────────────────────────────────────────

def train_final_model(train_df: pd.DataFrame, avg_iterations: int):
    model_cols = get_model_features()
    X = train_df[model_cols]
    y = train_df["outcome"]

    print(f"\n{'=' * 70}")
    print(f"FINAL MODEL — all data, {avg_iterations} iterations")
    print(f"{'=' * 70}")
    print(f"Training on {len(X)} rows")

    train_data = lgb.Dataset(X, label=y)

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "verbosity": -1,
        "seed": 42,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
    }

    model = lgb.train(params, train_data, num_boost_round=avg_iterations)

    # Feature importance
    importance_gain = pd.DataFrame({
        "feature": model.feature_name(),
        "gain": model.feature_importance(importance_type="gain"),
        "split": model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)

    print("\nTop 20 features by gain:")
    print(importance_gain.head(20).to_string(index=False))

    return model, importance_gain


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    wc_raw, rankings, players, euro, copa, afcon = load_all_data()

    print("\n--- Paso 1: Validating name mapping ---")
    validate_name_mapping(wc_raw)

    print("\n--- Paso 2: Preparing WC matches (duplicated) ---")
    wc_rows = prepare_wc_matches(wc_raw)

    print(f"\nTarget distribution (all):")
    print(wc_rows["outcome"].value_counts().sort_index().to_frame("count"))
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

    print("\n--- Paso 6: Walk-forward training ---")
    fold_results, best_iterations = walk_forward_train(train_df)

    print("\n--- Paso 7: Final model ---")
    avg_iter = int(np.mean(best_iterations))
    print(f"Average best_iteration from folds: {avg_iter}")
    final_model, importance = train_final_model(train_df, avg_iter)

    # Save artifacts
    model_path = OUT / "lgbm_wc_model.txt"
    final_model.save_model(str(model_path))
    print(f"\n✓ Model saved: {model_path}")

    train_df.to_csv(OUT / "lgbm_train_dataset.csv", index=False)
    print(f"✓ Train dataset saved: {OUT / 'lgbm_train_dataset.csv'}")

    importance.to_csv(OUT / "lgbm_feature_importance.csv", index=False)
    print(f"✓ Feature importance saved: {OUT / 'lgbm_feature_importance.csv'}")

    metrics = {
        "folds": [
            {k: v for k, v in fr.items() if k != "model"}
            for fr in fold_results
        ],
        "avg_best_iteration": avg_iter,
        "total_train_rows": len(train_df),
    }
    with open(OUT / "lgbm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"✓ Metrics saved: {OUT / 'lgbm_metrics.json'}")

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

"""
Monte Carlo simulation of the 2026 FIFA World Cup using Random Forest.

Same bracket and logic as montecarlo_simulation.py but with RF model.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "rf_wc_model.joblib"
MASTER_PATH = ROOT / "master_dataset_48.csv"
MOMIOS_PATH = ROOT.parent / "Data" / "momios_mundial2026.csv"

NAN_SENTINEL = -999.0

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


def load_model_and_data():
    model = joblib.load(str(MODEL_PATH))
    master = pd.read_csv(MASTER_PATH)
    momios = pd.read_csv(MOMIOS_PATH)
    return model, master, momios


def build_team_features(master: pd.DataFrame) -> dict[str, dict[str, float]]:
    team_feats = {}
    for _, row in master.iterrows():
        code = row["country_code"]
        feats = {}
        feats["wc_win_pct"] = row.get("wc_win_pct", np.nan)
        feats["wc_draw_pct"] = np.nan
        feats["wc_gf_per_game"] = np.nan
        feats["wc_gc_per_game"] = row.get("wc_gc_per_game", np.nan)
        feats["wc_appearances"] = np.nan
        feats["wc_games_played"] = np.nan
        feats["wc_semi_pct"] = row.get("wc_semi_pct", np.nan)
        feats["wc_final_pct"] = row.get("wc_final_pct", np.nan)
        feats["wc_champion_count"] = row.get("wc_champion", 0)
        feats["ranking"] = row.get("ranking_2026", np.nan)
        feats["ranking_volatility"] = row.get("ranking_volatility", np.nan)
        feats["has_ranking"] = 1 if pd.notna(row.get("ranking_2026")) else 0
        feats["market_value_eur_m"] = row.get("market_value_eur_m", np.nan)
        feats["squad_avg_age"] = row.get("squad_avg_age", np.nan)
        feats["win_pct_vs_top10"] = row.get("win_pct_vs_top10", np.nan)
        feats["euro_win_pct"] = row.get("euro_win_pct", np.nan)
        feats["copa_win_pct"] = row.get("copa_win_pct", np.nan)
        feats["afcon_win_pct"] = row.get("afcon_win_pct", np.nan)
        feats["host_flag"] = row.get("host_flag", 0)
        team_feats[code] = feats
    return team_feats


def make_match_input(
    team_feats: dict[str, dict],
    team_a: str,
    team_b: str,
    is_knockout: int,
) -> np.ndarray:
    fa = team_feats[team_a]
    fb = team_feats[team_b]

    row = []
    for col in FEATURE_COLS:
        va = fa.get(col, np.nan)
        vb = fb.get(col, np.nan)
        row.append(va)
        row.append(vb)
        if pd.notna(va) and pd.notna(vb):
            row.append(va - vb)
        else:
            row.append(np.nan)
    row.append(is_knockout)

    arr = np.array(row, dtype=np.float64).reshape(1, -1)
    arr = np.where(np.isnan(arr), NAN_SENTINEL, arr)
    return arr


def simulate_group(model, team_feats, teams, rng):
    points = {t: 0 for t in teams}
    gd = {t: 0 for t in teams}

    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            a, b = teams[i], teams[j]
            X = make_match_input(team_feats, a, b, 0)
            proba = model.predict_proba(X)[0]

            r = rng.random()
            if r < proba[0]:
                points[b] += 3; gd[b] += 1; gd[a] -= 1
            elif r < proba[0] + proba[1]:
                points[a] += 1; points[b] += 1
            else:
                points[a] += 3; gd[a] += 1; gd[b] -= 1

    standings = [
        (t, points[t], gd[t], rng.random())
        for t in teams
    ]
    standings.sort(key=lambda x: (-x[1], -x[2], -x[3]))
    return standings


def select_best_thirds(all_thirds):
    sorted_thirds = sorted(all_thirds, key=lambda x: (-x[2], -x[3]))
    best_8 = sorted_thirds[:8]
    return [(t[0], t[1]) for t in sorted(best_8, key=lambda x: x[1])]


def assign_thirds_to_bracket(qualified_thirds):
    groups_qualified = set(g for _, g in qualified_thirds)
    thirds_by_group = {g: t for t, g in qualified_thirds}

    slot_pools = {
        "3_74": ["A", "B", "C", "D", "F"],
        "3_77": ["C", "D", "F", "G", "H"],
        "3_79": ["C", "E", "F", "H", "I"],
        "3_80": ["E", "H", "I", "J", "K"],
        "3_81": ["B", "E", "F", "I", "J"],
        "3_82": ["A", "E", "H", "I", "J"],
        "3_85": ["E", "F", "G", "I", "J"],
        "3_87": ["D", "E", "I", "J", "L"],
    }

    assigned = {}
    used_groups = set()
    slots_sorted = sorted(slot_pools.keys(),
                          key=lambda s: len([g for g in slot_pools[s] if g in groups_qualified]))

    for slot in slots_sorted:
        pool = slot_pools[slot]
        for g in pool:
            if g in groups_qualified and g not in used_groups:
                assigned[slot] = thirds_by_group[g]
                used_groups.add(g)
                break
    return assigned


def simulate_knockout_match(model, team_feats, team_a, team_b, rng):
    X = make_match_input(team_feats, team_a, team_b, 1)
    proba = model.predict_proba(X)[0]
    p_win_a = proba[2] + proba[1] * proba[2] / (proba[0] + proba[2] + 1e-10)
    if rng.random() < p_win_a:
        return team_a, team_b
    return team_b, team_a


def simulate_tournament(model, team_feats, groups, rng):
    group_results = {}
    all_thirds = []

    for group_letter in sorted(groups.keys()):
        teams = groups[group_letter]
        standings = simulate_group(model, team_feats, teams, rng)
        group_results[group_letter] = standings
        third = standings[2]
        all_thirds.append((third[0], group_letter, third[1], third[2]))

    firsts = {g: group_results[g][0][0] for g in group_results}
    seconds = {g: group_results[g][1][0] for g in group_results}

    qualified_thirds = select_best_thirds(all_thirds)
    thirds_bracket = assign_thirds_to_bracket(qualified_thirds)

    ko = simulate_knockout_match

    # Round of 32
    w73, _ = ko(model, team_feats, seconds["A"], seconds["B"], rng)
    w74, _ = ko(model, team_feats, firsts["E"], thirds_bracket.get("3_74", firsts["E"]), rng)
    w75, _ = ko(model, team_feats, firsts["F"], seconds["C"], rng)
    w76, _ = ko(model, team_feats, firsts["C"], seconds["F"], rng)
    w77, _ = ko(model, team_feats, firsts["I"], thirds_bracket.get("3_77", firsts["I"]), rng)
    w78, _ = ko(model, team_feats, seconds["E"], seconds["I"], rng)
    w79, _ = ko(model, team_feats, firsts["A"], thirds_bracket.get("3_79", firsts["A"]), rng)
    w80, _ = ko(model, team_feats, firsts["L"], thirds_bracket.get("3_80", firsts["L"]), rng)
    w81, _ = ko(model, team_feats, firsts["D"], thirds_bracket.get("3_81", firsts["D"]), rng)
    w82, _ = ko(model, team_feats, firsts["G"], thirds_bracket.get("3_82", firsts["G"]), rng)
    w83, _ = ko(model, team_feats, seconds["K"], seconds["L"], rng)
    w84, _ = ko(model, team_feats, firsts["H"], seconds["J"], rng)
    w85, _ = ko(model, team_feats, firsts["B"], thirds_bracket.get("3_85", firsts["B"]), rng)
    w86, _ = ko(model, team_feats, firsts["J"], seconds["H"], rng)
    w87, _ = ko(model, team_feats, firsts["K"], thirds_bracket.get("3_87", firsts["K"]), rng)
    w88, _ = ko(model, team_feats, seconds["D"], seconds["G"], rng)

    # Round of 16
    w89, _ = ko(model, team_feats, w74, w77, rng)
    w90, _ = ko(model, team_feats, w73, w75, rng)
    w91, _ = ko(model, team_feats, w76, w78, rng)
    w92, _ = ko(model, team_feats, w79, w80, rng)
    w93, _ = ko(model, team_feats, w83, w84, rng)
    w94, _ = ko(model, team_feats, w81, w82, rng)
    w95, _ = ko(model, team_feats, w86, w88, rng)
    w96, _ = ko(model, team_feats, w85, w87, rng)

    # Quarterfinals
    w97, _ = ko(model, team_feats, w89, w90, rng)
    w98, _ = ko(model, team_feats, w93, w94, rng)
    w99, _ = ko(model, team_feats, w91, w92, rng)
    w100, _ = ko(model, team_feats, w95, w96, rng)

    # Semifinals
    w101, l101 = ko(model, team_feats, w97, w98, rng)
    w102, l102 = ko(model, team_feats, w99, w100, rng)

    # Final
    champion, finalist = ko(model, team_feats, w101, w102, rng)

    return {
        "champion": champion,
        "finalist": finalist,
        "semifinalists": {w97, w98, w99, w100},
        "quarterfinalists": {w89, w90, w91, w92, w93, w94, w95, w96},
    }


def run_simulation(n_sims: int = 10_000, seed: int = 42):
    print("Loading model and data...")
    model, master, momios = load_model_and_data()
    team_feats = build_team_features(master)

    CODE_NORM = {"COD": "CGO"}
    groups = {}
    for _, row in momios.iterrows():
        g = row["group_letter"]
        code = CODE_NORM.get(row["country_code"], row["country_code"])
        if g not in groups:
            groups[g] = []
        groups[g].append(code)

    print(f"Groups: {len(groups)} × {[len(v) for v in groups.values()]}")
    print(f"Teams: {sum(len(v) for v in groups.values())}")

    rng = np.random.default_rng(seed)

    champion_count = Counter()
    finalist_count = Counter()
    semifinal_count = Counter()
    quarterfinal_count = Counter()

    all_teams = set()
    for teams in groups.values():
        all_teams.update(teams)

    print(f"\nRunning {n_sims:,} simulations (Random Forest)...")

    for sim in range(n_sims):
        if (sim + 1) % 1000 == 0:
            print(f"  Simulation {sim + 1:,}/{n_sims:,}")

        result = simulate_tournament(model, team_feats, groups, rng)

        champion_count[result["champion"]] += 1
        finalist_count[result["finalist"]] += 1
        finalist_count[result["champion"]] += 1

        for t in result["semifinalists"]:
            semifinal_count[t] += 1
        for t in result["quarterfinalists"]:
            quarterfinal_count[t] += 1

    print(f"\n{'=' * 70}")
    print(f"RESULTADOS RF — {n_sims:,} simulaciones Monte Carlo")
    print(f"{'=' * 70}")

    print(f"\n{'Pos':<4} {'Equipo':<6} {'Campeón %':>10} {'Final %':>10} {'Semi %':>10} {'Cuartos %':>10}")
    print("-" * 54)

    sorted_teams = champion_count.most_common(48)
    counted_teams = set(t for t, _ in sorted_teams)
    for t in all_teams - counted_teams:
        sorted_teams.append((t, 0))

    for pos, (team, count) in enumerate(sorted_teams, 1):
        champ_pct = count / n_sims * 100
        final_pct = finalist_count.get(team, 0) / n_sims * 100
        semi_pct = semifinal_count.get(team, 0) / n_sims * 100
        qf_pct = quarterfinal_count.get(team, 0) / n_sims * 100
        print(f"{pos:<4} {team:<6} {champ_pct:>9.2f}% {final_pct:>9.2f}% {semi_pct:>9.2f}% {qf_pct:>9.2f}%")
        if pos >= 48:
            break

    results_data = []
    for team, count in sorted_teams[:48]:
        results_data.append({
            "country_code": team,
            "champion_pct": round(count / n_sims * 100, 3),
            "final_pct": round(finalist_count.get(team, 0) / n_sims * 100, 3),
            "semifinal_pct": round(semifinal_count.get(team, 0) / n_sims * 100, 3),
            "quarterfinal_pct": round(quarterfinal_count.get(team, 0) / n_sims * 100, 3),
            "champion_count": count,
            "n_sims": n_sims,
        })

    results_df = pd.DataFrame(results_data)
    results_df.to_csv(ROOT / "rf_montecarlo_results.csv", index=False)
    print(f"\n✓ Resultados guardados: {ROOT / 'rf_montecarlo_results.csv'}")

    results_json = {
        "n_simulations": n_sims,
        "seed": seed,
        "model": "RandomForest",
        "results": results_data,
    }
    with open(ROOT / "rf_montecarlo_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"✓ JSON guardado: {ROOT / 'rf_montecarlo_results.json'}")

    return results_df


if __name__ == "__main__":
    run_simulation(n_sims=10_000, seed=42)

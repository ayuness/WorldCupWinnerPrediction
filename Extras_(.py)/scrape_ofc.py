import re
import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict


OFC_DIRECTOS = {"NZL"}
OFC_PLAYOFF  = {"NCL"}
ALL_OFC      = OFC_DIRECTOS | OFC_PLAYOFF


def calc_stats_from_matches(soup) -> dict:
    """
    Calcula PJ, G, E, P, GF, GC, DG, Pts para NZL y NCL
    acumulando TODOS los partidos de todas las etapas.
    """
    stats = defaultdict(lambda: {"pj":0,"g":0,"e":0,"p":0,"gf":0,"gc":0})

    for match in soup.find_all("div", class_=re.compile(r"match-row_matchRowContainer")):
        teams  = match.find_all("abbr")
        scores = match.find_all("span", class_=re.compile(r"match-row_score__"))
        if len(teams) < 2 or len(scores) < 2:
            continue

        home = teams[0].get_text(strip=True)
        away = teams[1].get_text(strip=True)

        # Solo procesar equipos de interés
        if home not in ALL_OFC and away not in ALL_OFC:
            continue

        try:
            hs  = int(scores[0].get_text(strip=True))
            as_ = int(scores[1].get_text(strip=True))
        except ValueError:
            continue

        for team, gf, gc in [(home, hs, as_), (away, as_, hs)]:
            if team not in ALL_OFC:
                continue
            stats[team]["pj"] += 1
            stats[team]["gf"] += gf
            stats[team]["gc"] += gc
            if gf > gc:
                stats[team]["g"] += 1
            elif gf == gc:
                stats[team]["e"] += 1
            else:
                stats[team]["p"] += 1

    return dict(stats)


def build_ofc_dataset(filepath: str = "eliminatoria_ofc.html") -> pd.DataFrame:

    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    stats = calc_stats_from_matches(soup)

    rows = []
    for code in ALL_OFC:
        s = stats.get(code)
        if not s:
            print(f"⚠️  {code}: sin partidos encontrados")
            continue

        pj = s["pj"]
        gf = s["gf"]
        gc = s["gc"]
        dg = gf - gc
        pts = s["g"] * 3 + s["e"]

        rows.append({
            "country_code": code,
            "status":       "confirmed" if code in OFC_DIRECTOS else "playoff",
            "pj":           pj,
            "g":            s["g"],
            "e":            s["e"],
            "p":            s["p"],
            "pts":          pts,
            "gf":           gf,
            "gc":           gc,
            "dg":           dg,
            "gf_per_game":  round(gf / pj, 3) if pj > 0 else None,
            "gc_per_game":  round(gc / pj, 3) if pj > 0 else None,
        })

    df = pd.DataFrame(rows)
    df = df[[
        "country_code", "status",
        "pj", "g", "e", "p", "pts",
        "gf", "gc", "dg",
        "gf_per_game", "gc_per_game"
    ]]
    df = df.sort_values(
        ["status", "pts"],
        ascending=[True, False]
    ).reset_index(drop=True)

    return df


def main():
    df = build_ofc_dataset()

    df.to_csv("eliminatorias_ofc.csv", index=False)
    df.to_excel("eliminatorias_ofc.xlsx", index=False)

    print(df.to_string())
    print(f"\n✅  eliminatorias_ofc.csv  ({len(df)} filas)")

    # Verificación
    assert len(df) == 2, f"Esperaba 2 filas, got {len(df)}"
    assert set(df["country_code"]) == ALL_OFC
    assert df["gf"].isna().sum() == 0

    nzl = df[df["country_code"] == "NZL"].iloc[0]
    ncl = df[df["country_code"] == "NCL"].iloc[0]
    assert nzl["pj"] == 5 and nzl["g"] == 5 and nzl["gf"] == 29 and nzl["gc"] == 1
    assert ncl["pj"] == 5 and ncl["g"] == 3 and ncl["gf"] == 10 and ncl["gc"] == 7

    print("✅  Verificación OK")


if __name__ == "__main__":
    main()

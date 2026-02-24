import re
import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict

DIRECTOS = {"USA", "MEX", "CAN", "PAN", "CUW", "HAI"}
PLAYOFF  = {"JAM", "SUR"}
ALL      = DIRECTOS | PLAYOFF


def parse_concacaf(filepath: str) -> pd.DataFrame:
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    stats = defaultdict(lambda: {"pj":0, "g":0, "e":0, "p":0, "gf":0, "gc":0})

    for match in soup.find_all("div", class_=re.compile(r"match-row_matchRowContainer")):
        label = match.find("span", class_=re.compile(r"match-row_bottomLabel"))
        etapa = label.get_text(strip=True).split("·")[0].strip() if label else ""
        if etapa != "Ronda final":
            continue

        teams  = match.find_all("abbr")
        scores = match.find_all("span", class_=re.compile(r"match-row_score__"))
        if len(teams) < 2 or len(scores) < 2:
            continue

        home = teams[0].get_text(strip=True)
        away = teams[1].get_text(strip=True)
        try:
            hs  = int(scores[0].get_text(strip=True))
            as_ = int(scores[1].get_text(strip=True))
        except ValueError:
            continue

        for team, gf, gc in [(home, hs, as_), (away, as_, hs)]:
            stats[team]["pj"] += 1
            stats[team]["gf"] += gf
            stats[team]["gc"] += gc
            if   gf > gc:  stats[team]["g"] += 1
            elif gf == gc: stats[team]["e"] += 1
            else:          stats[team]["p"] += 1

    tabla = sorted(
        stats.items(),
        key=lambda x: (x[1]["g"]*3 + x[1]["e"], x[1]["gf"] - x[1]["gc"]),
        reverse=True
    )

    rows = []

    for code in ["USA", "MEX", "CAN"]:
        rows.append({
            "country_code": code, "status": "confirmed", "phase": "anfitrion",
            "position": None, "pj": None, "g": None, "e": None, "p": None,
            "pts": None, "gf": None, "gc": None, "dg": None,
            "gf_per_game": None, "gc_per_game": None,
        })

    pos = 1
    for team, d in tabla:
        if team not in ALL:
            continue
        pj  = d["pj"]
        gf  = d["gf"]
        gc  = d["gc"]
        pts = d["g"] * 3 + d["e"]
        rows.append({
            "country_code": team,
            "status":       "confirmed" if team in DIRECTOS else "playoff",
            "phase":        "ronda_final",
            "position":     pos,
            "pj":           pj,
            "g":            d["g"],
            "e":            d["e"],
            "p":            d["p"],
            "pts":          pts,
            "gf":           gf,
            "gc":           gc,
            "dg":           gf - gc,
            "gf_per_game":  round(gf / pj, 3),
            "gc_per_game":  round(gc / pj, 3),
        })
        pos += 1

    df = pd.DataFrame(rows)
    df = df[["country_code","status","phase","position",
             "pj","g","e","p","pts","gf","gc","dg",
             "gf_per_game","gc_per_game"]]
    return df.sort_values("position", na_position="first").reset_index(drop=True)


def main():
    df = parse_concacaf("eliminatorias_concacaf.html")
    df.to_csv("eliminatorias_concacaf.csv", index=False)
    df.to_excel("eliminatorias_concacaf.xlsx", index=False)
    print(df.to_string())
    print(f"\nGuardado: eliminatorias_concacaf.csv ({len(df)} filas)")
    assert len(df) == 8
    assert set(df["country_code"]) == ALL
    print("Verificacion OK")


if __name__ == "__main__":
    main()

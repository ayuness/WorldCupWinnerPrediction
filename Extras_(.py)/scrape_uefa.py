import re
import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict


UEFA_DIRECTOS = {
    "GER","SUI","SCO","FRA","ESP","POR",
    "NED","AUT","NOR","BEL","ENG","CRO"
}
UEFA_PLAYOFF = {
    "SVK","KVX","DEN","UKR","TUR","IRL",
    "POL","BIH","ITA","WAL","MKD","ALB",
    "CZE","NIR","SWE","ROU"
}
ALL_UEFA = UEFA_DIRECTOS | UEFA_PLAYOFF

# Kosovo aparece como KOS en los HTMLs de FIFA pero KVX en el dataset de rankings
CODE_REMAP = {"KOS": "KVX"}


def remap(code: str) -> str:
    return CODE_REMAP.get(code, code)


def parse_standings(filepath: str) -> dict:
    """Retorna {country_code: {pj, g, e, p, dg, pts}} desde las tablas."""
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    standings = {}
    seen_groups = set()

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        group_th = rows[0].find("th")
        group = group_th.get_text(strip=True) if group_th else ""

        # Deduplicar — el HTML tiene cada tabla dos veces
        if group in seen_groups:
            continue
        seen_groups.add(group)

        for row in rows[1:]:
            cols = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            cols = [c for c in cols if c]
            if len(cols) < 8:
                continue
            try:
                code = remap(cols[1])
                standings[code] = {
                    "pj":  int(cols[2]),
                    "g":   int(cols[3]),
                    "e":   int(cols[4]),
                    "p":   int(cols[5]),
                    "dg":  int(cols[6]),
                    "pts": int(cols[7]),
                }
            except (ValueError, IndexError):
                continue

    return standings


def parse_gf_gc(filepath: str) -> dict:
    """Retorna {country_code: {gf, gc}} acumulando todos los partidos."""
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    gf_gc = defaultdict(lambda: {"gf": 0, "gc": 0})

    for match in soup.find_all("div", class_=re.compile(r"match-row_matchRowContainer")):
        teams  = match.find_all("abbr")
        scores = match.find_all("span", class_=re.compile(r"match-row_score__"))
        if len(teams) < 2 or len(scores) < 2:
            continue
        home = remap(teams[0].get_text(strip=True))
        away = remap(teams[1].get_text(strip=True))
        try:
            hs  = int(scores[0].get_text(strip=True))
            as_ = int(scores[1].get_text(strip=True))
        except ValueError:
            continue
        gf_gc[home]["gf"] += hs
        gf_gc[home]["gc"] += as_
        gf_gc[away]["gf"] += as_
        gf_gc[away]["gc"] += hs

    return dict(gf_gc)


def build_uefa_dataset(
    standings_file: str = "eliminatorias_uefa.html",
    matches_file:   str = "partidos_eliminatoria_uefa.html",
) -> pd.DataFrame:

    standings = parse_standings(standings_file)
    gf_gc     = parse_gf_gc(matches_file)

    rows = []
    for code in ALL_UEFA:
        s = standings.get(code)
        if not s:
            print(f"⚠️  {code}: no encontrado en standings — revisar")
            continue

        pj = s["pj"]
        gf = gf_gc.get(code, {}).get("gf", 0)
        gc = gf_gc.get(code, {}).get("gc", 0)

        rows.append({
            "country_code": code,
            "status":       "confirmed" if code in UEFA_DIRECTOS else "playoff",
            "pj":           pj,
            "g":            s["g"],
            "e":            s["e"],
            "p":            s["p"],
            "pts":          s["pts"],
            "gf":           gf,
            "gc":           gc,
            "dg":           s["dg"],
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
    # Ordenar: primero confirmed (por pts desc), luego playoff (por pts desc)
    df = df.sort_values(
        ["status", "pts"],
        ascending=[True, False]
    ).reset_index(drop=True)

    return df


def main():
    df = build_uefa_dataset()

    df.to_csv("eliminatorias_uefa.csv", index=False)
    df.to_excel("eliminatorias_uefa.xlsx", index=False)

    print(df.to_string())
    print(f"\n✅  eliminatorias_uefa.csv  ({len(df)} filas)")

    # Verificación
    assert len(df) == 28, f"Esperaba 28 filas, got {len(df)}"
    assert set(df["country_code"]) == ALL_UEFA, \
        f"Diferencia: {set(df['country_code']).symmetric_difference(ALL_UEFA)}"
    assert df["gf"].isna().sum() == 0, "GF no debe tener nulos"
    assert set(df[df["status"]=="confirmed"]["country_code"]) == UEFA_DIRECTOS
    assert set(df[df["status"]=="playoff"]["country_code"])   == UEFA_PLAYOFF
    print("✅  Verificación OK")


if __name__ == "__main__":
    main()

import re
import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict


CAF_DIRECTOS = {"EGY","SEN","RSA","CPV","MAR","CIV","ALG","TUN","GHA"}
CAF_PLAYOFF  = {"NGA"}
ALL_CAF      = CAF_DIRECTOS | CAF_PLAYOFF


def parse_standings(soup) -> dict:
    """Retorna {country_code: {pj, g, e, p, dg, pts}} desde tablas de posiciones."""
    standings = {}
    seen_groups = set()

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        group_th = rows[0].find("th")
        group = group_th.get_text(strip=True) if group_th else ""

        # Deduplicar — el HTML repite cada tabla dos veces
        if group in seen_groups:
            continue
        seen_groups.add(group)

        for row in rows[1:]:
            cols = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            cols = [c for c in cols if c]
            if len(cols) < 8:
                continue
            try:
                code = cols[1]
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


def parse_gf_gc(soup) -> dict:
    """Retorna {country_code: {gf, gc}} usando SOLO partidos de 'Ronda uno'."""
    gf_gc = defaultdict(lambda: {"gf": 0, "gc": 0})

    for match in soup.find_all("div", class_=re.compile(r"match-row_matchRowContainer")):
        # Filtrar etapa
        label = match.find("span", class_=re.compile(r"match-row_bottomLabel"))
        etapa = label.get_text(strip=True).split("·")[0].strip() if label else ""
        if etapa != "Ronda uno":
            continue  # ignorar Round Two

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

        gf_gc[home]["gf"] += hs
        gf_gc[home]["gc"] += as_
        gf_gc[away]["gf"] += as_
        gf_gc[away]["gc"] += hs

    return dict(gf_gc)


def build_caf_dataset(filepath: str = "eliminatoria_caf.html") -> pd.DataFrame:

    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    standings = parse_standings(soup)
    gf_gc     = parse_gf_gc(soup)

    rows = []
    for code in ALL_CAF:
        s = standings.get(code)
        if not s:
            print(f"⚠️  {code}: no encontrado en standings")
            continue

        pj = s["pj"]
        gf = gf_gc.get(code, {}).get("gf", 0)
        gc = gf_gc.get(code, {}).get("gc", 0)

        rows.append({
            "country_code": code,
            "status":       "confirmed" if code in CAF_DIRECTOS else "playoff",
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
    # Ordenar: confirmed primero (pts desc), luego playoff (pts desc)
    df = df.sort_values(
        ["status", "pts"],
        ascending=[True, False]
    ).reset_index(drop=True)

    return df


def main():
    df = build_caf_dataset()

    df.to_csv("eliminatorias_caf.csv", index=False)
    df.to_excel("eliminatorias_caf.xlsx", index=False)

    print(df.to_string())
    print(f"\n✅  eliminatorias_caf.csv  ({len(df)} filas)")

    # Verificación
    assert len(df) == 10, f"Esperaba 10 filas, got {len(df)}"
    assert set(df["country_code"]) == ALL_CAF
    assert df["gf"].isna().sum() == 0, "GF no debe tener nulos"
    assert set(df[df["status"]=="confirmed"]["country_code"]) == CAF_DIRECTOS
    assert set(df[df["status"]=="playoff"]["country_code"])   == CAF_PLAYOFF
    print("✅  Verificación OK")


if __name__ == "__main__":
    main()

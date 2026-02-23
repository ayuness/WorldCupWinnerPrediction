#!/usr/bin/env python3
"""
Dataset AFC — Eliminatorias Mundial 2026

Genera eliminatorias_afc.csv con una fila por equipo × fase.
Solo incluye los 9 equipos clasificados (8 directos + 1 playoff).
GF y GC se extraen de los resultados de partidos del mismo HTML.

Uso:
    python parse_afc.py
    python parse_afc.py --input eliminatorias_asia.html
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

# ─── Equipos clasificados ─────────────────────────────────────────────────────

AFC_DIRECTOS = {"JPN", "KOR", "IRN", "AUS", "UZB", "JOR", "QAT", "KSA"}
AFC_PLAYOFF  = {"IRQ"}
ALL_AFC      = AFC_DIRECTOS | AFC_PLAYOFF

# ─── Mapeo de etiquetas de fase ───────────────────────────────────────────────

# Tablas de posiciones
PHASE_TABLE_MAP = {
    "Ronda 4":  "4ta_ronda",
    "3ª fase":  "3ra_fase",
    "2ª fase":  "2da_ronda",
}

# Etiquetas en match containers
PHASE_MATCH_MAP = {
    "Segunda ronda": "2da_ronda",
    "Ronda tres":    "3ra_fase",
    "Cuarta ronda":  "4ta_ronda",
}

# Orden de fases para ordenar el CSV al final
PHASE_ORDER = {"2da_ronda": 0, "3ra_fase": 1, "4ta_ronda": 2}

# Columnas finales
COL_ORDER = [
    "confederation", "phase", "group", "position", "country_code", "status",
    "pj", "g", "e", "p", "gf", "gc", "dg", "pts",
    "pts_per_game", "win_pct", "gf_per_game", "gc_per_game",
]


# ─── Parte 1: Tablas de posiciones ───────────────────────────────────────────

def parse_standings_afc(soup: BeautifulSoup) -> list[dict]:
    """
    Extrae todas las filas de posición de las 3 fases.
    Deduplicación por (country_code, phase, group) porque las tablas
    aparecen duplicadas en el DOM.

    Estructura de fila (tras filtrar celdas vacías):
        [pos, code, PJ, G, E, P, DG, Pts]
    """
    records = []
    seen = set()

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue

        prev = table.find_previous(["h2", "h3", "h4", "h5"])
        phase_text = prev.get_text(strip=True) if prev else ""

        phase = None
        for keyword, phase_label in PHASE_TABLE_MAP.items():
            if keyword in phase_text:
                phase = phase_label
                break
        if phase is None:
            continue

        # Grupo desde primera celda del encabezado
        group_cell = rows[0].find(["th", "td"])
        raw_group  = group_cell.get_text(strip=True) if group_cell else "–"
        group = re.sub(r"[Gg]rupo\s*", "", raw_group).strip() or "–"

        for row in rows[1:]:
            cols = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            cols = [c for c in cols if c]

            if len(cols) < 7:
                continue

            # Detectar offset (celda de flag vacía puede haber quedado)
            if cols[0].isdigit():
                offset = 0
            elif len(cols) > 1 and cols[1].isdigit():
                offset = 1
            else:
                continue

            try:
                pos  = int(cols[offset])
                code = cols[offset + 1].upper()
                pj   = int(cols[offset + 2])
                g    = int(cols[offset + 3])
                e    = int(cols[offset + 4])
                p    = int(cols[offset + 5])
                dg   = int(cols[offset + 6])
                pts  = int(cols[offset + 7])
            except (ValueError, IndexError):
                continue

            key = (code, phase, group)
            if key in seen:
                continue
            seen.add(key)

            records.append({
                "phase": phase, "group": group,
                "position": pos, "country_code": code,
                "pj": pj, "g": g, "e": e, "p": p,
                "dg": dg, "pts": pts,
            })

    return records


# ─── Parte 2: GF/GC de resultados de partidos ────────────────────────────────

def parse_match_results_afc(soup: BeautifulSoup) -> dict:
    """
    Extrae goles a favor y en contra de cada equipo por fase.
    Retorna: {(team_code, phase): {"gf": int, "gc": int}}

    Estructura de cada match container:
        <abbr>HOME</abbr>  <abbr>AWAY</abbr>
        <span class="match-row_score__...">HS</span>
        <span class="match-row_score__...">AS</span>
        <span class="match-row_bottomLabel__...">Etiqueta fase · Estadio</span>
    """
    gf_gc: dict = {}
    containers = soup.find_all("div", class_=re.compile(r"match-row_matchRowContainer"))

    for match in containers:
        # Fase
        label = match.find("span", class_=re.compile(r"match-row_bottomLabel"))
        if not label:
            continue
        etapa_raw = label.get_text(strip=True).split("·")[0].strip()
        phase = PHASE_MATCH_MAP.get(etapa_raw)
        if phase is None:
            continue  # Ronda uno y Round Five no nos interesan

        # Equipos y marcador
        teams  = [a.get_text(strip=True) for a in match.find_all("abbr")]
        scores = match.find_all("span", class_=re.compile(r"match-row_score__"))

        if len(teams) < 2 or len(scores) < 2:
            continue

        try:
            hs  = int(scores[0].get_text(strip=True))
            aws = int(scores[1].get_text(strip=True))
        except ValueError:
            continue

        home, away = teams[0], teams[1]
        for team, gf, gc in [(home, hs, aws), (away, aws, hs)]:
            key = (team, phase)
            if key not in gf_gc:
                gf_gc[key] = {"gf": 0, "gc": 0}
            gf_gc[key]["gf"] += gf
            gf_gc[key]["gc"] += gc

    return gf_gc


# ─── Parte 3: Combinar y filtrar ─────────────────────────────────────────────

def build_afc_dataset(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    standings = parse_standings_afc(soup)
    gf_gc     = parse_match_results_afc(soup)

    records = []
    for row in standings:
        code  = row["country_code"]
        phase = row["phase"]

        if code not in ALL_AFC:
            continue

        gf_gc_data = gf_gc.get((code, phase), {})
        gf = gf_gc_data.get("gf")
        gc = gf_gc_data.get("gc")
        pj = row["pj"]

        records.append({
            "confederation": "AFC",
            "phase":         phase,
            "group":         row["group"],
            "position":      row["position"],
            "country_code":  code,
            "status":        "confirmed" if code in AFC_DIRECTOS else "playoff",
            "pj":            pj,
            "g":             row["g"],
            "e":             row["e"],
            "p":             row["p"],
            "gf":            gf,
            "gc":            gc,
            "dg":            row["dg"],
            "pts":           row["pts"],
            "pts_per_game":  round(row["pts"] / pj, 3),
            "win_pct":       round(row["g"]   / pj, 3),
            "gf_per_game":   round(gf / pj, 3) if gf is not None else None,
            "gc_per_game":   round(gc / pj, 3) if gc is not None else None,
        })

    return records


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Dataset AFC — Eliminatorias Mundial 2026")
    _base = Path(__file__).parent.parent
    ap.add_argument("--input",
                    default=str(_base / "Local" / "eliminatorias_asia.html"),
                    metavar="HTML",
                    help="Ruta del HTML fuente")
    ap.add_argument("--output",
                    default=str(_base / "eliminatorias_afc.csv"),
                    metavar="CSV",
                    help="Ruta del CSV de salida")
    args = ap.parse_args()

    if not Path(args.input).exists():
        print(f"[!] Archivo no encontrado: {args.input}")
        return

    records = build_afc_dataset(args.input)
    if not records:
        print("[!] No se obtuvieron registros.")
        return

    df = pd.DataFrame(records)
    df = (df[COL_ORDER]
            .sort_values(
                ["phase", "group", "position"],
                key=lambda col: col.map(PHASE_ORDER) if col.name == "phase" else col
            )
            .reset_index(drop=True))

    df.to_csv(args.output, index=False)

    # ─── Resumen ──────────────────────────────────────────────────────────────
    print(df.to_string())
    print(f"\n[OK] {args.output}  ({len(df)} filas)")
    print(f"\nFilas por fase:")
    print(
        df.groupby("phase", sort=False)["country_code"]
          .apply(sorted)
          .to_string()
    )

    # ─── Verificación ─────────────────────────────────────────────────────────
    codes = set(df["country_code"].unique())
    missing = ALL_AFC - codes
    if missing:
        print(f"\n[!] Equipos sin datos: {missing}")
    else:
        print(f"\n[OK] Los 9 equipos clasificados tienen datos.")

    nulos = df[df["gf"].isna()]
    if not nulos.empty:
        print(f"[!] Filas con GF nulo: {len(nulos)}")
        print(nulos[["country_code", "phase"]].to_string())
    else:
        print("[OK] Sin filas con GF nulo.")


if __name__ == "__main__":
    main()

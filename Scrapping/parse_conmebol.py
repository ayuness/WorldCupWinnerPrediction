#!/usr/bin/env python3
"""
Dataset CONMEBOL — Eliminatorias Mundial 2026

Genera eliminatorias_conmebol.csv con las mismas columnas que eliminatorias_afc.csv.
Solo incluye los 7 equipos clasificados (6 directos + 1 playoff).

Uso:
    python parse_conmebol.py
    python parse_conmebol.py --input eliminatorias_conmebol.html
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

# ─── Clasificación ────────────────────────────────────────────────────────────
# Top 6 clasifican directo, 7mo va a repechaje intercontinental

CONMEBOL_DIRECTOS = {"ARG", "ECU", "COL", "URU", "BRA", "PAR"}
CONMEBOL_PLAYOFF  = {"BOL"}
ALL_CONMEBOL      = CONMEBOL_DIRECTOS | CONMEBOL_PLAYOFF

COL_ORDER = [
    "confederation", "phase", "group", "position", "country_code", "status",
    "pj", "g", "e", "p", "gf", "gc", "dg", "pts",
    "pts_per_game", "win_pct", "gf_per_game", "gc_per_game",
]


# ─── Parser ───────────────────────────────────────────────────────────────────

def parse_conmebol(html: str) -> list[dict]:
    """
    Parsea la tabla única de CONMEBOL.

    Estructura de fila (tras filtrar celdas vacías):
        [pos, NombreCODE, PJ, G, E, P, GF, GC, DG, Pts]

    GF y GC están directamente en la tabla (a diferencia de AFC).
    No hay match containers en el HTML.
    """
    soup = BeautifulSoup(html, "html.parser")
    records = []

    table = soup.find("table")
    if not table:
        return records

    for row in table.find_all("tr")[1:]:  # saltar fila de encabezado
        cols = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
        cols = [c for c in cols if c]

        # Estructura esperada: [pos, NombreCODE, PJ, G, E, P, GF, GC, DG, Pts]
        if len(cols) < 9:
            continue

        try:
            pos = int(cols[0])
        except ValueError:
            continue

        # Extraer código de 3 letras al final del campo "ArgentinaARG"
        code_match = re.search(r"([A-Z]{3})$", cols[1])
        if not code_match:
            continue
        code = code_match.group(1)

        # Solo clasificados (posiciones 1-7)
        if code not in ALL_CONMEBOL:
            continue

        try:
            pj  = int(cols[2])
            g   = int(cols[3])
            e   = int(cols[4])
            p   = int(cols[5])
            gf  = int(cols[6])
            gc  = int(cols[7])
            dg  = int(cols[8])
            pts = int(cols[9]) if len(cols) > 9 else None
        except (ValueError, IndexError):
            continue

        records.append({
            "confederation": "CONMEBOL",
            "phase":         "liga_unica",
            "group":         "–",
            "position":      pos,
            "country_code":  code,
            "status":        "confirmed" if code in CONMEBOL_DIRECTOS else "playoff",
            "pj":            pj,
            "g":             g,
            "e":             e,
            "p":             p,
            "gf":            gf,
            "gc":            gc,
            "dg":            dg,
            "pts":           pts,
            "pts_per_game":  round(pts / pj, 3) if pts else None,
            "win_pct":       round(g   / pj, 3),
            "gf_per_game":   round(gf  / pj, 3),
            "gc_per_game":   round(gc  / pj, 3),
        })

    return records


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Dataset CONMEBOL — Eliminatorias Mundial 2026")
    _base = Path(__file__).parent.parent
    ap.add_argument("--input",
                    default=str(_base / "Local" / "eliminatorias_conmebol.html"),
                    metavar="HTML",
                    help="Ruta del HTML fuente")
    ap.add_argument("--output",
                    default=str(_base / "eliminatorias_conmebol.csv"),
                    metavar="CSV",
                    help="Ruta del CSV de salida")
    args = ap.parse_args()

    if not Path(args.input).exists():
        print(f"[!] Archivo no encontrado: {args.input}")
        return

    with open(args.input, "r", encoding="utf-8") as f:
        html = f.read()

    records = parse_conmebol(html)
    if not records:
        print("[!] No se obtuvieron registros.")
        return

    df = pd.DataFrame(records)[COL_ORDER].sort_values("position").reset_index(drop=True)
    df.to_csv(args.output, index=False)

    # ─── Resumen ──────────────────────────────────────────────────────────────
    print(df.to_string())
    print(f"\n[OK] {args.output}  ({len(df)} filas)")

    # ─── Verificación ─────────────────────────────────────────────────────────
    missing = ALL_CONMEBOL - set(df["country_code"])
    if missing:
        print(f"\n[!] Equipos sin datos: {missing}")
    else:
        print(f"\n[OK] Los 7 equipos clasificados tienen datos.")


if __name__ == "__main__":
    main()

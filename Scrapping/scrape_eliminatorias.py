#!/usr/bin/env python3
"""
Scraper de eliminatorias del Mundial 2026 — todas las confederaciones.

Uso:
    python scrape_eliminatorias.py                      # Playwright (online)
    python scrape_eliminatorias.py --local              # HTMLs locales en directorio actual
    python scrape_eliminatorias.py --local --local-dir ./Local
    python scrape_eliminatorias.py --confederations AFC CONMEBOL
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

# ─── Rutas de archivos locales ────────────────────────────────────────────────

LOCAL_HTML_FILES = {
    "AFC":      "eliminatorias_asia.html",
    "CAF":      "eliminatorias_caf.html",
    "CONCACAF": "eliminatorias_concacaf.html",
    "CONMEBOL": "eliminatorias_conmebol.html",
    "UEFA":     "eliminatorias_uefa.html",
    "OFC":      "eliminatorias_ofc.html",
}

URLS = {
    "AFC":      "https://www.fifa.com/es/tournaments/mens/worldcup/canadamexicousa2026/qualifiers/afc/standings",
    "CAF":      "https://www.fifa.com/es/tournaments/mens/worldcup/canadamexicousa2026/qualifiers/caf/standings",
    "CONCACAF": "https://www.fifa.com/es/tournaments/mens/worldcup/canadamexicousa2026/qualifiers/concacaf/standings",
    "CONMEBOL": "https://www.fifa.com/es/tournaments/mens/worldcup/canadamexicousa2026/qualifiers/conmebol/standings",
    "UEFA":     "https://www.fifa.com/es/tournaments/mens/worldcup/canadamexicousa2026/qualifiers/uefa/standings",
    "OFC":      "https://www.fifa.com/es/tournaments/mens/worldcup/canadamexicousa2026/qualifiers/ofc/standings",
}

# Columnas finales en orden
COLS_ORDER = [
    "confederation", "phase", "group", "position", "country_code",
    "pj", "g", "e", "p", "gf", "gc", "dg", "pts",
    "pts_per_game", "win_pct", "gf_per_game", "gc_per_game",
]


# ─── AFC ──────────────────────────────────────────────────────────────────────

def parse_afc(html: str) -> list[dict]:
    """
    Parsea eliminatorias AFC.
    - Solo extrae la '3ª fase' (fase definitiva con clasificados directos).
    - El header viene duplicado en el DOM → usar 'in' para detectarlo.
    - Las tablas están duplicadas → deduplicar con set (country_code, group).
    - No tiene GF ni GC, solo DG.
    - Resultado esperado: 18 registros (3 grupos × 6 equipos).

    Estructura de fila (tras filtrar vacíos):
        [pos, code, PJ, G, E, P, DG, Pts]
    """
    soup = BeautifulSoup(html, "html.parser")
    records = []
    seen = set()  # (country_code, group)

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue

        prev = table.find_previous(["h2", "h3", "h4", "h5"])
        phase_text = prev.get_text(strip=True) if prev else ""
        if "3ª fase" not in phase_text:
            continue

        # Grupo desde la primera celda del encabezado
        group_cell = rows[0].find(["th", "td"])
        raw_group  = group_cell.get_text(strip=True) if group_cell else "–"
        group = re.sub(r"[Gg]rupo\s*", "", raw_group).strip() or "–"

        for row in rows[1:]:
            cols = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            cols = [c for c in cols if c]

            if len(cols) < 7:
                continue

            # Detectar offset (celda de flag puede quedar como vacío ya filtrado)
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

            key = (code, group)
            if key in seen:
                continue
            seen.add(key)

            records.append({
                "confederation": "AFC",
                "phase":         "3ra_fase",
                "group":         group,
                "position":      pos,
                "country_code":  code,
                "pj": pj, "g": g, "e": e, "p": p,
                "gf": None, "gc": None,
                "dg": dg,   "pts": pts,
            })

    return records


# ─── CONMEBOL ─────────────────────────────────────────────────────────────────

def parse_conmebol(html: str) -> list[dict]:
    """
    Parsea eliminatorias CONMEBOL.
    - Una sola tabla, sin grupos (liguilla única).
    - Nombre y código pegados: 'ArgentinaARG' → extraer código con regex.
    - Tiene GF y GC separados.
    - Resultado esperado: 10 registros.

    Estructura de fila (tras filtrar vacíos):
        [pos, NombreCODE, PJ, G, E, P, GF, GC, DG, Pts]
    """
    soup = BeautifulSoup(html, "html.parser")
    records = []

    table = soup.find("table")
    if not table:
        return records

    for row in table.find_all("tr")[1:]:  # saltar encabezado
        cols = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
        cols = [c for c in cols if c]

        if len(cols) < 9:
            continue

        try:
            pos = int(cols[0]) if cols[0].isdigit() else None
            if pos is None:
                continue

            code_match = re.search(r"([A-Z]{3})$", cols[1])
            if not code_match:
                continue
            code = code_match.group(1)

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
            "pj": pj, "g": g, "e": e, "p": p,
            "gf": gf, "gc": gc,
            "dg": dg, "pts": pts,
        })

    return records


# ─── Parsers pendientes (agregar cuando tengas los HTMLs) ─────────────────────

# def parse_caf(html: str) -> list[dict]: ...
# def parse_concacaf(html: str) -> list[dict]: ...
# def parse_uefa(html: str) -> list[dict]: ...
# def parse_ofc(html: str) -> list[dict]: ...

PARSERS = {
    "AFC":      parse_afc,
    "CONMEBOL": parse_conmebol,
    # "CAF":      parse_caf,
    # "CONCACAF": parse_concacaf,
    # "UEFA":     parse_uefa,
    # "OFC":      parse_ofc,
}


# ─── Carga de HTML ────────────────────────────────────────────────────────────

def load_html_local(confederation: str, local_dir: Path) -> str | None:
    path = local_dir / LOCAL_HTML_FILES[confederation]
    if not path.exists():
        print(f"  [!] Archivo no encontrado: {path}", file=sys.stderr)
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_html_playwright(confederation: str) -> str | None:
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        print(
            "  [!] Playwright no instalado.\n"
            "      Ejecuta: pip install playwright && playwright install chromium",
            file=sys.stderr,
        )
        return None

    url = URLS[confederation]
    print(f"  Cargando: {url}")
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            )
            page.goto(url, wait_until="networkidle", timeout=30_000)
            page.wait_for_selector("table", timeout=15_000)
            html = page.content()
            browser.close()
            return html
    except PWTimeout:
        print(f"  [!] Timeout — FIFA puede estar bloqueando ({confederation}).", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"  [!] Error Playwright ({confederation}): {exc}", file=sys.stderr)
        return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Scraper eliminatorias Mundial 2026")
    ap.add_argument("--local", action="store_true",
                    help="Leer HTMLs locales en lugar de Playwright")
    ap.add_argument("--local-dir", type=Path, default=Path("."), metavar="DIR",
                    help="Directorio con los HTMLs (default: directorio actual)")
    ap.add_argument("--confederations", nargs="+",
                    default=list(PARSERS.keys()),
                    choices=list(PARSERS.keys()), metavar="CONF",
                    help=f"Confederaciones a procesar (disponibles: {', '.join(PARSERS)})")
    ap.add_argument("--output", type=Path, default=Path("eliminatorias_2026"), metavar="NOMBRE",
                    help="Nombre base del CSV de salida (default: eliminatorias_2026)")
    args = ap.parse_args()

    all_records: list[dict] = []

    for conf in args.confederations:
        print(f"\n[{conf}]")
        html = load_html_local(conf, args.local_dir) if args.local else load_html_playwright(conf)

        if html is None:
            print(f"  [!] Sin HTML para {conf}, omitiendo.")
            continue

        records = PARSERS[conf](html)

        if not records:
            print(f"  [!] No se encontraron datos en el HTML de {conf}.")
            continue

        print(f"  OK — {len(records)} equipos  |  fase: {records[0]['phase']!r}")
        all_records.extend(records)

    if not all_records:
        print("\n[!] No se obtuvieron datos. Abortando.", file=sys.stderr)
        sys.exit(1)

    # ─── DataFrame y métricas ─────────────────────────────────────────────────
    df = pd.DataFrame(all_records)

    df["pts_per_game"] = (df["pts"] / df["pj"]).round(3)
    df["win_pct"]      = (df["g"]   / df["pj"]).round(3)
    df["gf_per_game"]  = (df["gf"]  / df["pj"]).round(3)  # NaN donde GF no está (AFC)
    df["gc_per_game"]  = (df["gc"]  / df["pj"]).round(3)  # NaN donde GC no está (AFC)

    df = (df[COLS_ORDER]
            .sort_values(["confederation", "group", "position"])
            .reset_index(drop=True))

    # ─── Guardar CSV ──────────────────────────────────────────────────────────
    csv_path = args.output.with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    # ─── Resumen ──────────────────────────────────────────────────────────────
    print(f"\n[OK] {csv_path}  ({len(df)} filas, {df['confederation'].nunique()} confederaciones)")
    print("\nRegistros por confederación:")
    print(df.groupby("confederation")["country_code"].count().to_string())


if __name__ == "__main__":
    main()

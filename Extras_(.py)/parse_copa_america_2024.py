#!/usr/bin/env python3
"""
Parser Copa América 2024 — Transfermarkt HTML → CSV

Extrae los 32 partidos de la Copa América 2024 (24 fase de grupos + 4 cuartos
+ 2 semifinales + 1 tercer puesto + 1 final) desde el HTML local de
Transfermarkt y genera un CSV con el mismo esquema que
Data/Copa_America/Copa_America.csv.

Uso:
    python parse_copa_america_2024.py
    python parse_copa_america_2024.py --input Local/copaA_2024.html --output Data/Copa_America/Copa_America_2024.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, NavigableString

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT  = ROOT / "Local" / "copaA_2024.html"
DEFAULT_OUTPUT = ROOT / "Data" / "Copa_America" / "Copa_America_2024.csv"

COLUMNS = ["Data", "Casa", "Fora", "Gols Casa", "Gols Fora", "Edição", "Fase"]

ROUND_MAP = {
    "cuartos de final":            "Cuartos de final",
    "semifinales":                 "Semifinales",
    "partido por el tercer puesto":"Tercer puesto",
    "final":                       "Final",
}

DATE_RE = re.compile(r"(\d{2})/(\d{2})/(\d{4})")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _score(ergebnis_a) -> tuple[int, int]:
    """Extrae (gols_casa, gols_fora) desde <a class="ergebnis-link">X:Y[<span>...</span>]</a>."""
    # Primer hijo de texto (antes de cualquier <span class="ergebnis_zusatz">)
    raw = ""
    for child in ergebnis_a.children:
        if isinstance(child, NavigableString):
            raw = str(child).strip()
            if raw:
                break
    if ":" not in raw:
        raise ValueError(f"Score inesperado: {ergebnis_a!r}")
    h, a = raw.split(":", 1)
    return int(h.strip()), int(a.strip())


def _iso_date(text: str) -> str:
    m = DATE_RE.search(text or "")
    if not m:
        raise ValueError(f"Fecha no encontrada en: {text!r}")
    d, mo, y = m.groups()
    return f"{y}-{mo}-{d}"


def _team(td) -> str:
    """Nombre de selección desde un <td> con <a title='...'>Nombre</a>."""
    a = td.find("a", title=True)
    if a is None:
        raise ValueError(f"No team anchor in: {td!r}")
    return a.get_text(strip=True) or a["title"].strip()


def _extract_match_row(tr, fase: str, fallback_date: str | None) -> dict:
    """
    Extrae un partido desde un <tr> de datos (sin clase bg_blau_20/bg_Sturm).

    Estrategia robusta: localizar los <td> por clase, independientemente de
    que la fila sea de fase de grupos o de eliminatorias.
    """
    casa_td = tr.find("td", class_=lambda c: c and "text-right" in c and "hauptlink" in c)
    fora_td = tr.find("td", class_=lambda c: c and "no-border-links" in c and "hauptlink" in c)
    score_a = tr.find("a", class_="ergebnis-link")

    if not (casa_td and fora_td and score_a):
        raise ValueError(f"Fila de partido incompleta: {tr!r}")

    gols_casa, gols_fora = _score(score_a)

    # Fecha: primer <td class="hide-for-small"> de la fila, si no usar fallback
    date_td = tr.find("td", class_="hide-for-small")
    date_text = date_td.get_text(" ", strip=True) if date_td else ""
    try:
        data = _iso_date(date_text)
    except ValueError:
        if fallback_date is None:
            raise
        data = fallback_date

    return {
        "Data": data,
        "Casa": _team(casa_td),
        "Fora": _team(fora_td),
        "Gols Casa": gols_casa,
        "Gols Fora": gols_fora,
        "Edição": 2024,
        "Fase": fase,
    }


# ─── Parser principal ─────────────────────────────────────────────────────────

def parse(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []

    # ── Fase de grupos ────────────────────────────────────────────────────────
    for h2 in soup.find_all("h2", class_="content-box-headline"):
        title = h2.get_text(strip=True)
        if not title.startswith("Grupo "):
            continue
        fase = title  # "Grupo A", "Grupo B", ...
        box = h2.find_parent("div", class_="box")
        if box is None:
            continue
        tables = box.find_all("table")
        if len(tables) < 2:
            continue
        match_table = tables[1]  # la primera es la tabla de posiciones
        tbody = match_table.find("tbody")
        if tbody is None:
            continue

        fallback_date: str | None = None
        for tr in tbody.find_all("tr", recursive=False):
            classes = tr.get("class") or []
            if "bg_blau_20" in classes:
                # Fila mobile de fecha — la usamos como fallback para la fila siguiente
                txt = tr.get_text(" ", strip=True)
                m = DATE_RE.search(txt)
                if m:
                    d, mo, y = m.groups()
                    fallback_date = f"{y}-{mo}-{d}"
                continue
            try:
                rows.append(_extract_match_row(tr, fase, fallback_date))
            except ValueError:
                # Filas que no son partidos (cabeceras, etc.)
                continue

    # ── Eliminatorias ─────────────────────────────────────────────────────────
    elim_h2 = next(
        (h2 for h2 in soup.find_all("h2", class_="content-box-headline")
         if h2.get_text(strip=True).lower() == "eliminatorias"),
        None,
    )
    if elim_h2 is not None:
        box = elim_h2.find_parent("div", class_="box")
        # Cada ronda vive en su propio <tbody>, encabezado por <tr class="bg_Sturm">
        current_fase: str | None = None
        fallback_date = None
        for tr in box.find_all("tr"):
            classes = tr.get("class") or []
            if "bg_Sturm" in classes:
                label = tr.get_text(" ", strip=True).lower()
                current_fase = None
                for key, value in ROUND_MAP.items():
                    if key in label:
                        current_fase = value
                        break
                fallback_date = None
                continue
            if current_fase is None:
                continue
            if "bg_blau_20" in classes:
                txt = tr.get_text(" ", strip=True)
                m = DATE_RE.search(txt)
                if m:
                    d, mo, y = m.groups()
                    fallback_date = f"{y}-{mo}-{d}"
                continue
            try:
                rows.append(_extract_match_row(tr, current_fase, fallback_date))
            except ValueError:
                continue

    return rows


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",  type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    html = args.input.read_text(encoding="utf-8")
    rows = parse(html)

    df = pd.DataFrame(rows, columns=COLUMNS)

    # Validación
    expected_counts = {
        "Grupo A": 6, "Grupo B": 6, "Grupo C": 6, "Grupo D": 6,
        "Cuartos de final": 4, "Semifinales": 2,
        "Tercer puesto": 1, "Final": 1,
    }
    counts = df["Fase"].value_counts().to_dict()
    assert len(df) == 32, f"Esperaba 32 partidos, obtuve {len(df)}\n{counts}"
    for fase, n in expected_counts.items():
        got = counts.get(fase, 0)
        assert got == n, f"{fase}: esperaba {n}, obtuve {got}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    final_row = df[df["Fase"] == "Final"].iloc[0]
    winner = final_row["Casa"] if final_row["Gols Casa"] > final_row["Gols Fora"] else final_row["Fora"]

    print(f"OK — {len(df)} partidos → {args.output}")
    for fase, n in expected_counts.items():
        print(f"  {fase:20s} {n}")
    print(f"Campeón: {winner}")


if __name__ == "__main__":
    main()

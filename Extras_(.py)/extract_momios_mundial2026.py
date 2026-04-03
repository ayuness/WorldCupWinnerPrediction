"""
Extracción unificada de momios (Caliente + Codere + FanDuel) — Mundial 2026.

Genera: Data/momios_mundial2026.csv
Fuentes:
  - Caliente: HTML local (caliente_mundial_2026.html)
  - Codere: datos hardcodeados (momios americanos)
  - FanDuel: datos hardcodeados para Grupo I (momios americanos)

Fecha snapshot: 1 de abril de 2026.
"""

import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

HTML_FILE = Path(__file__).parent / "caliente_mundial_2026.html"
OUTPUT_FILE = Path(__file__).parent / "Data" / "momios_mundial2026.csv"

# ---------------------------------------------------------------------------
# Composición oficial de los grupos (validación)
# ---------------------------------------------------------------------------

GRUPOS_MUNDIAL_2026 = {
    "A": ["MEX", "RSA", "KOR", "CZE"],
    "B": ["SUI", "CAN", "QAT", "BIH"],
    "C": ["BRA", "MAR", "SCO", "HAI"],
    "D": ["USA", "TUR", "PAR", "AUS"],
    "E": ["GER", "ECU", "CIV", "CUW"],
    "F": ["NED", "JPN", "SWE", "TUN"],
    "G": ["BEL", "EGY", "IRN", "NZL"],
    "H": ["ESP", "URU", "KSA", "CPV"],
    "I": ["FRA", "SEN", "IRQ", "NOR"],
    "J": ["ARG", "AUT", "ALG", "JOR"],
    "K": ["POR", "COL", "COD", "UZB"],
    "L": ["ENG", "CRO", "GHA", "PAN"],
}

# ---------------------------------------------------------------------------
# Mapeo nombre → country_code
# ---------------------------------------------------------------------------

NAME_TO_CODE = {
    # Caliente names
    "España": "ESP", "Inglaterra": "ENG", "Francia": "FRA", "Brasil": "BRA",
    "Argentina": "ARG", "Portugal": "POR", "Alemania": "GER", "Países Bajos": "NED",
    "Noruega": "NOR", "Bélgica": "BEL", "Colombia": "COL", "EE.UU": "USA",
    "Marruecos": "MAR", "Suiza": "SUI", "Ecuador": "ECU", "Japón": "JPN",
    "Turquía": "TUR", "Uruguay": "URU", "Croacia": "CRO", "México": "MEX",
    "Senegal": "SEN", "Austria": "AUT", "Suecia": "SWE", "Paraguay": "PAR",
    "Canada": "CAN", "Escocia": "SCO", "Republica Checa": "CZE", "Egipto": "EGY",
    "Costa de Marfil": "CIV", "Bosnia & Herzegovina": "BIH", "Argelia": "ALG",
    "Ghana": "GHA", "Corea del Sur": "KOR", "Túnez": "TUN", "Irán": "IRN",
    "Australia": "AUS", "Congo DR": "COD", "Sudafrica": "RSA", "Nueva Zelanda": "NZL",
    "Arabia Saudita": "KSA", "Qatar": "QAT", "Panamá": "PAN", "Irak": "IRQ",
    "Jordania": "JOR", "Uzbekistan": "UZB", "Haiti": "HAI", "Curazao": "CUW",
    "Cabo Verde": "CPV",
    # Codere-specific variations
    "EE.UU.": "USA", "República Checa": "CZE", "Sudáfrica": "RSA",
    "Bosnia-Herzegovina": "BIH", "Catar": "QAT", "Haití": "HAI",
    "Arabia Saudí": "KSA", "Uzbekistán": "UZB",
    "Rep. Dem. del Congo": "COD",
    # Fallback accented variants
    "Canadá": "CAN",
}

# ---------------------------------------------------------------------------
# Datos hardcodeados — Codere (momios americanos)
# ---------------------------------------------------------------------------

CODERE_GRUPOS = {
    "A": {
        "México": +100,
        "República Checa": +225,
        "Corea del Sur": +350,
        "Sudáfrica": +1100,
    },
    "B": {
        "Suiza": -112,
        "Canadá": +250,
        "Bosnia-Herzegovina": +275,
        "Catar": +2400,
    },
    "C": {
        "Brasil": -625,
        "Marruecos": +600,
        "Escocia": +1300,
        "Haití": +9900,
    },
    "D": {
        "EE.UU.": +145,
        "Turquía": +165,
        "Paraguay": +400,
        "Australia": +700,
    },
    "E": {
        "Alemania": -358,
        "Ecuador": +450,
        "Costa de Marfil": +800,
        "Curazao": +9900,
    },
    "F": {
        "Países Bajos": -134,
        "Japón": +350,
        "Suecia": +400,
        "Túnez": +700,
    },
    "H": {
        "España": -500,
        "Uruguay": +450,
        "Arabia Saudí": +1900,
        "Cabo Verde": +4900,
    },
    "J": {
        "Argentina": -334,
        "Austria": +450,
        "Argelia": +600,
        "Jordania": +3900,
    },
    "K": {
        "Portugal": -250,
        "Colombia": +275,
        "Rep. Dem. del Congo": +800,
        "Uzbekistán": +3400,
    },
    "L": {
        "Inglaterra": -358,
        "Croacia": +400,
        "Ghana": +1100,
        "Panamá": +4900,
    },
}

# ---------------------------------------------------------------------------
# Datos hardcodeados — FanDuel (Grupo I, momios americanos)
# ---------------------------------------------------------------------------

FANDUEL_GRUPO_I = {
    "I": {
        "Francia": -175,
        "Noruega": +250,
        "Senegal": +700,
        "Irak": +3000,
    },
}

# ---------------------------------------------------------------------------
# Funciones de conversión
# ---------------------------------------------------------------------------


def us_odds_to_implied_prob(odds: int) -> float:
    """Convierte momios americanos a probabilidad implícita."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def normalize_probs(probs: dict[str, float]) -> dict[str, float]:
    """Normaliza probabilidades para que sumen 1.0 (remueve vig)."""
    total = sum(probs.values())
    return {k: v / total for k, v in probs.items()}


def resolve_code(name: str) -> str | None:
    """Resuelve un nombre de equipo a su country_code."""
    code = NAME_TO_CODE.get(name)
    if code is None:
        # Intentar sin acentos o variaciones menores
        stripped = name.strip()
        code = NAME_TO_CODE.get(stripped)
    if code is None:
        print(f"  WARNING: nombre no mapeado → '{name}'")
    return code


# ---------------------------------------------------------------------------
# PASO 1: Extraer datos de Caliente (HTML)
# ---------------------------------------------------------------------------

def extract_caliente(html_path: Path):
    """Parsea el HTML de Caliente y retorna los tres mercados."""
    print("=== PASO 1: Extrayendo datos de Caliente ===")

    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Encontrar todos los bloques de mercado
    markets = soup.select("div.mkt")

    ganador = {}          # team_name → odds_dec
    grupos_caliente = {}  # group_letter → {team_name: odds_dec}
    for mkt in markets:
        name_span = mkt.select_one("span.mkt-name")
        if not name_span:
            continue
        mkt_name = name_span.get_text(strip=True)

        # Extraer selecciones de este mercado
        selections = {}
        for li in mkt.select("li"):
            seln_span = li.select_one("span.seln-name")
            dec_span = li.select_one("span.price.dec")
            if seln_span and dec_span:
                team = seln_span.get_text(strip=True)
                try:
                    odds_d = float(dec_span.get_text(strip=True))
                except ValueError:
                    continue
                selections[team] = odds_d

        if mkt_name == "Ganador":
            ganador = selections
        elif "Grupo" in mkt_name and "Ganador" in mkt_name:
            # Extraer letra del grupo: "Grupo C - Ganador" → "C"
            parts = mkt_name.split()
            for i, p in enumerate(parts):
                if p == "Grupo" and i + 1 < len(parts):
                    letter = parts[i + 1].strip(" -")
                    if len(letter) == 1 and letter.isalpha():
                        grupos_caliente[letter] = selections
                    break

    print(f'Mercado "Ganador": {len(ganador)} equipos encontrados')
    print(f'Mercado "Grupos": {len(grupos_caliente)} grupos → {sorted(grupos_caliente.keys())}')

    return ganador, grupos_caliente


# ---------------------------------------------------------------------------
# PASO 2: Procesar Codere y FanDuel
# ---------------------------------------------------------------------------

def process_hardcoded_source(source_data: dict, source_name: str) -> dict[str, dict[str, float]]:
    """Convierte datos hardcodeados (US odds) a clean probs por grupo.

    Returns: {group_letter: {country_code: clean_prob}}
    """
    result = {}
    for group_letter, teams in source_data.items():
        implied = {}
        for team_name, us_odds in teams.items():
            code = resolve_code(team_name)
            if code is None:
                print(f"  ERROR: No se pudo mapear '{team_name}' ({source_name}, Grupo {group_letter})")
                sys.exit(1)
            implied[code] = us_odds_to_implied_prob(us_odds)
        result[group_letter] = normalize_probs(implied)
    return result


# ---------------------------------------------------------------------------
# PASO 3: Caliente grupos → clean probs
# ---------------------------------------------------------------------------

def caliente_groups_to_clean(grupos_caliente: dict) -> dict[str, dict[str, float]]:
    """Convierte grupos de Caliente (odds decimales) a clean probs por grupo.

    Returns: {group_letter: {country_code: clean_prob}}
    """
    result = {}
    for group_letter, teams in grupos_caliente.items():
        implied = {}
        for team_name, odds_dec in teams.items():
            code = resolve_code(team_name)
            if code is None:
                print(f"  ERROR: No se pudo mapear '{team_name}' (Caliente, Grupo {group_letter})")
                sys.exit(1)
            implied[code] = 1.0 / odds_dec
        result[group_letter] = normalize_probs(implied)
    return result


def caliente_market_to_clean(market: dict) -> dict[str, float]:
    """Convierte un mercado de Caliente (odds decimales) a clean probs.

    Returns: {country_code: clean_prob}
    """
    implied = {}
    for team_name, odds_dec in market.items():
        code = resolve_code(team_name)
        if code is None:
            print(f"  WARNING: equipo sin mapear en mercado Caliente → '{team_name}'")
            continue
        implied[code] = 1.0 / odds_dec
    return normalize_probs(implied)


# ---------------------------------------------------------------------------
# PASO 4: Combinar fuentes por grupo
# ---------------------------------------------------------------------------

def combine_groups(
    caliente_clean: dict[str, dict[str, float]],
    codere_clean: dict[str, dict[str, float]],
    fanduel_clean: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Combina fuentes según la lógica de combinación especificada.

    Returns: {group_letter: {country_code: final_prob}}
    """
    print("\n=== PASO 3: Combinando fuentes ===")

    combined = {}
    for letter in "ABCDEFGHIJKL":
        has_caliente = letter in caliente_clean
        has_codere = letter in codere_clean
        has_fanduel = letter in fanduel_clean

        if has_fanduel:
            source_label = "FanDuel"
            combined[letter] = fanduel_clean[letter]
        elif has_caliente and has_codere:
            source_label = "Promedio"
            # Promediar y re-normalizar
            cal = caliente_clean[letter]
            cod = codere_clean[letter]
            merged = {}
            all_codes = set(cal.keys()) | set(cod.keys())
            for code in all_codes:
                p_cal = cal.get(code)
                p_cod = cod.get(code)
                if p_cal is not None and p_cod is not None:
                    merged[code] = (p_cal + p_cod) / 2.0
                elif p_cal is not None:
                    merged[code] = p_cal
                else:
                    merged[code] = p_cod
            combined[letter] = normalize_probs(merged)
        elif has_caliente:
            source_label = "Caliente only"
            combined[letter] = caliente_clean[letter]
        elif has_codere:
            source_label = "Codere only"
            combined[letter] = codere_clean[letter]
        else:
            print(f"  ERROR: Grupo {letter} sin datos de ninguna fuente")
            sys.exit(1)

        # Imprimir resumen
        sorted_teams = sorted(combined[letter].items(), key=lambda x: -x[1])
        teams_str = " ".join(f"{c}({p:.4f})" for c, p in sorted_teams)
        label = f"Grupo {letter}: {source_label:14s}"
        print(f"{label} → {teams_str}")

    return combined


# ---------------------------------------------------------------------------
# Validaciones
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame):
    """Ejecuta todas las validaciones sobre el DataFrame final."""
    print("\n=== VALIDACIONES ===")
    errors = []

    # 12 grupos
    groups = sorted(df["group_letter"].unique())
    if len(groups) == 12:
        print("✅ 12 grupos procesados (I vía FanDuel)")
    else:
        errors.append(f"❌ Se esperaban 12 grupos, se encontraron {len(groups)}: {groups}")

    # 48 filas
    if len(df) == 48:
        print("✅ 48 filas en CSV")
    else:
        errors.append(f"❌ Se esperaban 48 filas, se encontraron {len(df)}")

    # Todos los codes mapeados
    unmapped = df[df["country_code"].isna()]
    if len(unmapped) == 0:
        print("✅ Todos los country_codes mapeados")
    else:
        errors.append(f"❌ {len(unmapped)} country_codes sin mapear")

    # group_win_prob suma 1.0 por grupo
    group_sums = df.groupby("group_letter")["group_win_prob"].sum()
    all_ok = True
    for g, s in group_sums.items():
        if abs(s - 1.0) > 0.001:
            errors.append(f"❌ Grupo {g}: group_win_prob suma {s:.6f}")
            all_ok = False
    if all_ok:
        print("✅ group_win_prob suman 1.0 por grupo")

    # champion_prob suma 1.0
    champ_sum = df["champion_prob"].sum()
    if abs(champ_sum - 1.0) < 0.001:
        print("✅ champion_prob suma 1.0 global")
    elif df["champion_prob"].isna().all():
        errors.append("❌ champion_prob: todas las filas son NaN")
    else:
        errors.append(f"❌ champion_prob suma {champ_sum:.6f}")

    # Validar composición de grupos
    for letter, expected_codes in GRUPOS_MUNDIAL_2026.items():
        actual_codes = set(df[df["group_letter"] == letter]["country_code"])
        expected_set = set(expected_codes)
        if actual_codes != expected_set:
            errors.append(
                f"❌ Grupo {letter}: esperado {expected_set}, obtenido {actual_codes}"
            )

    # 4 equipos por grupo
    group_counts = df.groupby("group_letter").size()
    for g, c in group_counts.items():
        if c != 4:
            errors.append(f"❌ Grupo {g}: {c} equipos (se esperan 4)")

    if errors:
        print("\n--- ERRORES ---")
        for e in errors:
            print(e)
        sys.exit(1)
    else:
        print("✅ Composición de grupos verificada contra FIFA oficial")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Paso 1: Caliente ---
    if not HTML_FILE.exists():
        print(f"ERROR: No se encontró {HTML_FILE}")
        print("Descarga el HTML de Caliente y guárdalo como caliente_mundial_2026.html")
        sys.exit(1)

    ganador_raw, grupos_caliente_raw = extract_caliente(HTML_FILE)

    # Convertir mercados de Caliente a clean probs
    caliente_group_clean = caliente_groups_to_clean(grupos_caliente_raw)
    champion_clean = caliente_market_to_clean(ganador_raw) if ganador_raw else {}

    # --- Paso 2: Codere + FanDuel ---
    print("\n=== PASO 2: Procesando datos de Codere ===")
    codere_clean = process_hardcoded_source(CODERE_GRUPOS, "Codere")
    print(f"Grupos hardcodeados: {len(CODERE_GRUPOS)} → {sorted(CODERE_GRUPOS.keys())}")

    print("\nProcesando datos de FanDuel (Grupo I)...")
    fanduel_clean = process_hardcoded_source(FANDUEL_GRUPO_I, "FanDuel")

    # --- Paso 3: Combinar ---
    combined = combine_groups(caliente_group_clean, codere_clean, fanduel_clean)

    # --- Construir DataFrame ---
    rows = []
    for letter in "ABCDEFGHIJKL":
        group_probs = combined[letter]
        # Ordenar por prob descendente
        for code, prob in sorted(group_probs.items(), key=lambda x: -x[1]):
            rows.append({
                "country_code": code,
                "group_letter": letter,
                "group_win_prob": round(prob, 6),
                "champion_prob": round(champion_clean.get(code, float("nan")), 6)
                    if code in champion_clean else float("nan"),
            })

    df = pd.DataFrame(rows)

    # --- Validaciones ---
    validate(df)

    # --- Guardar ---
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n=== ARCHIVO GENERADO ===")
    print(f"✅ {OUTPUT_FILE.name} ({len(df)} filas, {len(df.columns)} columnas)")
    print(f"   Ruta: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

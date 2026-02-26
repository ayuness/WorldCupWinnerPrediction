import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup


# ── Configuración ──────────────────────────────────────────────────────────────

BASE_URL = (
    "https://www.transfermarkt.com/vereins-statistik/"
    "wertvollstenationalmannschaften/marktwertetop"
    "?kontinent_id=0&plus=1&page={page}"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

NAME_TO_CODE = {
    "Argentina": "ARG", "Brazil": "BRA", "Colombia": "COL",
    "Ecuador": "ECU", "Paraguay": "PAR", "Uruguay": "URU",
    "Chile": "CHI", "Bolivia": "BOL", "Peru": "PER", "Venezuela": "VEN",
    "England": "ENG", "France": "FRA", "Spain": "ESP", "Portugal": "POR",
    "Germany": "GER", "Italy": "ITA", "Netherlands": "NED", "Belgium": "BEL",
    "Norway": "NOR", "Switzerland": "SUI", "Scotland": "SCO", "Croatia": "CRO",
    "Austria": "AUT", "Türkiye": "TUR", "Sweden": "SWE", "Denmark": "DEN",
    "Ukraine": "UKR", "Serbia": "SRB", "Poland": "POL", "Albania": "ALB",
    "Czech Republic": "CZE", "Bosnia-Herzegovina": "BIH", "Wales": "WAL",
    "Romania": "ROU", "North Macedonia": "MKD",
    "Republic of Ireland": "IRL", "Slovakia": "SVK",
    "Kosovo": "KOS", "Northern Ireland": "NIR",
    # Variantes de nombre en transfermarkt.com
    "Türkiye": "TUR", "Turkiye": "TUR",
    "Democratic Republic of the Congo": "COD",
    "Morocco": "MAR", "Senegal": "SEN", "Ivory Coast": "CIV",
    "Nigeria": "NGA", "Cameroon": "CMR", "Egypt": "EGY", "Ghana": "GHA",
    "Algeria": "ALG", "Tunisia": "TUN", "South Africa": "RSA",
    "DR Congo": "COD", "Cape Verde": "CPV",
    "Japan": "JPN", "South Korea": "KOR", "Iran": "IRN",
    "Australia": "AUS", "Qatar": "QAT", "Saudi Arabia": "KSA",
    "Uzbekistan": "UZB", "Jordan": "JOR", "Iraq": "IRQ",
    "United States": "USA", "Mexico": "MEX", "Canada": "CAN",
    "Panama": "PAN", "Honduras": "HON", "Costa Rica": "CRC",
    "Jamaica": "JAM", "Suriname": "SUR",
    "New Zealand": "NZL", "New Caledonia": "NCL",
}

TARGET_CODES = set(NAME_TO_CODE.values())


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_market_value(text: str) -> float | None:
    """
    Convierte texto de valor a millones de euros.
    Formatos soportados:
      "€1.30bn"        → 1300.0
      "€51.80m"        → 51.80
      "1.30 billion €" → 1300.0   (legacy / local HTML)
      "€932.00 million"→ 932.0    (legacy / local HTML)
    """
    text = text.strip().replace(",", ".")
    # Formato .com: "€1.30bn"
    m = re.search(r"([\d.]+)bn", text, re.IGNORECASE)
    if m:
        return round(float(m.group(1)) * 1000, 2)
    # Formato .com: "€51.80m"
    m = re.search(r"([\d.]+)m\b", text, re.IGNORECASE)
    if m:
        return round(float(m.group(1)), 2)
    # Formato legacy: "1.30 billion €"
    m = re.search(r"([\d.]+)\s*billion", text, re.IGNORECASE)
    if m:
        return round(float(m.group(1)) * 1000, 2)
    # Formato legacy: "€932.00 million"
    m = re.search(r"([\d.]+)\s*million", text, re.IGNORECASE)
    if m:
        return round(float(m.group(1)), 2)
    return None


def parse_page(html: str) -> list[dict]:
    """
    Extrae todas las filas de una página.

    Estructura real (verificada): tabla[1], filas con 7 <td> directos:
      td[0] → rank
      td[1] → country name
      td[2] → confederation
      td[3] → members
      td[4] → avg_age
      td[5] → market_value total
      td[6] → avg_player_value
    Las filas decorativas (bandera) tienen 2 <td> y se saltan automáticamente.
    """
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    if len(tables) < 2:
        print("⚠️  Tabla principal no encontrada")
        return []

    rows_out = []
    for row in tables[1].find_all("tr"):
        tds = row.find_all("td", recursive=False)
        if len(tds) != 7:
            continue

        try:
            rank    = int(tds[0].get_text(strip=True))
            name    = tds[1].get_text(strip=True)
            conf    = tds[2].get_text(strip=True)
            members_txt = tds[3].get_text(strip=True)
            members = int(members_txt) if members_txt != "-" else None
            age_txt = tds[4].get_text(strip=True).replace(",", ".")
            avg_age = float(age_txt) if age_txt != "-" else None
            mv_total = parse_market_value(tds[5].get_text(strip=True))
            mv_avg   = parse_market_value(tds[6].get_text(strip=True))
        except (ValueError, AttributeError) as e:
            print(f"  ⚠️  Error parseando fila: {e}")
            continue

        rows_out.append({
            "rank_tm":                rank,
            "country_name_tm":        name,
            "country_code":           NAME_TO_CODE.get(name),
            "confederation":          conf,
            "members":                members,
            "avg_age":                avg_age,
            "market_value_eur_m":     mv_total,
            "avg_player_value_eur_m": mv_avg,
        })

    return rows_out


# ── Scraping ───────────────────────────────────────────────────────────────────

def scrape_all_pages(total_pages: int = 10) -> pd.DataFrame:
    all_rows = []

    for page in range(1, total_pages + 1):
        url = BASE_URL.format(page=page)
        print(f"Descargando página {page}/10 → {url}")

        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  ❌ Error en página {page}: {e}")
            continue

        rows = parse_page(resp.text)
        print(f"  → {len(rows)} equipos encontrados")
        all_rows.extend(rows)

        if page < total_pages:
            time.sleep(2)

    return pd.DataFrame(all_rows)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    df_all = scrape_all_pages()
    print(f"\nTotal equipos scrapeados: {len(df_all)}")

    # Filtrar las 64 selecciones del Mundial 2026 y deduplicar
    df = df_all[df_all["country_code"].isin(TARGET_CODES)].copy()
    df = df.drop_duplicates(subset="rank_tm")
    df = df.sort_values("rank_tm").reset_index(drop=True)

    print(f"Selecciones del Mundial 2026 encontradas: {len(df)}")

    # Detectar selecciones no encontradas
    found   = set(df["country_code"])
    missing = TARGET_CODES - found
    if missing:
        print(f"\n⚠️  No encontradas en Transfermarkt: {sorted(missing)}")
        print("   → Pueden aparecer con otro nombre; revisar NAME_TO_CODE")

    # Detectar nombres sin mapeo en df_all (para depuración)
    unmapped = df_all[df_all["country_code"].isna()]["country_name_tm"].unique()
    if len(unmapped) > 0:
        print(f"\n⚠️  Nombres sin mapeo en NAME_TO_CODE: {list(unmapped)}")

    df.to_csv("transfermarkt_selecciones.csv", index=False)
    df.to_excel("transfermarkt_selecciones.xlsx", index=False)

    print("\n=== TOP 20 ===")
    print(df.head(20).to_string())
    print(f"\n✅  transfermarkt_selecciones.csv  ({len(df)} filas)")


if __name__ == "__main__":
    main()

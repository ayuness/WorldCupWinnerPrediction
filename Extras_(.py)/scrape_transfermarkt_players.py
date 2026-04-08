"""
Scrape historical + current player rosters from Transfermarkt
for all 48 World Cup 2026 qualified teams.

Outputs: Data/transfermarkt_players.csv
"""

import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

# ── Configuración ──────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

DELAY_SECONDS = 3  # Respetar rate-limit

# 48 equipos clasificados al Mundial 2026
# Formato: country_code -> (verein_id, slug)
TEAM_REGISTRY = {
    "ALG": ("3614", "algerien"),
    "ARG": ("3437", "argentinien"),
    "AUS": ("3433", "australien"),
    "AUT": ("3383", "osterreich"),
    "BEL": ("3382", "belgien"),
    "BIH": ("3446", "bosnien-herzegowina"),
    "BRA": ("3439", "brasilien"),
    "CAN": ("3510", "kanada"),
    "CIV": ("3591", "elfenbeinkuste"),
    "CGO": ("3854", "demokratische-republik-kongo"),  # DR Congo (COD en momios)
    "COL": ("3816", "kolumbien"),
    "CPV": ("4311", "kap-verde"),
    "CRO": ("3556", "kroatien"),
    "CUW": ("32364", "curacao"),
    "CZE": ("3445", "tschechien"),
    "ECU": ("5750", "ecuador"),
    "EGY": ("3672", "agypten"),
    "ENG": ("3299", "england"),
    "ESP": ("3375", "spanien"),
    "FRA": ("3377", "frankreich"),
    "GER": ("3262", "deutschland"),
    "GHA": ("3441", "ghana"),
    "HAI": ("14161", "haiti"),
    "IRN": ("3582", "iran"),
    "IRQ": ("3560", "irak"),
    "JOR": ("15737", "jordanien"),
    "JPN": ("3435", "japan"),
    "KOR": ("3589", "sudkorea"),
    "KSA": ("3807", "saudi-arabien"),
    "MAR": ("3575", "marokko"),
    "MEX": ("6303", "mexiko"),
    "NED": ("3379", "niederlande"),
    "NOR": ("3440", "norwegen"),
    "NZL": ("9171", "neuseeland"),
    "PAN": ("3577", "panama"),
    "PAR": ("3581", "paraguay"),
    "POR": ("3300", "portugal"),
    "QAT": ("14162", "katar"),
    "RSA": ("3806", "sudafrika"),
    "SCO": ("3380", "schottland"),
    "SEN": ("3499", "senegal"),
    "SUI": ("3384", "schweiz"),
    "SWE": ("3557", "schweden"),
    "TUN": ("3670", "tunesien"),
    "TUR": ("3381", "turkei"),
    "URU": ("3449", "uruguay"),
    "USA": ("3505", "vereinigte-staaten"),
    "UZB": ("3563", "usbekistan"),
}

# Años a scrapear: WC years, Euro years, Copa years + current
SCRAPE_YEARS = [
    2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014,
    2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025,
]

# Top 5 ligas europeas (para top_league_pct feature)
TOP_LEAGUES_PATTERNS = [
    # Premier League clubs
    "Arsenal", "Chelsea", "Liverpool", "Man City", "Manchester City", "Man Utd",
    "Manchester United", "Tottenham", "Newcastle", "Aston Villa", "Brighton",
    "West Ham", "Wolves", "Crystal Palace", "Fulham", "Bournemouth", "Everton",
    "Nottingham", "Brentford", "Leicester", "Southampton", "Ipswich",
    # La Liga
    "Real Madrid", "Barcelona", "Atlético", "Atletico Madrid", "Real Sociedad",
    "Villarreal", "Real Betis", "Sevilla FC", "Athletic", "Valencia",
    "Girona", "Celta", "Mallorca", "Osasuna", "Las Palmas", "Espanyol",
    "Getafe", "Alavés", "Rayo Vallecano", "Leganés", "Real Valladolid",
    # Serie A
    "Inter Milan", "AC Milan", "Juventus", "Napoli", "AS Roma", "Roma",
    "Lazio", "Atalanta", "Fiorentina", "Bologna", "Torino", "Udinese",
    "Genoa", "Monza", "Cagliari", "Parma", "Empoli", "Como", "Hellas Verona",
    "Venezia", "Lecce",
    # Bundesliga
    "Bayern Munich", "Bay. Munich", "B. Dortmund", "Bor. Dortmund", "RB Leipzig",
    "Bayer Leverkusen", "B. Leverkusen", "Eintracht Frankfurt", "E. Frankfurt",
    "Stuttgart", "VfB Stuttgart", "Wolfsburg", "Freiburg", "1.FC Union Berlin",
    "Hoffenheim", "Augsburg", "Mainz", "Werder Bremen", "Bochum", "Heidenheim",
    "Darmstadt", "FC Köln", "Mönchengladbach", "St. Pauli", "Holstein Kiel",
    # Ligue 1
    "Paris SG", "PSG", "Paris Saint-Germain", "Marseille", "Monaco", "Lille",
    "Lyon", "Nice", "Lens", "Rennes", "Strasbourg", "Nantes", "Toulouse",
    "Montpellier", "Brest", "Reims", "Le Havre", "Clermont", "Lorient", "Metz",
    "Auxerre", "Angers", "Saint-Étienne",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_market_value(text: str) -> float | None:
    """Convierte texto de valor a millones de euros."""
    text = text.strip().replace(",", ".")
    m = re.search(r"([\d.]+)bn", text, re.IGNORECASE)
    if m:
        return round(float(m.group(1)) * 1000, 2)
    m = re.search(r"([\d.]+)m\b", text, re.IGNORECASE)
    if m:
        return round(float(m.group(1)), 2)
    m = re.search(r"([\d.]+)\s*billion", text, re.IGNORECASE)
    if m:
        return round(float(m.group(1)) * 1000, 2)
    m = re.search(r"([\d.]+)\s*million", text, re.IGNORECASE)
    if m:
        return round(float(m.group(1)), 2)
    # Thousands: "€500k" or "€500Th."
    m = re.search(r"([\d.]+)\s*(?:k|th)", text, re.IGNORECASE)
    if m:
        return round(float(m.group(1)) / 1000, 4)
    return None


def is_top_league(club_name: str) -> bool:
    """Determinar si el club pertenece a una de las top 5 ligas."""
    if not club_name:
        return False
    for pattern in TOP_LEAGUES_PATTERNS:
        if pattern.lower() in club_name.lower():
            return True
    return False


def parse_squad_page(html: str, country_code: str, year: int) -> list[dict]:
    """
    Parsear la página de plantilla de un equipo nacional.

    Estructura HTML verificada (con /plus/1):
      td[0]: número de camiseta + clase CSS de posición (bg_Torwart, bg_Abwehr, bg_Mittelfeld, bg_Sturm)
      td[1]: nombre + posición detallada en texto
      td[2]: fecha de nacimiento con edad entre paréntesis, ej "Mar 7, 1994 (30)"
      td[3]: bandera (vacío)
      td[4]: altura
      td[5]: pie dominante
      td[6]: internacionalidades
      td[7]: goles
      td[8]: fecha de debut
      td[9]: valor de mercado, ej "€20.00m"
    """
    soup = BeautifulSoup(html, "html.parser")
    players = []

    table = soup.find("table", class_="items")
    if not table:
        return players

    tbody = table.find("tbody")
    if not tbody:
        return players

    rows = tbody.find_all("tr", recursive=False)

    for row in rows:
        tds = row.find_all("td", recursive=False)
        if len(tds) < 10:
            continue

        try:
            # Posición desde la clase CSS de td[0]
            td0_classes = " ".join(tds[0].get("class", []))
            if "bg_Torwart" in td0_classes:
                pos_category = "GK"
            elif "bg_Abwehr" in td0_classes:
                pos_category = "DEF"
            elif "bg_Sturm" in td0_classes:
                pos_category = "FWD"
            else:
                pos_category = "MID"

            # Nombre del jugador desde td[1]
            player_name = ""
            for link in tds[1].find_all("a"):
                href = link.get("href", "")
                if "/profil/spieler/" in href:
                    player_name = link.get_text(strip=True)
                    break

            if not player_name:
                continue

            # Club desde td[3] (contiene logo + link del club)
            club = ""
            for link in tds[3].find_all("a"):
                title = link.get("title", "")
                if title:
                    club = title
                    break
            if not club:
                for img in tds[3].find_all("img"):
                    title = img.get("title", "")
                    if title and title != player_name:
                        club = title
                        break

            # Edad desde td[2]: "Mar 7, 1994 (30)"
            age = None
            age_text = tds[2].get_text(strip=True)
            age_match = re.search(r"\((\d+)\)", age_text)
            if age_match:
                age = int(age_match.group(1))

            # Valor de mercado desde td[9]: "€20.00m"
            mv_text = tds[9].get_text(strip=True)
            market_value = parse_market_value(mv_text) if mv_text and mv_text != "-" else None

            players.append({
                "country_code": country_code,
                "year": year,
                "player_name": player_name,
                "position": pos_category,
                "age": age,
                "market_value_eur_m": market_value,
                "club": club,
                "is_top_league": is_top_league(club),
            })

        except Exception:
            continue

    return players


# ── Scraping ───────────────────────────────────────────────────────────────────

def scrape_team_year(country_code: str, verein_id: str, slug: str, year: int) -> list[dict]:
    """Scrapear plantilla de un equipo en un año dado."""
    url = f"https://www.transfermarkt.us/{slug}/kader/verein/{verein_id}/saison_id/{year}/plus/1"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return []
        return parse_squad_page(resp.text, country_code, year)
    except requests.RequestException as e:
        print(f"  Error: {e}")
        return []


def scrape_all():
    """Scrapear todas las combinaciones equipo × año."""
    all_players = []
    total = len(TEAM_REGISTRY) * len(SCRAPE_YEARS)
    done = 0

    for code, (vid, slug) in sorted(TEAM_REGISTRY.items()):
        for year in SCRAPE_YEARS:
            done += 1
            players = scrape_team_year(code, vid, slug, year)

            if players:
                all_players.extend(players)
                if done % 20 == 0 or done == total:
                    print(f"  [{done}/{total}] {code} {year}: {len(players)} jugadores "
                          f"(total acumulado: {len(all_players)})")
            else:
                if done % 50 == 0:
                    print(f"  [{done}/{total}] {code} {year}: sin datos")

            time.sleep(DELAY_SECONDS)

    return pd.DataFrame(all_players)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Scrapeando {len(TEAM_REGISTRY)} equipos × {len(SCRAPE_YEARS)} años = "
          f"{len(TEAM_REGISTRY) * len(SCRAPE_YEARS)} páginas")
    print(f"Delay: {DELAY_SECONDS}s por página")
    print(f"Tiempo estimado: ~{len(TEAM_REGISTRY) * len(SCRAPE_YEARS) * DELAY_SECONDS / 60:.0f} minutos\n")

    df = scrape_all()

    print(f"\nTotal jugadores scrapeados: {len(df)}")
    print(f"Equipos únicos: {df['country_code'].nunique()}")
    print(f"Años únicos: {sorted(df['year'].unique())}")

    # Guardar
    out_path = Path(__file__).parent.parent / "Data" / "transfermarkt_players.csv"
    df.to_csv(out_path, index=False)
    print(f"\nGuardado en {out_path}")

    # Stats
    print(f"\nJugadores por posición:")
    print(df['position'].value_counts())

    print(f"\nTop 10 jugadores por valor de mercado:")
    top = df.nlargest(10, 'market_value_eur_m')
    for _, p in top.iterrows():
        print(f"  {p['player_name']} ({p['country_code']} {p['year']}): "
              f"€{p['market_value_eur_m']}M - {p['club']}")


if __name__ == "__main__":
    main()

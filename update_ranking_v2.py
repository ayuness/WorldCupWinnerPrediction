"""
update_ranking_v2.py
====================
Updates ranking_mundial_2026.csv to the final WC2026 team list:
  - Removes 6 wrong teams (NGA, CMR, COD, SRB, HON, CRC)
  - Adds 23 missing teams (fetched from FIFA country pages)
  - Adds `status` column: "confirmed" | "playoff"
  - Adds `team_name_es` column (Spanish names)
  - Saves ranking_mundial_2026_v2.csv and .xlsx
"""

import re
import time
import json
import requests
import pandas as pd

# ──────────────────────────────────────────────
# Team definitions
# ──────────────────────────────────────────────

CONFIRMED = {
    "MEX": "México",        "USA": "Estados Unidos",  "CAN": "Canadá",
    "ARG": "Argentina",     "BRA": "Brasil",           "ECU": "Ecuador",
    "URU": "Uruguay",       "PAR": "Paraguay",         "COL": "Colombia",
    "ENG": "Inglaterra",    "CRO": "Croacia",          "FRA": "Francia",
    "POR": "Portugal",      "NOR": "Noruega",          "NED": "Países Bajos",
    "GER": "Alemania",      "BEL": "Bélgica",          "AUT": "Austria",
    "ESP": "España",        "SUI": "Suiza",            "SCO": "Escocia",
    "MAR": "Marruecos",     "TUN": "Túnez",            "EGY": "Egipto",
    "ALG": "Argelia",       "GHA": "Ghana",            "CPV": "Cabo Verde",
    "RSA": "Sudáfrica",     "CIV": "Costa de Marfil",  "SEN": "Senegal",
    "JPN": "Japón",         "IRN": "Irán",             "UZB": "Uzbekistán",
    "KOR": "Corea del Sur", "JOR": "Jordania",         "AUS": "Australia",
    "QAT": "Catar",         "KSA": "Arabia Saudita",
    "PAN": "Panamá",        "CUW": "Curazao",          "HAI": "Haití",
    "NZL": "Nueva Zelanda",
}

PLAYOFF = {
    "BOL": "Bolivia",               "NCL": "Nueva Caledonia",
    "CGO": "Rep. del Congo",        "IRQ": "Irak",
    "JAM": "Jamaica",               "SUR": "Surinam",
    "IRL": "Irlanda",               "UKR": "Ucrania",
    "POL": "Polonia",               "ITA": "Italia",
    "ALB": "Albania",               "CZE": "Rep. Checa",
    "SVK": "Eslovaquia",            "WAL": "Gales",
    "ROU": "Rumania",               "MKD": "Macedonia del Norte",
    "SWE": "Suecia",                "NIR": "Irlanda del Norte",
    "TUR": "Turquía",               "BIH": "Bosnia y Herzegovina",
    "DEN": "Dinamarca",             "KVX": "Kosovo",
}

ALL_TEAMS = {**CONFIRMED, **PLAYOFF}

# ──────────────────────────────────────────────
# DATE_MAP — FIFA dateId closest to June 15
# ──────────────────────────────────────────────

DATE_MAP = {
    1993: ("id2",     "1993-08-08"),
    1994: ("id11",    "1994-06-14"),
    1995: ("id20",    "1995-06-13"),
    1996: ("id31",    "1996-07-03"),
    1997: ("id40",    "1997-06-18"),
    1998: ("id50",    "1998-05-20"),
    1999: ("id62",    "1999-06-16"),
    2000: ("id74",    "2000-06-07"),
    2001: ("id86",    "2001-06-20"),
    2002: ("id98",    "2002-07-03"),
    2003: ("id110",   "2003-06-25"),
    2004: ("id122",   "2004-06-09"),
    2005: ("id134",   "2005-06-15"),
    2006: ("id146",   "2006-07-12"),
    2007: ("id8198",  "2007-06-13"),
    2008: ("id8555",  "2008-06-04"),
    2009: ("id8919",  "2009-06-03"),
    2010: ("id9276",  "2010-05-26"),
    2011: ("id9675",  "2011-06-29"),
    2012: ("id10018", "2012-06-06"),
    2013: ("id10383", "2013-06-06"),
    2014: ("id10747", "2014-06-05"),
    2015: ("id11111", "2015-06-04"),
    2016: ("id11475", "2016-06-02"),
    2017: ("id11839", "2017-06-01"),
    2018: ("id12210", "2018-06-07"),
    2019: ("id12582", "2019-06-14"),
    2020: ("id12883", "2020-06-11"),
    2021: ("id13295", "2021-05-27"),
    2022: ("id13687", "2022-06-23"),
    2023: ("id14058", "2023-06-29"),
    2024: ("id14415", "2024-06-20"),
    2025: ("id14800", "2025-07-10"),
    2026: ("id14993", "2026-01-19"),
}

TARGET_YEARS = sorted(DATE_MAP.keys())  # 1993–2026

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

HTML_PATH = "ranking-historico-html.html"


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def parse_html_snapshot(path: str) -> dict[str, int]:
    """Parse Jan-2026 rendered HTML table → {code: rank}."""
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    rankings: dict[str, int] = {}

    for span in soup.find_all("span", class_=re.compile(r"rank-cell_rank__")):
        try:
            rank = int(span.get_text(strip=True))
        except ValueError:
            continue
        row = span.find_parent("tr")
        if row is None:
            continue
        link = row.find("a", href=re.compile(r"/fifa-world-ranking/\w+\?gender=men"))
        if link is None:
            continue
        m = re.search(r"/fifa-world-ranking/(\w+)\?gender=men", link["href"])
        if m:
            code = m.group(1).upper()
            if code not in rankings:
                rankings[code] = rank

    return rankings


def fetch_country_history(
    session: requests.Session,
    code: str,
    retries: int = 2,
) -> tuple[dict[int, int], str | None]:
    """
    GET fifa.com/fifa-world-ranking/{code}?gender=men, extract
    historyRanking["1"].historyTable.rows → {year: finalRank}.
    Returns ({}, None) on failure.
    """
    url = f"https://www.fifa.com/fifa-world-ranking/{code}?gender=men"
    for attempt in range(retries + 1):
        try:
            r = session.get(url, timeout=20)
            if r.status_code != 200:
                return {}, None

            m = re.search(
                r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
                r.text, re.DOTALL,
            )
            if not m:
                return {}, None

            data = json.loads(m.group(1))
            page_data = data["props"]["pageProps"]["pageData"]
            team_name: str | None = page_data.get("intro", {}).get("header")

            rows = (
                page_data
                .get("historyRanking", {})
                .get("1", {})
                .get("historyTable", {})
                .get("rows", [])
            )

            year_rank = {
                int(row["year"]): int(row["finalRank"])
                for row in rows
                if row.get("year") and row.get("finalRank")
            }
            return year_rank, team_name

        except (requests.RequestException, json.JSONDecodeError, KeyError) as exc:
            if attempt < retries:
                time.sleep(3)
            else:
                print(f"    ERROR [{code}]: {exc}")
                return {}, None

    return {}, None


def build_rows_for_team(
    code: str,
    year_rank: dict[int, int],
    snapshot_2026: dict[str, int],
    team_name_en: str | None,
) -> list[dict]:
    """Build one record per year for a single team."""
    rows = []
    status = "confirmed" if code in CONFIRMED else "playoff"
    name_es = ALL_TEAMS[code]
    name_en = team_name_en or code

    for year in TARGET_YEARS:
        _, date_str = DATE_MAP[year]

        if year == 2026:
            rank = snapshot_2026.get(code) or year_rank.get(year)
            src = "html_snapshot" if snapshot_2026.get(code) else "country_page"
        else:
            rank = year_rank.get(year)
            src = "country_page" if rank is not None else "missing"

        rows.append({
            "country_code": code,
            "year": year,
            "ranking": rank,
            "date_used": date_str,
            "team_name": name_en,
            "team_name_es": name_es,
            "status": status,
            "source": src,
        })
    return rows


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("Updating ranking_mundial_2026 → v2")
    print("=" * 65)

    # ── Load existing CSV ──
    df_old = pd.read_csv("ranking_mundial_2026.csv", index_col=["country_code", "year"])
    existing_codes = set(df_old.index.get_level_values("country_code").unique())

    codes_to_remove = existing_codes - set(ALL_TEAMS.keys())
    codes_to_add    = set(ALL_TEAMS.keys()) - existing_codes

    print(f"\nTeams to remove ({len(codes_to_remove)}): {sorted(codes_to_remove)}")
    print(f"Teams to add    ({len(codes_to_add)}):    {sorted(codes_to_add)}")

    # ── Step 1: Remove wrong teams ──
    df_keep = df_old.drop(index=list(codes_to_remove), level="country_code")
    print(f"\nAfter removal: {df_keep.index.get_level_values('country_code').nunique()} teams")

    # ── Step 2: Parse 2026 HTML snapshot ──
    print("\nParsing 2026 HTML snapshot…")
    snapshot_2026 = parse_html_snapshot(HTML_PATH)
    print(f"  {len(snapshot_2026)} teams found in snapshot")

    # ── Step 3: Fetch new teams from FIFA ──
    print(f"\nFetching {len(codes_to_add)} new teams from FIFA…")

    session = requests.Session()
    session.headers.update(HEADERS)
    print("  Warming up session…")
    try:
        session.get("https://www.fifa.com/fifa-world-ranking/men", timeout=15)
    except requests.RequestException:
        pass
    time.sleep(2)

    new_records: list[dict] = []
    codes_sorted = sorted(codes_to_add)

    for i, code in enumerate(codes_sorted, start=1):
        print(f"  [{i:2d}/{len(codes_sorted)}] {code} …", end=" ", flush=True)
        year_rank, team_name_en = fetch_country_history(session, code)
        time.sleep(1.2)

        if not year_rank:
            print("NO DATA — using NaN for all years")
        else:
            print(f"OK  ({len(year_rank)} years, '{team_name_en}')")

        new_records.extend(
            build_rows_for_team(code, year_rank, snapshot_2026, team_name_en)
        )

    # ── Step 4: Add metadata columns to existing rows ──
    df_keep = df_keep.reset_index()

    # Add status
    df_keep["status"] = df_keep["country_code"].map(
        lambda c: "confirmed" if c in CONFIRMED else "playoff"
    )
    # Add Spanish name
    df_keep["team_name_es"] = df_keep["country_code"].map(ALL_TEAMS)

    # Ensure source column exists
    if "source" not in df_keep.columns:
        df_keep["source"] = "country_page"

    # ── Step 5: Combine old (cleaned) + new rows ──
    df_new = pd.DataFrame(new_records)
    df_combined = pd.concat([df_keep, df_new], ignore_index=True)

    # Deduplicate (keep first occurrence per country_code+year)
    df_combined = df_combined.drop_duplicates(subset=["country_code", "year"], keep="first")

    # Cast types
    df_combined["ranking"] = pd.to_numeric(df_combined["ranking"], errors="coerce")
    df_combined["year"]    = df_combined["year"].astype(int)

    # Set MultiIndex
    df_combined = df_combined.set_index(["country_code", "year"]).sort_index()

    # ── Step 6: Save ──
    out_csv  = "ranking_mundial_2026_v2.csv"
    out_xlsx = "ranking_mundial_2026_v2.xlsx"
    df_combined.to_csv(out_csv)
    df_combined.to_excel(out_xlsx)
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_xlsx}")

    # ── Step 7: Verification ──
    unique_codes = sorted(df_combined.index.get_level_values("country_code").unique())
    n_teams = len(unique_codes)

    print(f"\n{'='*65}")
    print(f"df.shape      : {df_combined.shape}")
    print(f"Unique teams  : {n_teams}  (expected {len(ALL_TEAMS)})")
    print(f"Years per team: {len(TARGET_YEARS)}")

    total = len(df_combined)
    filled = df_combined["ranking"].notna().sum()
    print(f"Coverage      : {filled}/{total} ({100*filled/total:.1f}% non-NaN)")

    print(f"\nAll country codes ({n_teams}):")
    # Show in columns of 10
    for i in range(0, len(unique_codes), 10):
        print("  ", "  ".join(unique_codes[i:i+10]))

    # Teams where ALL years are NaN
    all_nan = (
        df_combined.groupby("country_code")["ranking"]
        .apply(lambda s: s.isna().all())
    )
    all_nan_teams = sorted(all_nan[all_nan].index.tolist())
    if all_nan_teams:
        print(f"\nTeams with ALL years NaN ({len(all_nan_teams)}): {all_nan_teams}")
    else:
        print("\nNo teams with ALL years NaN.")

    # NaN summary per year
    cov = df_combined.groupby("year")["ranking"].apply(lambda s: s.notna().sum())
    print(f"\nRanked teams per year (out of {n_teams}):")
    print(cov.to_string())

    # Confirm status split
    status_counts = df_combined.groupby("status")["ranking"].count() // len(TARGET_YEARS)
    print(f"\nStatus split: {dict(status_counts)}")

    if sorted(unique_codes) == sorted(ALL_TEAMS.keys()):
        print("\n✓ Country codes match ALL_TEAMS exactly.")
    else:
        missing  = set(ALL_TEAMS.keys()) - set(unique_codes)
        extra    = set(unique_codes) - set(ALL_TEAMS.keys())
        print(f"\n✗ Mismatch! Missing: {missing}  Extra: {extra}")


if __name__ == "__main__":
    main()

"""
build_ranking.py
================
Builds a DataFrame indexed by (country_code, year) → ranking for the
48 World Cup 2026 teams, covering years 1993–2026.

Data strategy
─────────────
• Primary  : FIFA country-specific pages (fifa.com/fifa-world-ranking/{CODE})
             Each page embeds historyRanking.rows with one record per year
             containing `finalRank` (year-end rank) for every year on record.
• 2026 exact: The saved HTML snapshot is parsed for the Jan 19, 2026 table
             position, which is used to cross-check / override the country-page
             value for 2026.
• Note      : Year-end ranks differ from the June-15 target dates in DATE_MAP.
             FIFA's direct API is Akamai-protected and not accessible without a
             real browser session, so `finalRank` is the best publicly available
             approximation for 1993-2025. 2026 uses the exact snapshot date.
"""

import re
import time
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

# Pre-calculated: FIFA dateId closest to June 15 each year
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

WC2026_CODES = [
    # CONMEBOL
    "ARG", "BRA", "COL", "ECU", "URU", "PAR",
    # CONCACAF
    "USA", "MEX", "CAN", "PAN", "HON", "CRC", "JAM",
    # UEFA
    "ESP", "FRA", "GER", "POR", "ENG", "NED", "BEL", "ITA",
    "CRO", "DEN", "AUT", "SUI", "SCO", "TUR", "SRB", "ALB",
    # CAF
    "MAR", "SEN", "EGY", "NGA", "RSA", "CIV", "CMR", "COD", "GHA",
    # AFC
    "JPN", "KOR", "IRN", "AUS", "KSA", "UZB", "JOR", "QAT",
    # OFC
    "NZL",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

HTML_PATH = "ranking-historico-html.html"
TARGET_YEARS = set(DATE_MAP.keys())   # 1993-2026


# ──────────────────────────────────────────────
# Step 1 — Parse HTML snapshot (2026 table)
# ──────────────────────────────────────────────

def parse_html_snapshot(path: str) -> dict[str, int]:
    """
    Parse the rendered FIFA ranking table from the saved HTML snapshot.
    Returns {countryCode: rank} for all teams found in the page.
    """
    print(f"Parsing HTML snapshot: {path}")
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()

    # First try: extract from embedded __NEXT_DATA__ full ranking table
    m = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    if m:
        try:
            data = json.loads(m.group(1))
            # Country-specific page has ranking.menRanking.rows  (full table)
            rows = (
                data.get("props", {})
                    .get("pageProps", {})
                    .get("pageData", {})
                    .get("ranking", {})
                    .get("rankings", {})
                    .get("menRanking", {})
                    .get("rows", [])
            )
            if rows:
                result = {}
                for row in rows:
                    code = row.get("countryCode") or row.get("code")
                    rank = row.get("rank") or row.get("rankingPosition")
                    if code and rank:
                        result[str(code).upper()] = int(rank)
                if result:
                    print(f"  HTML __NEXT_DATA__ menRanking: {len(result)} teams")
                    return result
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # Fallback: parse rendered HTML table
    soup = BeautifulSoup(html, "html.parser")
    rankings: dict[str, int] = {}

    rank_spans = soup.find_all("span", class_=re.compile(r"rank-cell_rank__"))
    for span in rank_spans:
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
        match = re.search(r"/fifa-world-ranking/(\w+)\?gender=men", link["href"])
        if match:
            code = match.group(1).upper()
            if code not in rankings:
                rankings[code] = rank

    print(f"  HTML rendered table: {len(rankings)} teams")
    return rankings


# ──────────────────────────────────────────────
# Step 2 — FIFA country-page scraper
# ──────────────────────────────────────────────

def fetch_country_history(
    session: requests.Session,
    code: str,
    retries: int = 2,
) -> tuple[dict[int, int], str | None]:
    """
    Fetch the FIFA country-ranking page for `code` and extract the
    annual ranking history from __NEXT_DATA__.

    Returns:
        ({year: finalRank}, team_name)  — year range: whatever FIFA provides
    """
    url = f"https://www.fifa.com/fifa-world-ranking/{code}?gender=men"
    for attempt in range(retries + 1):
        try:
            r = session.get(url, timeout=20)
            if r.status_code != 200:
                print(f"  [{code}] HTTP {r.status_code}")
                return {}, None

            m = re.search(
                r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
                r.text,
                re.DOTALL,
            )
            if not m:
                print(f"  [{code}] __NEXT_DATA__ not found")
                return {}, None

            data = json.loads(m.group(1))
            page_data = data["props"]["pageProps"]["pageData"]

            # Team name from intro
            team_name: str | None = page_data.get("intro", {}).get("header")

            # historyRanking["1"] = Men, ["2"] = Women
            hist = page_data.get("historyRanking", {})
            mens_hist = hist.get("1", {})
            rows = mens_hist.get("historyTable", {}).get("rows", [])

            year_rank: dict[int, int] = {}
            for row in rows:
                year = row.get("year")
                rank = row.get("finalRank")
                if year and rank:
                    year_rank[int(year)] = int(rank)

            return year_rank, team_name

        except (requests.RequestException, json.JSONDecodeError, KeyError) as exc:
            if attempt < retries:
                print(f"  [{code}] attempt {attempt+1} failed ({exc}), retrying…")
                time.sleep(3)
            else:
                print(f"  [{code}] ERROR: {exc}")
                return {}, None

    return {}, None


# ──────────────────────────────────────────────
# Step 3 — Build the DataFrame
# ──────────────────────────────────────────────

def build_dataframe() -> pd.DataFrame:
    # ── Parse 2026 snapshot from saved HTML ──
    snapshot_2026 = parse_html_snapshot(HTML_PATH)
    print(f"  2026 snapshot: {len(snapshot_2026)} teams found\n")

    # ── Open a persistent session (shares cookies) ──
    session = requests.Session()
    session.headers.update(HEADERS)

    # Warm-up: fetch the ranking index page to get valid cookies
    print("Warming up session (fetching FIFA ranking index)…")
    try:
        session.get("https://www.fifa.com/fifa-world-ranking/men", timeout=15)
    except requests.RequestException as exc:
        print(f"  Warning: warm-up failed ({exc}), continuing anyway")
    time.sleep(2)

    # ── Fetch one team at a time ──
    records = []
    errors = []

    for i, code in enumerate(WC2026_CODES, start=1):
        print(f"[{i:2d}/{len(WC2026_CODES)}] {code} …", end=" ", flush=True)
        year_rank, team_name = fetch_country_history(session, code)
        time.sleep(1.2)  # polite delay

        if not year_rank:
            errors.append(code)
            print("NO DATA")
        else:
            print(f"OK  ({len(year_rank)} years, name='{team_name}')")

        for year in sorted(TARGET_YEARS):
            _, date_str = DATE_MAP[year]

            # For 2026, prefer the exact snapshot rank; fall back to year_rank
            if year == 2026:
                rank = snapshot_2026.get(code) or year_rank.get(year)
                source = "html_snapshot" if snapshot_2026.get(code) else "country_page"
            else:
                rank = year_rank.get(year)
                source = "country_page" if rank is not None else "missing"

            records.append(
                {
                    "country_code": code,
                    "year": year,
                    "ranking": rank,
                    "date_used": date_str,
                    "team_name": team_name or code,
                    "source": source,
                }
            )

    if errors:
        print(f"\nTeams with no data: {errors}")

    df = pd.DataFrame(records)
    df["ranking"] = pd.to_numeric(df["ranking"], errors="coerce")
    df = df.set_index(["country_code", "year"]).sort_index()
    return df


# ──────────────────────────────────────────────
# Step 4 — Save and report
# ──────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("FIFA World Ranking — WC2026 teams  (1993–2026)")
    print("=" * 65)

    df = build_dataframe()

    # ── Save ──
    out_csv = "ranking_mundial_2026.csv"
    out_xlsx = "ranking_mundial_2026.xlsx"
    df.to_csv(out_csv)
    df.to_excel(out_xlsx)
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_xlsx}")

    # ── Summary ──
    print(f"\ndf.shape : {df.shape}")
    total = len(df)
    filled = df["ranking"].notna().sum()
    print(f"Coverage : {filled}/{total} ({100*filled/total:.1f}% non-NaN)")

    # Coverage by year
    cov = df.groupby("year")["ranking"].apply(lambda s: s.notna().sum())
    print(f"\nCoverage per year ({len(WC2026_CODES)} teams):")
    print(cov.to_string())

    print("\ndf.head(20):")
    print(df.head(20).to_string())


if __name__ == "__main__":
    main()

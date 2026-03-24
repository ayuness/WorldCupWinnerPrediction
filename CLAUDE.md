# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Data Science project to predict the FIFA World Cup 2026 champion using Machine Learning (match outcome classification) and Monte Carlo simulation (full tournament simulation).

## Running the Project

There is no build system. The workflow is:

1. **Data collection** — run scripts in `Extras_(.py)/` to scrape and build datasets
2. **Analysis** — open and run Jupyter notebooks interactively

```bash
# Run notebooks
jupyter notebook

# Run individual data pipeline scripts
python3 Extras_\(.py\)/scrape_eliminatorias.py
python3 Extras_\(.py\)/build_ranking.py
python3 Extras_\(.py\)/estandarizar_eliminatorias.py
```

No test suite, linting configuration, or CI/CD exists in this project.

## Architecture & Data Pipeline

```
[Web Scrapers]          [Local HTML]          [CSV Datasets]
scrape_*.py  ──────→  (raw HTML pages) ──→  Data/Eliminatorias/
build_ranking.py                             Data/WorldCupHistory/
                                                      ↓
                                           Dataset.ipynb  (loads & merges data)
                                                      ↓
                                           EDA.ipynb  (exploration & visualization)
                                                      ↓
                                           ModelDataset.ipynb  (feature engineering → master dataset)
                                                      ↓
                                           Data/master_dataset_64.csv  (64 teams × 17 features)
                                                      ↓
                                           [ML Model: XGBoost classifier]
                                                      ↓
                                           [Monte Carlo Simulation]
```

### Key Data Sources

| File | Description |
|------|-------------|
| `Data/WorldCupHistory/matches_1930_2022.csv` | All World Cup match results |
| `Data/WorldCupHistory/worldcups.csv` | World Cup editions and winners (1930–2018) |
| `Data/WorldCupHistory/historial_mundialista.csv` | Per-team World Cup history metrics |
| `Data/Eliminatorias/eliminatorias_conjuntas.csv` | 2026 qualifiers (all 6 confederations combined) |
| `Data/Eliminatorias/eliminatorias_<conf>.csv` | Per-confederation qualifier data (afc, caf, concacaf, conmebol, ofc, uefa) |
| `Data/ranking_mundial_2026_v2.csv` | Historical FIFA rankings 1993–2026 |
| `Data/transfermarkt_selecciones.csv` | Team market valuations (Transfermarkt) |
| `Data/Copa_America.csv` | Copa America historical results |
| `Data/euro_histoia/` | Euro Championship historical data (1960–2024), one CSV per edition |
| `Data/master_dataset_64.csv` | **Master dataset** — 64 teams × 17 features, ready for model training |

### Scripts in `Extras_(.py)/`

- `scrape_eliminatorias.py` — scrapes 2026 qualifiers from all 6 FIFA confederations (AFC, CAF, CONCACAF, CONMEBOL, UEFA, OFC)
- `build_ranking.py` — builds historical FIFA ranking dataset (1993–2026)
- `update_ranking_v2.py` — updates ranking datasets
- `generar_historial_mundialista.py` — generates compact World Cup history per team
- `estandarizar_eliminatorias.py` — normalizes qualifiers data across confederations
- `scrape_transfermarkt.py` — scrapes team market valuations
- `parse_afc.py`, `parse_conmebol.py` — parse region-specific qualifier data

### Notebooks

- `Dataset.ipynb` — loads and merges all data sources into the master dataset
- `EDA.ipynb` — exploratory data analysis with visualizations (matplotlib, seaborn, plotly)
- `ModelDataset.ipynb` — feature engineering pipeline; builds `Data/master_dataset_64.csv` (64 teams × 17 features) from 8 raw sources for XGBoost training

#### Features in `master_dataset_64.csv`

| Group | Features |
|-------|----------|
| FIFA Ranking | `ranking_2026`, `ranking_volatility` |
| World Cup history | `wc_win_pct`, `wc_gc_per_game`, `wc_titles` |
| 2026 Qualifiers | `qual_gf_per_game`, `qual_gc_per_game`, `qual_points_per_game` |
| Market value | `market_value_eur_m`, `squad_avg_age` |
| Continental | `copa_win_pct` (CONMEBOL only), `euro_win_pct` (UEFA only) |
| Performance vs elite | `win_pct_vs_top10` (weighted: WC ×3 recent, ×2 mid, ×1 old; + Euro/Copa) |
| Flags | `wc_debut_flag`, `host_flag`, `is_playoff`, `confederation_strength_index` |

**Key engineering decisions:**
- Debutant teams (9 teams, 0 WC games): `win_pct=0`, `gc/game` = confederation median
- USA/MEX/CAN (hosts, 0 qualifier games): assigned best CONCACAF qualifier values
- Kosovo uses code `KVX` (Transfermarkt mismatch with `KOS`)
- Missing market values filled with confederation minimum

## Tech Stack

- Python 3, Jupyter Notebooks
- `pandas`, `numpy` for data manipulation
- `requests`, `BeautifulSoup` for web scraping
- `matplotlib`, `seaborn`, `plotly` for visualization
- `scikit-learn` for cross-validation and preprocessing
- `xgboost` (next step) for match outcome classification model

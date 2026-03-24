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
                                           [Feature Engineering + ML Model]
                                                      ↓
                                           [Monte Carlo Simulation]
```

### Key Data Sources

| File | Description |
|------|-------------|
| `Data/WorldCupHistory/matches_1930_2022.csv` | All World Cup match results |
| `Data/WorldCupHistory/historial_mundialista.csv` | Per-team World Cup history metrics |
| `Data/Eliminatorias/eliminatorias_conjuntas.csv` | 2026 qualifiers (all 6 confederations combined) |
| `ranking_mundial_2026_v2.csv` | Current FIFA rankings for 2026 participants |
| `transfermarkt_selecciones.csv` | Team market valuations |
| `Copa_America.csv` | Copa America historical results |
| `Data/euro_histoia/` | Euro Championship historical data (1960–2024) |

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

## Tech Stack

- Python 3, Jupyter Notebooks
- `pandas`, `numpy` for data manipulation
- `requests`, `BeautifulSoup` for web scraping
- `matplotlib`, `seaborn`, `plotly` for visualization
- `scikit-learn` (expected) for classification model and cross-validation

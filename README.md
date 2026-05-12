# Prediccion del Campeon del Mundial FIFA 2026

Proyecto de Data Science que predice al campeon del Mundial FIFA 2026 combinando **Machine Learning** (clasificacion multiclase de partidos) y **Simulacion Monte Carlo** (recreacion completa del torneo miles de veces).

El sistema opera en dos niveles: un clasificador estima la probabilidad de victoria, empate o derrota para cada partido individual, y luego una simulacion Monte Carlo recrea el torneo completo — desde la fase de grupos hasta la final — para obtener la probabilidad de campeonato de cada seleccion.

## Resultados

### Prediccion de campeon (consenso de 3 modelos)

| Pos | Seleccion | Prob. Campeon | Momios Reales |
|-----|-----------|--------------|---------------|
| 1 | Espana | 20.16% | 14.44% |
| 2 | Argentina | 14.06% | 8.82% |
| 3 | Francia | 13.66% | 13.23% |
| 4 | Inglaterra | 13.11% | 11.34% |
| 5 | Brasil | 11.49% | 8.82% |
| 6 | Portugal | 4.40% | 7.94% |
| 7 | Paises Bajos | 3.99% | 3.78% |
| 8 | Alemania | 6.20% | 6.11% |
| 9 | Belgica | 2.81% | 2.34% |
| 10 | Marruecos | 1.45% | 1.30% |

### Comparacion de modelos

| Modelo | Accuracy | Log-Loss | Corr. Momios |
|--------|----------|----------|-------------|
| LightGBM | **62.2%** | **0.905** | 0.922 |
| XGBoost | 60.7% | 0.941 | **0.944** |
| Logistic Regression | 55.5% | 0.979 | 0.923 |
| Random Forest | 52.1% | 0.963 | — |

- **LightGBM** obtiene la mejor accuracy y log-loss en clasificacion de partidos
- **XGBoost** tiene la mayor correlacion con las probabilidades del mercado de apuestas (0.944)
- **Logistic Regression** captura ~96% de la senal del XGBoost, demostrando que la mayor parte del aprendizaje es lineal
- **Random Forest** queda rezagado en todas las metricas

## Pipeline

```
[Web Scrapers / CSV]                   [Datos Historicos]
scrape_eliminatorias.py    ──────→    Data/Eliminatorias/
build_ranking.py           ──────→    Data/ranking_mundial_2026_v2.csv
scrape_transfermarkt.py    ──────→    Data/transfermarkt_selecciones.csv
                                              │
                                              ▼
                                    Dataset.ipynb (union de fuentes)
                                              │
                                              ▼
                                    EDA.ipynb (analisis exploratorio)
                                              │
                                              ▼
                                    ModelDataset.ipynb (feature engineering)
                                              │
                                              ▼
                                    master_dataset_48.csv / master_dataset_64
                                              │
                                              ▼
                              ┌────────────────┼────────────────┐
                              │                │                │
                        XGBoost          LightGBM        Random Forest
                        (22 feat)        (58 feat)        (58 feat)
                              │                │                │
                              ▼                ▼                ▼
                         Monte Carlo     Monte Carlo      Monte Carlo
                         (50K sims)      (10K sims)       (pendiente)
                              │                │
                              ▼                ▼
                    Probabilidades de campeonato por seleccion
```

## Datos utilizados

| Fuente | Registros | Uso |
|--------|-----------|-----|
| World Cup matches 1930-2022 | 964 partidos | Entrenamiento principal |
| Euro Championship | 388 partidos | Senal competitiva UEFA |
| Copa America | 212 partidos | Senal competitiva CONMEBOL |
| FIFA Rankings 1993-2026 | 2,176 entradas | Priors anuales de fuerza |
| Transfermarkt | 39,903 jugadores | Valor de mercado y estructura de plantel |
| Eliminatorias 2026 | 6 confederaciones | Forma reciente para inferencia |
| Momios Mundial 2026 | 48 equipos | Benchmark contra el mercado |

## Modelos

### XGBoost (modelo principal)
- **22 features delta** entre equipos: Elo, forma reciente, historial mundialista, ranking FIFA, valor de mercado, microestructura del plantel
- Validacion temporal Leave-One-Tournament-Out (6 folds: 2002-2022)
- 762 arboles, optimizado con Optuna (12 trials)
- **Accuracy: 60.7% | Log-loss: 0.941 | Corr. momios: 0.944**
- Simulacion Monte Carlo: 50,000 torneos

### LightGBM
- **58 features**: team_X, opp_X y delta_X para 19 variables base
- Validacion temporal (3 folds: 2014, 2018, 2022)
- Early stopping, avg 46 iteraciones
- **Accuracy: 62.2% | Log-loss: 0.905 | Corr. momios: 0.922**
- Simulacion Monte Carlo: 10,000 torneos

### Regresion Logistica (Elastic Net)
- **22 features delta** (mismos que XGBoost) + StandardScaler
- Regularizacion ElasticNet (C=0.039, l1_ratio=0.29) + calibracion isotonica post-hoc
- Optimizado con Optuna (100 trials)
- **Accuracy: 55.5% | Log-loss: 0.979 | Corr. momios: 0.923**
- Modelo interpretable: los coeficientes revelan que `delta_rank_strength`, `tournament_weight` y `delta_form_gc` son los drivers principales
- Simulacion Monte Carlo: 50,000 torneos

### Random Forest
- **58 features** (mismas que LightGBM)
- Validacion temporal (3 folds: 2014, 2018, 2022)
- **Accuracy: 52.1% | Log-loss: 0.963**
- Simulacion Monte Carlo pendiente por costo computacional

## Metodologia

### Ingenieria de variables
Cada partido se representa como **diferencias pareadas** entre equipos (delta_X = equipo_A - equipo_B), lo que permite al modelo aprender ventajas relativas:

| Familia | Variables |
|---------|-----------|
| Fuerza dinamica | delta_elo |
| Forma reciente | delta_form_wp, delta_form_gf, delta_form_gc |
| Historial mundialista | delta_wc_win_pct, delta_wc_titles, delta_wc_experience |
| Ranking y mercado | delta_rank_strength, delta_squad_market_value_log |
| Microestructura del plantel | delta_star_concentration, delta_top5_avg_value, delta_gk_value |
| Contexto | is_knockout, tournament_weight |

### Validacion temporal
Esquema **Leave-One-Tournament-Out**: para predecir el Mundial 2014, solo se entrena con partidos anteriores a 2014. Esto evita fuga de informacion del futuro.

### Simulacion Monte Carlo
1. Simulacion de fase de grupos (12 grupos, round-robin)
2. Clasificacion: 2 primeros + 8 mejores terceros
3. Bracket de eliminacion directa hasta la final
4. Para partidos en cancha neutral: promedio de ambas orientaciones (A vs B) y (B vs A)

## Estructura del proyecto

```
├── Data/                          # Datasets crudos y procesados
│   ├── WorldCupHistory/           # Historial de mundiales
│   ├── Eliminatorias/             # Eliminatorias 2026 por confederacion
│   ├── euro_histoia/              # Historial de Eurocopa
│   ├── master_dataset_48.csv      # Dataset maestro (48 equipos)
│   └── momios_mundial2026.csv     # Momios de casas de apuestas
├── Models/                        # Pipelines y artefactos de modelos
│   ├── current_model_pipeline.py  # Pipeline XGBoost
│   ├── logreg_model.py            # Pipeline Logistic Regression
│   ├── compare_models.py          # Comparacion entre modelos
│   └── artifacts/                 # Modelos serializados y metricas
├── sim_Adolfo/                    # Modelos LightGBM y Random Forest
│   ├── lgbm_pipeline.py           # Pipeline LightGBM
│   ├── rf_pipeline.py             # Pipeline Random Forest
│   ├── montecarlo_simulation.py   # Simulacion MC con LightGBM
│   ├── rf_montecarlo_simulation.py
│   └── model_comparison_metrics.json  # JSON con metricas de los 4 modelos
├── Extras_(.py)/                  # Scripts de scraping y ETL
├── Dataset.ipynb                  # Union de fuentes de datos
├── EDA.ipynb                      # Analisis exploratorio
└── ModelDataset.ipynb             # Feature engineering
```

## Tech Stack

- Python 3, Jupyter Notebooks
- **Modelos**: XGBoost, LightGBM, scikit-learn (LogisticRegression, RandomForest)
- **Optimizacion**: Optuna
- **Datos**: pandas, numpy
- **Scraping**: requests, BeautifulSoup
- **Visualizacion**: matplotlib, seaborn, plotly

## Como ejecutar

```bash
# Notebooks interactivos
jupyter notebook

# Scripts de datos
python3 Extras_\(.py\)/scrape_eliminatorias.py
python3 Extras_\(.py\)/build_ranking.py

# Entrenar y comparar modelos XGBoost
python3 Models/compare_models.py

# Pipeline LightGBM + Monte Carlo
python3 sim_Adolfo/lgbm_pipeline.py
python3 sim_Adolfo/montecarlo_simulation.py

# Pipeline Random Forest + Monte Carlo
python3 sim_Adolfo/rf_pipeline.py
python3 sim_Adolfo/rf_montecarlo_simulation.py
```

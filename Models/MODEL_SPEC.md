# World Cup 2026 Model Technical Specification

## 1. Purpose and Scope

This document is the technical specification for the World Cup 2026 prediction system in this repository. It explains:

- what problem the model solves
- which data sources feed the system
- how the feature engineering evolved across model iterations
- how the competing models were evaluated
- why the final winner was selected
- how the winning model was improved after the comparison
- what remains imperfect and how to improve it further

The current source-of-truth implementation is code-first, not notebook-first:

- `current_model_pipeline.py`: final winning pipeline
- `master48_alt_model.py`: alternative `master_dataset_48` experiment
- `compare_models.py`: head-to-head evaluation and simulation comparison

The original notebook workflow in `Baseline_Current_Model.ipynb` is still valuable as historical context, but the final production logic is better represented by the Python pipelines and the CSV artifacts under `Data/`.

## 2. Modeling Objective

The system is not a direct "predict the champion" model. It is a two-stage pipeline:

1. Train a multiclass XGBoost classifier to predict single-match outcome probabilities:
   - class `0`: away win
   - class `1`: draw
   - class `2`: home win
2. Convert those match probabilities into tournament-level champion probabilities through Monte Carlo simulation of the full 2026 tournament structure.

This design matters because champion probabilities are an emergent property of many sampled matches, not a directly supervised label.

## 3. High-Level Architecture

The final system has four layers:

1. Historical match unification
   - World Cup, Euro, and Copa America matches are normalized into one chronological match table.
2. Match-level feature construction
   - each row becomes a pairwise difference between Team A and Team B at the time of the match
3. XGBoost multiclass training
   - trained with temporal folds so future years do not leak into past evaluation
4. 2026 Monte Carlo simulation
   - simulated groups, advancing teams, valid round-of-32 bracket assignment, then knockout rounds

## 4. Data Sources

The project intentionally mixes match-history data, team-level priors, market information, and roster structure.

### 4.1 Core Historical Match Sources

| File | Rows | Role |
| --- | ---: | --- |
| `Data/WorldCupHistory/matches_1930_2022.csv` | 964 | Historical World Cup match results |
| `Data/euro_histoia/*.csv` | 388 after score filtering across 17 files | Historical Euro matches |
| `Data/Copa_America.csv` | 212 | Historical Copa America matches |
| `Data/WorldCupHistory/worldcups.csv` | 21 | Tournament-level World Cup metadata, winners, finalists, attendance |

These sources are unified into one chronologically ordered historical corpus. After filtering and reconstruction in the final pipeline, the winning model uses `1,560` historical match rows.

### 4.2 Team Prior and Ranking Sources

| File | Rows | Role |
| --- | ---: | --- |
| `Data/ranking_mundial_2026_v2.csv` | 2,176 | Annual ranking history by team |
| `Data/master_dataset_64.csv` | 64 | 2026 team prior table used by the main pipeline for 2026 inference features |
| `Data/master_dataset_48.csv` | 48 | Alternative static team prior table used by the `master48` experiment |
| `Data/WorldCupHistory/historial_mundialista.csv` | 64 | Historical World Cup participation summary |

### 4.3 Squad and Roster Sources

| File | Rows | Role |
| --- | ---: | --- |
| `Data/transfermarkt_players.csv` | 39,903 | Player-level yearly roster data, market value, position, age, club context |
| `Data/transfermarkt_selecciones.csv` | repository artifact | squad-level Transfermarkt support data |

### 4.4 2026 Tournament Inputs

| File | Rows | Role |
| --- | ---: | --- |
| `Data/momios_mundial2026.csv` | 48 | 2026 groups plus market-implied champion priors used as an external sanity benchmark |
| `Data/Eliminatorias/eliminatorias_conjuntas.csv` | 64 | Qualifier aggregate performance used in 2026 form construction |

### 4.5 Where the High-Level Signals Came From

The important team-strength signals did not originate inside `master48` itself. They came from raw source tables that were first explored in `EDA.ipynb`, then later packaged or rebuilt in different ways across the model iterations.

Signal origins:

- ranking signals came from `Data/ranking_mundial_2026_v2.csv`
- market-value and squad-age signals came from `Data/transfermarkt_players.csv` and earlier static summary work in `Data/transfermarkt_selecciones.csv`
- qualifier-strength signals came from `Data/Eliminatorias/eliminatorias_conjuntas.csv`
- World Cup history signals came from `Data/WorldCupHistory/historial_mundialista.csv` and `Data/WorldCupHistory/worldcups.csv`

In `EDA.ipynb`, these sources were merged into one exploratory table and inspected through correlations and scatter plots. The exploratory composite-score step used:

- `ranking_2026`
- `market_value_eur_m`
- `win_pct_elim`
- `win_pct_hist`
- `gf_per_game_elim`

That exploratory step helped identify strong signals, but it was not itself a deployed training pipeline.

### 4.6 Clarification on the Three Models

The repository is easiest to understand if the three real model stages are described this way:

1. Model 1: original baseline current model
   - notebook-first temporal match model
   - based on historical match rows and dynamic match features
2. Model 2: `master48` alternative
   - static team-table model using `Data/master_dataset_48.csv`
   - still trained against historical match outcomes
3. Model 3: final temporal-master winner
   - keeps the temporal match backbone of Model 1
   - adds a subset of the useful team-prior signals revealed by Model 2
   - rebuilds those priors from raw yearly data instead of using `master_dataset_48.csv` directly

Important clarification:

- the repository does not currently contain a separate saved or productionized "4-feature model" artifact
- the phrase "4-signal model" only makes sense as informal shorthand for early EDA-driven signal selection, not as a distinct implemented baseline on disk
- therefore the correct comparison is Model 1 versus Model 2, followed by Model 3 as an upgraded successor to Model 1

## 5. Historical Data Unification

The winning pipeline builds one unified historical match table from World Cup, Euro, and Copa America data.

Each unified row contains:

- `date`
- `home_code`
- `away_code`
- `home_score`
- `away_score`
- `tournament`
- `year`
- `stage`
- `tournament_weight`
- derived `outcome`
- derived `is_knockout`

Tournament weights are intentionally asymmetric:

- World Cup: `1.00`
- Euro: `0.85`
- Copa America: `0.80`

These weights affect downstream Elo updates and also enter the XGBoost feature vector as direct context.

The unified corpus is chronologically sorted before any dynamic feature is computed. That guarantees Elo, recent-form windows, and cumulative World Cup statistics are built only from information available prior to each match row.

## 6. Original Current Model

The original notebook model was the first serious baseline and already had the right overall structure:

- multiclass XGBoost
- dynamic Elo
- recent-form features
- World Cup history features
- player-value-derived features
- Monte Carlo simulation

The original notebook summary reported:

- training matches: `1,564`
- features: `16`
- Optuna trials: `200`
- mean log-loss: `1.0053`
- mean accuracy: `0.518`
- Monte Carlo simulations: `50,000`
- correlation with betting priors (`momios`): `0.8496`

The later code extraction and cleanup reconstructed `1,560` rows instead of `1,564`. That small difference comes from stricter normalization and filtering in the script pipeline. It does not change the overall model-selection conclusion.

Original top 5 champion probabilities:

| Team | Champion % |
| --- | ---: |
| BRA | 22.956 |
| ESP | 17.406 |
| ENG | 17.010 |
| ARG | 10.364 |
| GER | 8.402 |

### 6.1 Original Feature Families

The first generation current model relied on four main signal families:

1. Dynamic strength
   - Elo rating difference
2. Short-run form
   - recent win percentage
   - recent goals for
   - recent goals against
3. World Cup history
   - win percentage
   - goals conceded per game
   - titles
   - experience
4. Player-value structure
   - value concentration
   - goalkeeper value
   - depth and age/value interactions

This baseline was meaningful and not random. It clearly beat naive multiclass baselines. But it still had structural issues that needed correction before any model-selection decision could be trusted.

### 6.2 Exact Data Used by Model 1

Model 1 was a historical match model. Its training rows were historical matches, not team rows.

Primary training-row sources:

- `Data/WorldCupHistory/matches_1930_2022.csv`
- `Data/euro_histoia/*.csv`
- `Data/Copa_America.csv`

Supporting sources used to build features or 2026 inference inputs:

- `Data/WorldCupHistory/worldcups.csv`
- `Data/ranking_mundial_2026_v2.csv`
- `Data/transfermarkt_players.csv`
- `Data/WorldCupHistory/historial_mundialista.csv`
- `Data/Eliminatorias/eliminatorias_conjuntas.csv`
- `Data/master_dataset_64.csv`
- `Data/momios_mundial2026.csv`

Model 1 did not train directly from `Data/master_dataset_48.csv`.

### 6.3 Exact Feature Logic in Model 1

Model 1 built one row per historical match and computed feature deltas at match time.

Feature families:

- dynamic strength:
  - Elo before the match
- recent form:
  - recent win percentage
  - recent goals for
  - recent goals against
- World Cup history:
  - pre-match cumulative win percentage
  - goals conceded per game
  - title count
  - experience
- match context:
  - knockout flag
  - tournament weight
- roster microstructure:
  - star concentration
  - top-5 player value
  - goalkeeper value
  - squad depth ratio
  - age-weighted value
  - top-league share

So Model 1 was not a "4-feature model". It was already a broader temporal match model.

## 7. Audit Findings and Correctness Fixes

Before comparing models, the repo was reviewed for methodological and simulation correctness. The most important issues were:

### 7.1 Historical World Cup Carry-Forward Bug

The early World Cup-history logic incorrectly reset cumulative World Cup features for teams that skipped the latest prior World Cup. That meant a historically strong team could lose its existing title count or prior World Cup experience just because it was absent from one edition.

This was fixed by carrying cumulative World Cup state forward across all teams seen so far, not only teams participating in the latest edition.

### 7.2 Neutral-Site Order Dependence

Because the classifier was trained on ordered home/away rows, raw inference on neutral-site 2026 matches initially depended on which team was placed first in the pair.

This was fixed by scoring both orientations:

- `(A, B)`
- `(B, A)`

and averaging them into one symmetric neutral-site probability vector. This eliminated order sensitivity in tournament simulation.

### 7.3 Incorrect Round-of-32 Construction

The early simulator did not implement a valid 48-team knockout structure. In particular, the old bracket logic could create structurally wrong pairings, including third-vs-third matches.

This was replaced with a valid slot-assignment procedure:

- top 2 from each of 12 groups advance
- best 8 third-place teams advance
- third-place slots are assigned through a constrained matching step
- a valid 16-match round of 32 is built with 32 unique teams

### 7.4 Incorrect Best-Third Accounting

The simulator previously miscounted some best third-place qualifiers as group exits because of a type mismatch between team strings and `(team, group)` tuples.

This was fixed so advancement and group-exit accounting reflect the actual advancing teams.

### 7.5 Stale or Synthetic Team Priors

The original model used some features that were either:

- temporally invalid for historical rows
- based on static 2026 information applied backward through time
- or missing for many historical teams and therefore replaced with synthetic defaults

This issue became central in the comparison against the `master48` alternative and directly motivated the final temporal-master improvement.

## 8. Alternative Model: `master_dataset_48`

The `master48` model was designed as a deliberate alternative baseline:

- use `Data/master_dataset_48.csv`
- use all usable columns as features
- train XGBoost on pairwise deltas from that table
- run the same tournament simulation on top

This implementation lives in `master48_alt_model.py`.

### 8.1 Exact Data Used by Model 2

Model 2 was built around `Data/master_dataset_48.csv`.

Primary team-feature source:

- `Data/master_dataset_48.csv`

Historical match labels and row generation still came from:

- `Data/WorldCupHistory/matches_1930_2022.csv`
- `Data/euro_histoia/*.csv`
- `Data/Copa_America.csv`

Additional supporting inputs:

- `Data/ranking_mundial_2026_v2.csv` for country-code alignment and name mapping
- `Data/momios_mundial2026.csv` for 2026 groups and external benchmarking

So Model 2 did not replace historical match labels. It replaced the feature-construction strategy by using a static team table.

### 8.2 How `master48` Builds Features

The alternative model:

- one-hot encodes categorical columns such as `confederation` and `status`
- keeps all remaining numeric columns
- drops only the identifier column `country_code`
- builds match rows as feature deltas `TeamA - TeamB`
- appends `is_knockout` and `tournament_weight`

The underlying team table includes variables such as:

- `ranking_2026`
- `ranking_volatility`
- `wc_win_pct`
- `wc_gc_per_game`
- `wc_titles`
- `qual_gf_per_game`
- `qual_gc_per_game`
- `qual_points_per_game`
- `market_value_eur_m`
- `squad_avg_age`
- `copa_win_pct`
- `euro_win_pct`
- `win_pct_vs_top10`
- `wc_debut_flag`
- `host_flag`
- `is_playoff`
- `confederation_strength_index`

There is overlap in signal types between Model 1 support data and Model 2 packaged priors. For example, ranking, World Cup history, qualifier strength, market value, and age all appear in Model 2. The difference is that Model 2 pulls them from one precomputed static team table instead of rebuilding them dynamically at match time.

### 8.3 Why `master48` Was Useful

The experiment was useful for one important reason: it exposed that annual ranking and squad-strength priors carry real predictive value.

In other words, it discovered useful signal.

### 8.4 Why `master48` Was Not a Valid Final Production Model

Despite being useful, `master48` had two major structural weaknesses.

#### A. Future-Information Leakage

The model loads one 2026 static team table and reuses it to encode historical matches from 2002-2022. That means variables like `ranking_2026` and `market_value_eur_m` are effectively projected backward into historical training rows.

That makes `master48` a good feature-discovery experiment, but a weak methodological choice for final deployment.

#### B. Coverage Collapse

Only teams present in the 48-team table can be used. As a result:

- total historical rows in the main pipeline: `1,560`
- rows retained by `master48`: `689`
- retained ratio: `44.17%`

Retained breakdown inside `master48`:

- World Cup: `479`
- Euro: `133`
- Copa America: `77`

That means `master48` discards more than half the historical corpus.

## 9. How the Models Were Evaluated

The project uses temporal evaluation, not random train/test splitting.

### 9.1 Fold Design

The test folds are the six modern World Cups:

- 2002
- 2006
- 2010
- 2014
- 2018
- 2022

For each fold:

- training uses only matches with `year < test_year`
- testing uses only World Cup matches from `test_year`

This is effectively leave-one-tournament-out temporal validation on modern World Cups.

### 9.2 Metrics

Three metrics are used for match prediction:

- `log_loss`: primary metric
- `accuracy`: secondary classification metric
- multiclass `Brier score`: probability-quality metric

For tournament outputs, one extra benchmark is used:

- Pearson correlation between simulated `champion_pct` and market-implied `momios_pct`

This market correlation is not the training objective and should not override temporal holdout metrics. It is used only as an external reasonableness check on the tournament distribution.

### 9.3 Why Multiple Benchmarks Were Necessary

Directly comparing the original current model with `master48` is not fair unless both are scored on the same match pool.

Therefore the comparison included:

1. `full_coverage`
   - the current model evaluated on all valid historical rows
2. `shared_train_and_test`
   - both models restricted to the same 689-row historical coverage so the benchmark is apples-to-apples

This distinction is critical. A model can appear stronger simply because it avoids difficult matches by filtering them out.

## 10. Intermediate Comparison: Why `master48` Initially Looked Competitive

Before the temporal-master upgrade, the comparison showed a split picture.

From `Data/model_comparison_summary_manual.csv`:

| Model | Benchmark | Train Rows | Test Rows | Mean Log-Loss | Mean Accuracy | Mean Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| current | full_coverage_cleaned | 1,560 | 384 | 1.019022 | 0.531250 | 0.606249 |
| current | shared_test_full_training | 1,560 | 225 | 1.022802 | 0.527145 | 0.608318 |
| current | shared_train_and_test | 689 | 225 | 1.053706 | 0.532820 | 0.628274 |
| master48 | shared_train_and_test | 689 | 225 | 1.003056 | 0.523394 | n/a |

Interpretation:

- on strict shared-subset log-loss, `master48` looked better
- but that gain came from a model using static 2026 priors on historical rows
- therefore the experiment identified valuable signals, but not a clean final architecture

This was the turning point in the project. The correct response was not "ship `master48`." The correct response was "extract the useful priors from `master48`, then reintroduce them in a leakage-safe temporal form."

## 11. Final Improvement: Temporal-Master Upgrade

The winning improvement was implemented in `current_model_pipeline.py`.

The idea was simple:

- keep the current model's temporal validity
- keep its broader historical coverage
- absorb the strongest `master48` signals
- but only in a year-aware, leakage-safe way

### 11.1 Exact Flow from Model 1 to Model 3

The upgrade path from Model 1 to Model 3 was:

1. Start from the Model 1 temporal match backbone
   - same historical match corpus
   - same one-row-per-match training design
   - same Elo, form, World Cup history, and roster-microstructure backbone
2. Train Model 2 (`master48`) as a feature-discovery alternative
   - observe that ranking and squad-strength priors improve predictive quality on the shared subset
3. Do not adopt Model 2 directly
   - because it uses a static 2026-style team table on historical rows
   - because it reduces training coverage from `1,560` to `689` rows
4. Rebuild the useful prior families from raw yearly sources
   - ranking priors rebuilt from `Data/ranking_mundial_2026_v2.csv`
   - squad-value and age priors rebuilt from `Data/transfermarkt_players.csv`
5. Add those priors to the Model 1 temporal backbone
   - producing Model 3, the temporal-master winner

So Model 3 is not an ensemble of Model 1 and Model 2. It is a new single model whose backbone is Model 1 and whose added priors were motivated by Model 2.

### 11.2 Exact Data Used by Model 3

Model 3 uses the same historical match-row backbone as Model 1:

- `Data/WorldCupHistory/matches_1930_2022.csv`
- `Data/euro_histoia/*.csv`
- `Data/Copa_America.csv`

It also uses the same baseline-supporting sources:

- `Data/WorldCupHistory/worldcups.csv`
- `Data/WorldCupHistory/historial_mundialista.csv`
- `Data/ranking_mundial_2026_v2.csv`
- `Data/transfermarkt_players.csv`
- `Data/Eliminatorias/eliminatorias_conjuntas.csv`
- `Data/master_dataset_64.csv`
- `Data/momios_mundial2026.csv`

Model 3 does not train directly from `Data/master_dataset_48.csv`.

### 11.3 New Temporal-Master Priors

The final model adds annual team priors derived from yearly ranking history and yearly squad aggregates:

- `rank_strength`
- `rank_volatility`
- `squad_market_value_log`
- `squad_avg_age`
- `rank_available`
- `squad_available`

#### Rank Strength

Ranking is transformed into a bounded strength score:

`rank_strength = 1 - (ranking - 1) / (max_rank - 1)`

So lower ranking numbers map to higher strength scores.

#### Rank Volatility

For each team, a rolling 4-year standard deviation of ranking is computed. This gives a measure of ranking instability, which can matter when distinguishing stable elites from noisy risers.

#### Squad Market Value and Average Age

From `transfermarkt_players.csv`, yearly squad aggregates are built:

- total squad market value
- average age

Market value is log-transformed with `log1p`.

#### Availability Flags

Instead of silently forcing missing values into the same semantic bucket as true low values, the model also includes:

- `both_rank_available`
- `both_squad_available`

These let XGBoost learn that "missing" is structurally different from "low."

### 11.4 What Changed Versus Model 1

Model 3 keeps all the major Model 1 match features and adds six new prior features:

- `delta_rank_strength`
- `delta_rank_volatility`
- `delta_squad_market_value_log`
- `delta_squad_avg_age`
- `both_rank_available`
- `both_squad_available`

These are the features that most directly represent the influence of Model 2 on the final winner.

### 11.5 Year-Consistent Retrieval

The critical design constraint is that each historical match only sees priors available at or before that year.

The retrieval rule is:

- for a match in year `Y`, use the latest available team-year row with `year <= Y`
- ranking and squad priors are resolved independently
- if a specific prior is unavailable, fill with year-level defaults and set the availability flag to `0`

This is the key methodological difference between the final model and `master48`.

### 11.6 One-Page Comparison of Model 1, Model 2, and Model 3

| Item | Model 1: baseline current | Model 2: `master48` alternative | Model 3: final temporal-master |
| --- | --- | --- | --- |
| Training row type | historical match rows | historical match rows | historical match rows |
| Main feature source | dynamic match-time feature engineering | `Data/master_dataset_48.csv` | Model 1 feature engineering plus yearly ranking/squad priors |
| Uses `Data/master_dataset_48.csv` directly | No | Yes | No |
| Main historical label sources | WC + Euro + Copa match results | WC + Euro + Copa match results | WC + Euro + Copa match results |
| Ranking source | `Data/ranking_mundial_2026_v2.csv` | packaged inside `Data/master_dataset_48.csv` | `Data/ranking_mundial_2026_v2.csv` |
| Squad / market source | `Data/transfermarkt_players.csv` | packaged inside `Data/master_dataset_48.csv` | `Data/transfermarkt_players.csv` |
| Qualifier source | `Data/Eliminatorias/eliminatorias_conjuntas.csv` mainly for 2026 inference | packaged inside `Data/master_dataset_48.csv` | `Data/Eliminatorias/eliminatorias_conjuntas.csv` mainly for 2026 inference |
| World Cup history source | `Data/WorldCupHistory/historial_mundialista.csv` and `worldcups.csv` | packaged inside `Data/master_dataset_48.csv` | `Data/WorldCupHistory/historial_mundialista.csv` and `worldcups.csv` |
| Coverage | `1,560` rows | `689` rows | `1,560` rows |
| Core weakness | missing some strong present-day priors | static priors leak future information into history | none of the major weaknesses of Model 1 or Model 2 remain dominant |

## 12. Final Winning Feature Set

The final model uses 22 match-level features.

### 12.1 Dynamic Team Strength

- `delta_elo`

Elo starts at `1500` and is updated chronologically with `K = 40`, scaled by tournament weight.

### 12.2 Recent Form

- `delta_form_wp`
- `delta_form_gf`
- `delta_form_gc`

Recent form is computed from a rolling 10-match window over the unified historical corpus. If a team has fewer than 3 recent matches, the default is:

- win pct: `0.33`
- goals for: `1.0`
- goals against: `1.0`

### 12.3 World Cup Structural History

- `delta_wc_win_pct`
- `delta_wc_gc_per_game`
- `delta_wc_titles`
- `delta_wc_experience`

These are cumulative pre-tournament World Cup priors carried forward correctly over time.

### 12.4 Match Context

- `is_knockout`
- `tournament_weight`

### 12.5 Temporal-Master Priors

- `delta_rank_strength`
- `delta_rank_volatility`
- `delta_squad_market_value_log`
- `delta_squad_avg_age`
- `both_rank_available`
- `both_squad_available`

### 12.6 Player Microstructure

- `delta_star_concentration`
- `delta_top5_avg_value`
- `delta_gk_value`
- `delta_squad_depth_ratio`
- `delta_age_weighted_value`
- `delta_top_league_pct`

Definitions:

- `star_concentration`: top 3 player value share of total squad value
- `top5_avg_value`: average value of the top 5 players
- `gk_value`: value of the best goalkeeper
- `squad_depth_ratio`: median value divided by mean value
- `age_weighted_value`: value tilted toward younger prime-age players
- `top_league_pct`: share of players marked as belonging to top leagues

## 13. XGBoost Training Design

The model uses `xgboost.XGBClassifier` with:

- objective: `multi:softprob`
- classes: `3`
- eval metric: `mlogloss`
- fixed random seed

### 13.1 Hyperparameter Search

Optuna performs Bayesian hyperparameter search over:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `min_child_weight`
- `subsample`
- `colsample_bytree`
- `reg_alpha`
- `reg_lambda`
- `gamma`

Tuning is restricted to inner temporal folds up to `2014`, so later modern World Cups remain untouched during tuning.

### 13.2 Why Log-Loss Is the Main Metric

The final application is Monte Carlo simulation. That means calibrated probabilities matter more than raw class labels. A model that gets the winner right slightly more often but assigns poor probabilities can still be worse for tournament simulation.

That is why log-loss and Brier score matter at least as much as accuracy.

## 14. 2026 Inference Layer

Once the final model is fitted on all historical rows, a 2026 feature table is built.

### 14.1 2026 Form Construction

For 2026, form is not taken blindly from stale tournament history. It is blended:

- `70%` qualifier aggregate form
- `30%` recent major-tournament form

This is materially better than pretending that an old World Cup or Euro record alone represents current form.

### 14.2 Neutral-Site Symmetry

For every 2026 pair:

- score `A vs B`
- score `B vs A`
- average the implied win/draw/win structure

This produces order-invariant neutral-site probabilities for both:

- group stage matches
- knockout matches

## 15. Monte Carlo Tournament Simulation

The simulator is intentionally separated from the classifier. The classifier estimates single-match probabilities; the simulator transforms them into tournament outcomes.

### 15.1 Group Stage

Each group is simulated round-robin.

Standings use:

- points
- goal difference
- goals scored

### 15.2 Goal Sampling

The simulator samples goals conditionally on the sampled match outcome using simple Poisson-based heuristics.

This goal process is not predicted by XGBoost directly. It exists only to produce plausible:

- group-stage points
- goal difference
- goals scored

for tie-breaking purposes.

### 15.3 Advancement

The simulator advances:

- top 2 from each of the 12 groups
- best 8 third-place teams

### 15.4 Round of 32

The round-of-32 bracket is built through constrained slot assignment so that:

- the bracket contains exactly 16 matches
- the bracket contains exactly 32 unique teams
- there are no invalid third-vs-third pairings

### 15.5 Knockout Draw Resolution

If a knockout match is sampled as a draw, the simulator resolves the winner by renormalizing the non-draw mass between the two teams.

This is a pragmatic approximation rather than an explicit extra-time/penalties model.

## 16. Final Results: Winning Model

The final winning model is the temporal-master upgraded current pipeline.

From `Data/current_temporal_master_summary.csv`:

| Metric | Value |
| --- | ---: |
| Optuna trials | 12 |
| Monte Carlo simulations | 50,000 |
| Mean log-loss | 0.941038 |
| Mean accuracy | 0.606771 |
| Mean Brier score | 0.552750 |
| Correlation vs `momios` champion priors | 0.943708 |

Per-fold results from `Data/current_temporal_master_cv.csv`:

| Test Year | Train Rows | Test Rows | Log-Loss | Accuracy | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2002 | 748 | 64 | 1.025515 | 0.546875 | 0.613879 |
| 2006 | 869 | 64 | 0.787008 | 0.703125 | 0.450470 |
| 2010 | 990 | 64 | 0.983146 | 0.546875 | 0.584602 |
| 2014 | 1,111 | 64 | 0.892406 | 0.640625 | 0.527094 |
| 2018 | 1,284 | 64 | 0.929238 | 0.656250 | 0.550045 |
| 2022 | 1,445 | 64 | 1.028916 | 0.546875 | 0.590413 |

Final top 5 2026 champion probabilities from `Data/simulation_results_current_temporal_master.csv`:

| Team | Champion % |
| --- | ---: |
| ESP | 22.790 |
| ARG | 15.862 |
| BRA | 14.966 |
| FRA | 14.804 |
| ENG | 13.514 |

## 17. Final Head-to-Head: Winner vs `master48`

After the temporal-master upgrade, the current model was rechecked on the exact shared coverage used by `master48`.

From `Data/model_comparison_summary_temporal_master.csv`:

| Model | Benchmark | Train Rows | Test Rows | Mean Log-Loss | Mean Accuracy | Mean Brier | Corr vs Momios |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current_temporal_master | full_coverage | 1,560 | 384 | 0.941038 | 0.606771 | 0.552750 | 0.943708 |
| current_temporal_master | shared_train_and_test | 689 | 225 | 0.955446 | 0.599622 | 0.562886 | n/a |
| master48 | shared_train_and_test | 689 | 225 | 0.992485 | 0.536695 | 0.587786 | 0.847518 |

This is the core model-selection result.

Interpretation:

- on the full historical corpus, the temporal-master current model is clearly strong
- on the strict shared subset, it still beats `master48`
- it also aligns more closely with external 2026 market priors

At that point the competition is effectively over.

## 18. Why the Final Model Is the Best Version Yet

The final version wins because it combines three properties that no earlier version had simultaneously.

### 18.1 It Is Temporally Correct

The final model uses yearly priors with a `latest <= year` rule. It does not train historical 2002 or 2010 matches on 2026 team attributes.

That makes it methodologically defensible.

### 18.2 It Has Broader Historical Coverage

The winning model uses `1,560` historical rows. The `master48` alternative only uses `689`.

More coverage matters because:

- the model sees more tournament structures
- more national-team archetypes are represented
- rare historical patterns are less likely to be discarded

### 18.3 It Internalized the Best Discovery From `master48`

The reason `master48` mattered was not that it should be shipped unchanged. The reason it mattered was that it revealed the value of:

- ranking-derived team strength
- ranking stability
- squad value
- squad age

The final model took those signals, rebuilt them in a leakage-safe annual form, and merged them into the broader temporal pipeline. That is why it improved decisively.

### 18.4 It Dominates Both the Old Baseline and the Alternative

Relative to the original notebook baseline:

- old current log-loss: `1.0053`
- final current log-loss: `0.9410`
- old current accuracy: `0.518`
- final current accuracy: `0.6068`
- old simulation correlation vs `momios`: `0.8496`
- final correlation: `0.9437`

Relative to `master48` on the shared benchmark:

- final current log-loss: `0.9554`
- `master48` log-loss: `0.9925`
- final current accuracy: `0.5996`
- `master48` accuracy: `0.5367`
- final current Brier: `0.5629`
- `master48` Brier: `0.5878`

This is the strongest quantitative argument available in the repository.

## 19. Conclusions

### 19.1 Final Decision

The production winner is:

`current_model_pipeline.py`

specifically the temporal-master upgraded version.

### 19.2 What `master48` Ultimately Was

`master48` was not wasted effort. It served an important role:

- it was a feature-discovery probe
- it highlighted which present-day team priors had the most predictive value
- it forced a fair common-subset benchmark

But it should not be the deployed champion model because it bakes static 2026 information into historical training rows and discards too much history.

### 19.3 What the Project Learned

The winning approach is not "more static priors" and not "more historical dynamics" in isolation. The best version is the hybrid:

- dynamic temporal match history
- cumulative tournament history
- roster microstructure
- annual ranking and squad priors
- strict temporal evaluation
- corrected tournament simulator

## 20. Recommended Next Improvements to the Winner

The current winner is the best version yet, but it is not finished in the absolute sense. The next improvements with the highest expected return are:

### 20.1 Add Probability Calibration

Recommendation:

- fit a post-hoc calibration layer on out-of-fold predictions
- test temperature scaling, multiclass isotonic calibration, or Dirichlet calibration

Why:

- Monte Carlo quality depends on calibrated probabilities more than on hard labels
- the model is already strong enough that calibration is a logical next gain

### 20.2 Improve the Goal Model Used for Group Tie-Breakers

Current state:

- goals are sampled from a simple outcome-conditioned heuristic

Recommendation:

- train a separate goal model or expected-goals proxy
- condition goal distributions on team-strength deltas rather than on win/draw/loss alone

Why:

- group standings depend on points, goal difference, and goals scored
- a better score-generation layer will improve ranking realism even if the win-probability model stays fixed

### 20.3 Expand the Match Corpus

Current training history uses:

- World Cup
- Euro
- Copa America

Recommendation:

- add qualifiers by confederation
- add Nations League / Gold Cup / Asian Cup / AFCON if cleanly available
- optionally add weighted friendlies if data quality is acceptable

Why:

- recent form becomes less sparse
- smaller confederations gain more representative history
- 2026 inference becomes less dependent on defaults

### 20.4 Add Explicit Home/Host/Travel Context for Inference

Recommendation:

- incorporate host advantage for USA, MEX, CAN
- model regional travel asymmetries if a venue schedule becomes available

Why:

- the 2026 tournament is geographically distributed
- host effects are real and currently only partially encoded through static flags

### 20.5 Store Reproducible Run Manifests

Recommendation:

- persist tuned hyperparameters
- persist dataset hashes
- persist the exact fold metrics and seed state for every major run

Why:

- model provenance becomes clearer
- notebook and script outputs stay synchronized
- comparisons become easier to audit later

## 21. Final Statement

This repository now has a clear best model.

It is the temporal-master upgraded current pipeline because it is:

- more correct than the original notebook baseline
- more methodologically defensible than `master48`
- stronger on temporal holdout metrics
- stronger on shared apples-to-apples comparison
- better aligned with external market priors
- backed by a corrected tournament simulator

In practical terms, this is the first version in the repo that simultaneously satisfies:

- predictive quality
- temporal validity
- broad historical coverage
- simulation correctness

That is why it is the best version yet.

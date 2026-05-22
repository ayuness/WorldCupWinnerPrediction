# DEFENSA TECNICA — WorldCupWinnerPrediction

---

## 1. Resumen ejecutivo

Este proyecto predice al campeón de la Copa del Mundo FIFA 2026 mediante un sistema de dos etapas: (1) un clasificador de Machine Learning que estima las probabilidades de resultado para cualquier partido entre dos selecciones (victoria local, empate, victoria visitante), y (2) una simulación de Monte Carlo que usa esas probabilidades para simular el torneo completo miles de veces y generar una distribución empírica del campeón.

El sistema se alimenta de 9 datasets que cubren resultados históricos de mundiales (1930–2022), rankings FIFA (1993–2026), valores de mercado de Transfermarkt, eliminatorias de las 6 confederaciones, y datos de torneos continentales (Copa America, Eurocopa). A partir de estos, se construye un master dataset de 48 equipos clasificados con ~17 features cada uno.

Se entrenan tres modelos — Logistic Regression (baseline), Random Forest (contender) y XGBoost (principal) — y se evalúan con split temporal (train: 1930–2018, test: 2022) para evitar data leakage. La métrica principal es log-loss porque necesitamos probabilidades calibradas, no solo la clase ganadora.

**Decisiones de diseño más defendibles:**
- Split temporal en lugar de random split para respetar la estructura cronológica de los datos.
- Uso de `predict_proba` (no `predict`) porque Monte Carlo necesita distribuciones de probabilidad, no etiquetas.
- Separación estricta entre ML y momios: el modelo se entrena solo con datos históricos; los momios solo entran en la ponderación de Monte Carlo.

---

## 2. Datasets y consistencia

| # | Dataset | Aporte al modelo | Columnas clave para feature engineering |
|---|---------|-------------------|-----------------------------------------|
| 1 | `matches_1930_2022.csv` | Resultados históricos de todos los partidos mundialistas — es la base del set de entrenamiento | `home_team`, `away_team`, `home_score`, `away_score`, `year` |
| 2 | `ranking_mundial_2026_v2.csv` | Ranking FIFA histórico (1993–2026) — señal de fuerza relativa | `country_code`, `ranking`, `date` |
| 3 | `transfermarkt_selecciones.csv` | Valor de mercado y composición del plantel | `market_value_eur_m`, `squad_avg_age`, `squad_size` |
| 4 | `eliminatorias_conmebol.csv` | Performance en clasificatorias CONMEBOL (liga única, 18 jornadas) | `gf_per_game`, `gc_per_game`, `pts`, `pj` |
| 5 | `eliminatorias_uefa.csv` | Performance en clasificatorias UEFA (grupos + playoffs) | Mismo schema estandarizado |
| 6 | `eliminatorias_afc.csv` | Performance en clasificatorias AFC (fases múltiples) | Mismo schema estandarizado |
| 7 | `eliminatorias_caf.csv` | Performance en clasificatorias CAF | Mismo schema estandarizado |
| 8 | `eliminatorias_concacaf.csv` | Performance en clasificatorias CONCACAF | Mismo schema estandarizado |
| 9 | `eliminatorias_ofc.csv` | Performance en clasificatorias OFC | Mismo schema estandarizado |
| — | `worldcups.csv` | Historial de ediciones y campeones (1930–2018) — para features históricas | `winner`, `year`, `host` |
| — | `historial_mundialista.csv` | Métricas compactas por equipo derivadas de los anteriores | `wc_win_pct`, `wc_gc_per_game`, `wc_titles` |

### Consistencia entre datasets

**Country codes unificados:** todos los datasets usan un código ISO-3 estandarizado. Caso especial: Kosovo usa `KVX` (Transfermarkt) vs `KOS` (otras fuentes) — se homologa manualmente.

**Schema de eliminatorias estandarizado** (script `estandarizar_eliminatorias.py`): independientemente de la confederación, cada equipo tiene el mismo vector de columnas:

```
country_code | status | pj | g | e | p | pts | gf | gc | dg | gf_per_game | gc_per_game
```

Esto permite tratar las 6 eliminatorias como una sola tabla, a pesar de que los formatos originales son radicalmente distintos (CONMEBOL: liga de 10 equipos; UEFA: grupos + playoffs; AFC: 4 rondas eliminatorias).

---

## 3. Feature engineering

### 3.1. Filosofia de seleccion

Partimos de ~38 features candidatas extraídas de los 9 datasets. El master dataset tiene $N \approx 600\text{–}800$ observaciones (partidos históricos de mundiales). La regla empírica para evitar overfitting en modelos tabulares es mantener un ratio de al menos 40–50 observaciones por feature:

$$\frac{N}{p} \geq 40 \implies p \leq \frac{800}{40} = 20$$

Con 38 features y ~700 filas, el ratio sería $\approx 18$, insuficiente. Los modelos con alto ratio $p/N$ tienden a memorizar ruido en lugar de aprender señal, especialmente los de alta capacidad como XGBoost. Por eso reducimos de ~38 a 12–20 features finales, buscando un ratio $\geq 40$.

### 3.2. Las 6 categorias de features

**A) Ranking & Strength**

Captura la fuerza relativa oficial de cada selección según FIFA.

- Origen: `ranking_mundial_2026_v2.csv`
- Features: `ranking_2026`, `ranking_volatility`
- Ejemplo de cálculo: `ranking_volatility` mide la desviación estándar del ranking de un equipo en los últimos N años — un equipo con volatilidad alta es impredecible (podría estar en racha ascendente o descendente).

**B) Market Value**

Proxy de la calidad individual de los jugadores, agregada a nivel selección.

- Origen: `transfermarkt_selecciones.csv`
- Features: `market_value_eur_m` (valor total del plantel en millones de euros)
- Ejemplo: Francia con un plantel valorado en ~1,200M EUR vs una selección debutante valorada en ~15M EUR. Esta diferencia de 80x captura algo que el ranking FIFA no: la profundidad y calidad individual del plantel.

**C) Squad Profile**

Características demográficas del plantel.

- Origen: `transfermarkt_selecciones.csv`
- Features: `squad_avg_age`
- Ejemplo: un equipo con edad promedio de 29.5 años tiene experiencia pero menor proyección física que uno de 25.5. La edad promedio captura madurez competitiva.

**D) Qualifying Performance**

Rendimiento reciente en el ciclo eliminatorio 2026 — la señal más "fresca" disponible.

- Origen: `eliminatorias_<conf>.csv` (estandarizadas)
- Features: `qual_gf_per_game`, `qual_gc_per_game`, `qual_points_per_game`
- Ejemplo: Argentina en eliminatorias CONMEBOL con 2.1 goles a favor por partido y 0.5 en contra, vs un equipo de la OFC con 1.0 y 1.8. Las métricas *por partido* permiten comparar entre confederaciones con distinto número de partidos.

**E) Historical World Cup Record**

Performance acumulada en mundiales previos (1930–2022).

- Origen: `historial_mundialista.csv` (derivado de `matches_1930_2022.csv`)
- Features: `wc_win_pct`, `wc_gc_per_game`, `wc_titles`, `wc_debut_flag`
- Ejemplo: Brasil con `wc_titles=5` y `wc_win_pct=0.71` vs un debutante con `wc_titles=0` y `wc_win_pct=0` (imputado). El flag `wc_debut_flag` permite al modelo tratar debutantes como una categoría distinta.

**F) Confederation & Context**

Variables contextuales que no son de rendimiento puro pero afectan resultados.

- Features: `host_flag`, `is_playoff`, `confederation_strength_index`, `copa_win_pct` (solo CONMEBOL), `euro_win_pct` (solo UEFA)
- Ejemplo: `host_flag=1` para USA, MEX y CAN. La ventaja de local está documentada históricamente (los anfitriones ganan ~67% de sus partidos en mundiales). `confederation_strength_index` captura que un equipo clasificado por la UEFA enfrentó competencia distinta que uno de la OFC.

### 3.3. Features derivadas / de interaccion

Cuando el modelo predice el resultado de un partido entre equipo A y equipo B, las features absolutas de cada equipo por separado son menos informativas que la *diferencia* entre ellas. Esto se debe a que el resultado depende de la comparación relativa, no del nivel absoluto.

Features de interacción:
- `ranking_diff = ranking_A - ranking_B` — diferencia de posiciones (menor es mejor para A)
- `market_value_ratio = market_value_A / market_value_B` — ratio de valores de mercado
- `qual_gf_diff = qual_gf_per_game_A - qual_gf_per_game_B` — diferencia ofensiva

**Por que diferencias y no valores absolutos:** un partido entre el equipo #5 y el #8 del ranking tiene una dinámica similar a uno entre #15 y #18 (diferencia de 3 posiciones en ambos casos). Si pasamos rankings absolutos, el modelo necesita aprender esta relación implícitamente, desperdiciando capacidad. Con `ranking_diff`, la señal es directa.

Además, las features de diferencia hacen al modelo **simétrico**: intercambiar local y visitante solo cambia el signo, lo cual es consistente con la física del problema.

### 3.4. Manejo de nulls

**Caso 1: Equipos sin participación previa en mundiales (9 equipos debutantes)**

Son equipos con 0 partidos mundialistas. No se puede calcular `wc_win_pct` ni `wc_gc_per_game`.

Estrategia:
- `wc_win_pct = 0` (no han ganado nunca en un mundial)
- `wc_gc_per_game` = mediana de la confederación del equipo (no la media global, que sesgaría hacia el centro)
- `wc_debut_flag = 1` — flag binario para que el modelo pueda aprender un efecto "debutante" sin depender de la imputación

**Caso 2: Equipos sin valor en Transfermarkt (~30 de los 48 no cubiertos directamente)**

Transfermarkt cubre ~70 selecciones de las ligas principales. Para equipos sin cobertura:
- Imputación con el mínimo de la confederación (no la mediana, porque estos equipos tienden a ser los más débiles de su zona)

**Caso 3: Eliminatorias con esquemas distintos por confederación**

CONMEBOL juega una liga de ida y vuelta (18 partidos), AFC juega 4 rondas (de 4 a 18 partidos según fase), OFC juega mucho menos. La homologación se logra con métricas *por partido* (`gf_per_game`, `gc_per_game`, `points_per_game`), no totales acumulados. Así un equipo con 18 partidos y uno con 8 son comparables.

**Caso 4: Países anfitriones (USA, MEX, CAN) que no jugaron eliminatorias**

Estos tres equipos clasifican automáticamente, por lo que tienen NaN en todas las columnas de eliminatorias (`pj, g, e, p, pts, gf, gc, dg`).

Estrategia:
- Se les asignan los mejores valores de su confederación (CONCACAF), bajo la lógica de que son hosts precisamente porque son potencias regionales
- `host_flag = 1` para codificar su condición especial

**Regla general de imputación:** mediana de la confederación, no media global. La media global tira todo hacia el centro y borra la señal regional. La mediana de confederación preserva el contexto competitivo real del equipo.

### 3.5. Metodos de seleccion final (de ~38 a 12–20)

Tres métodos complementarios:

**A) Analisis de correlacion (filtro previo)**

Se calcula la matriz de correlación de Pearson entre todas las features numéricas. Pares con $|r| > 0.85$ son candidatos a redundancia.

Correlaciones peligrosas conocidas:
- `market_value_total` vs `avg_player_value`: $r \approx 0.95$ — son casi la misma señal. Solución: quedarse con una sola (total, porque captura tamaño del plantel implícitamente).
- `ranking_diff` vs `ranking_ratio`: alta colinealidad — ambas codifican "quién es más fuerte" de forma casi idéntica. Solución: quedarse con `ranking_diff` (más interpretable y lineal).

La multicolinealidad no daña el accuracy de modelos de árboles, pero sí infla la importancia aparente de features (se reparte entre las colineales) y complica la interpretación de coeficientes en Logistic Regression.

**B) SHAP values (post-entrenamiento)**

SHAP (SHapley Additive exPlanations) calcula la contribución marginal de cada feature a la predicción, basándose en la teoría de juegos cooperativos. Se entrena un modelo con todas las features y se calculan los SHAP values globales. Features con bajo SHAP medio se eliminan.

Ventaja sobre feature importance nativa de árboles: SHAP es consistente (si una feature contribuye más a la predicción, su SHAP value es mayor — lo cual, sorprendentemente, la importancia basada en Gini no siempre garantiza).

**C) RFE — Recursive Feature Elimination**

Proceso iterativo:
1. Entrenar el modelo con todas las features
2. Eliminar la feature con menor importancia
3. Re-entrenar y evaluar
4. Repetir hasta encontrar el subset óptimo (evaluado por cross-validation)

```python
# Pseudocodigo RFE
features = todas_las_38
best_score = infinito
while len(features) > min_features:
    model.fit(X_train[features], y_train)
    score = cross_val_score(model, X_train[features], y_train, scoring='neg_log_loss')
    if mean(score) < best_score:
        best_score = mean(score)
        best_features = features.copy()
    worst = feature_con_menor_importancia(model)
    features.remove(worst)
```

---

## 4. Los modelos — construccion, diferencias, pros y contras

### 4.1. Logistic Regression (baseline)

**Construccion:** formulación multinomial (softmax) para clasificar cada partido en 3 clases: $W$ (victoria local), $D$ (empate), $L$ (victoria visitante).

Para cada clase $k \in \{W, D, L\}$, el modelo estima:

$$P(Y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j \in \{W,D,L\}} e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$

donde $\mathbf{w}_k$ es el vector de coeficientes de la clase $k$ y $\mathbf{x}$ es el vector de features del partido.

Se optimiza minimizando el log-loss (cross-entropy) multiclass:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{3} y_{ik} \log(\hat{p}_{ik})$$

donde $y_{ik} = 1$ si la observación $i$ pertenece a la clase $k$, y $\hat{p}_{ik}$ es la probabilidad predicha.

**Por que como baseline:** los coeficientes $\mathbf{w}_k$ son directamente interpretables. Si el coeficiente de `ranking_diff` para la clase $W$ es positivo, significa que a mayor diferencia de ranking a favor del local, mayor probabilidad de victoria local. Esto permite validar que el modelo captura relaciones coherentes antes de pasar a modelos más complejos.

**Preprocesamiento critico — StandardScaler:**

La regresión logística usa la magnitud de los coeficientes en la regularización L2 ($\lambda \sum w_k^2$). Si `ranking_diff` va de -150 a 150 y `market_value_ratio` va de 0.01 a 80, los coeficientes tendrán escalas muy distintas y la regularización los penalizará desproporcionadamente. StandardScaler transforma cada feature a media 0 y varianza 1:

$$x' = \frac{x - \mu}{\sigma}$$

Los modelos de árboles no necesitan esto porque sus splits son comparaciones de umbrales ($x_j \leq t$), invariantes ante transformaciones monótonas.

**Pros:** interpretabilidad total, baja varianza, rápido de entrenar y debuggear.

**Contras:** asume que el logit es una función lineal de las features. Si la relación real es no lineal (e.g., el ranking importa mucho entre top-10 pero poco entre posición 80 y 90), LR no puede capturarlo sin features polinomiales manuales.

### 4.2. Random Forest

**Construccion:** ensemble de $B$ árboles de decisión, cada uno entrenado con:
- Una muestra bootstrap (con reemplazo) del dataset de entrenamiento — esto es *bagging* (Bootstrap AGGregating)
- En cada split de cada árbol, solo un subset aleatorio de $m$ features es considerado ($m \approx \sqrt{p}$ para clasificación)

La predicción final es el promedio de las probabilidades de todos los árboles:

$$\hat{p}_k = \frac{1}{B} \sum_{b=1}^{B} \hat{p}_k^{(b)}$$

**Por que incluirlo:** captura interacciones no lineales sin ingeniería manual. Si la ventaja del ranking solo importa cuando el valor de mercado es alto, Random Forest puede aprender ese condicional automáticamente.

**Hiperparametros clave:**
- `n_estimators`: número de árboles ($B$). Más árboles = menos varianza, pero rendimiento marginal decreciente y mayor costo computacional.
- `max_depth`: profundidad máxima de cada árbol. Controla la complejidad individual.
- `min_samples_leaf`: mínimo de observaciones en una hoja. Previene hojas con una sola observación (overfitting).
- `max_features`: número de features candidatas en cada split. Menor = más diversidad entre árboles = más reducción de varianza.

**Pros:** bajo riesgo de overfitting vs un árbol individual (la varianza del ensemble es $\text{Var}/B$ si los árboles son independientes), da feature importance, no requiere escalado, robusto a outliers.

**Contras:** menos interpretable que LR, los árboles individuales no se corrigen entre sí (a diferencia de boosting), puede ser superado por gradient boosting en datasets tabulares.

### 4.3. XGBoost (modelo principal — desarrollo a fondo)

**Construccion:** XGBoost implementa gradient boosting sobre árboles de decisión. A diferencia de Random Forest (donde los árboles se entrenan en paralelo e independientemente), en boosting los árboles se entrenan **secuencialmente**: cada nuevo árbol $f_t$ se ajusta para corregir los errores residuales del ensemble acumulado hasta el paso $t-1$.

El ensemble en el paso $t$ es:

$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta \cdot f_t(\mathbf{x}_i)$$

donde $\eta$ es el learning rate que controla cuánto "peso" se da a cada nuevo árbol.

**Funcion objetivo de XGBoost:**

Esta es la diferencia matemática clave vs gradient boosting clásico. XGBoost optimiza:

$$\text{Obj}^{(t)} = \sum_{i=1}^{N} L(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)) + \Omega(f_t)$$

donde $\Omega(f_t)$ es el término de regularización sobre la estructura del árbol:

$$\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

- $T$: número de hojas del árbol $f_t$
- $w_j$: peso (score) de la hoja $j$
- $\gamma$: penalización por cada hoja adicional (controla la complejidad estructural)
- $\lambda$: penalización L2 sobre los pesos de las hojas (suaviza las predicciones)

**Aproximacion de segundo orden:**

GBM clásico usa solo el gradiente de primer orden de la pérdida. XGBoost realiza una expansión de Taylor de segundo orden:

$$L(y_i, \hat{y}_i^{(t-1)} + f_t) \approx L(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i)$$

donde:
- $g_i = \frac{\partial L}{\partial \hat{y}_i^{(t-1)}}$ — gradiente (primera derivada)
- $h_i = \frac{\partial^2 L}{\partial (\hat{y}_i^{(t-1)})^2}$ — Hessiano (segunda derivada)

Al incluir el Hessiano, XGBoost tiene información sobre la *curvatura* de la pérdida, no solo la pendiente. Esto le permite dar pasos más grandes donde la pérdida es suave (Hessiano bajo) y pasos más cautelosos donde cambia rápido (Hessiano alto). Resultado: convergencia más rápida y más robusta que GBM de primer orden.

El peso óptimo de cada hoja se calcula analíticamente:

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

Y la ganancia de un split se evalúa como:

$$\text{Gain} = \frac{1}{2}\left[\frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I} h_i + \lambda}\right] - \gamma$$

El término $-\gamma$ al final actúa como poda: si la ganancia del mejor split posible es menor que $\gamma$, el árbol no crece esa rama. Esto es regularización estructural integrada en el proceso de construcción del árbol.

**Hiperparametros de la version principal:**

| Hiperparámetro | Rol |
|----------------|-----|
| `objective='multi:softprob'` | Produce probabilidades para las 3 clases, no solo la clase argmax |
| `n_estimators` + early stopping | Número máximo de árboles; early stopping detiene si la métrica de validación no mejora en $k$ rondas |
| `learning_rate` ($\eta$) | Shrinkage: valores bajos (0.01–0.1) requieren más árboles pero generalizan mejor |
| `max_depth` | Profundidad máxima de cada árbol (típicamente 3–6 para datasets chicos) |
| `min_child_weight` | Suma mínima de Hessianos en una hoja — análogo a `min_samples_leaf` pero ponderado por la curvatura de la pérdida |
| `subsample` | Fracción de filas usadas para entrenar cada árbol (e.g., 0.8). Introduce aleatoriedad tipo bagging *dentro* del boosting |
| `colsample_bytree` | Fracción de features usadas para cada árbol. Misma idea que `max_features` en RF |
| `reg_alpha` | Regularización L1 sobre los pesos de las hojas: $\alpha \sum |w_j|$. Fomenta sparsity |
| `reg_lambda` | Regularización L2 sobre los pesos de las hojas: $\frac{1}{2}\lambda \sum w_j^2$. Suaviza |

**Por que `predict_proba` y no `predict`:**

Monte Carlo necesita las probabilidades de las 3 clases para muestrear resultados. Si el modelo dice $P(W)=0.45$, $P(D)=0.30$, $P(L)=0.25$, entonces en cada simulación se lanza un "dado de 3 caras" con esas probabilidades. Usar `predict` (que devuelve solo el argmax, en este caso $W$) haría que Monte Carlo simule siempre victoria del mismo equipo, eliminando la incertidumbre que justifica toda la simulación.

**Manejo nativo de NaN:** XGBoost no necesita imputación explícita. Para cada split, aprende automáticamente si los NaN deben ir al hijo izquierdo o derecho (elige la dirección que maximiza la ganancia). Esto es superior a la imputación porque no introduce señal artificial.

**Pros:** estado del arte para datos tabulares, regularización integrada en la función objetivo, captura interacciones de alto orden, manejo nativo de NaN, produce probabilidades calibrables.

**Contras:** más hiperparámetros que tunear, riesgo de overfitting si se usa sin early stopping ni regularización, menor interpretabilidad (mitigado con SHAP).

### 4.4. Tabla comparativa

| Criterio | Logistic Regression | Random Forest | XGBoost |
|----------|-------------------|---------------|---------|
| **Tipo** | Modelo lineal generalizado | Ensemble de árboles (bagging) | Ensemble de árboles (boosting) |
| **Interpretabilidad** | Alta (coeficientes directos) | Media (feature importance) | Baja-Media (SHAP) |
| **Manejo de nulls** | Requiere imputación | Requiere imputación | Nativo (aprende dirección de split) |
| **Requiere escalado** | Si (StandardScaler) | No | No |
| **Captura no linealidad** | No (sin features manuales) | Si | Si |
| **Riesgo de overfitting** | Bajo | Bajo-Medio | Medio-Alto (sin regularización) |
| **Costo computacional** | Bajo | Medio | Medio |
| **Rol en el proyecto** | Baseline | Contender | Principal |

---

## 5. Evaluacion y minimizacion del error

### 5.1. Division de datos

**Split temporal:** entrenamiento con partidos de mundiales 1930–2018, test con partidos del mundial 2022.

**Por que temporal y no random:**

En un split aleatorio, un partido de 2022 podría caer en train y uno de 2018 en test. Esto crea data leakage temporal: el modelo "vería el futuro" durante el entrenamiento. El desempeño reportado en test sería artificialmente alto porque el modelo habría aprendido patrones de la misma época que está prediciendo.

El split temporal respeta la causalidad: el modelo solo aprende del pasado para predecir el futuro, exactamente como operará en producción (prediciendo el mundial 2026 con datos hasta 2022).

### 5.2. Cross-validation

Se usa **StratifiedKFold** (e.g., 5 folds) dentro del set de entrenamiento (1930–2018).

**Por que Stratified:** el empate es la clase minoritaria en partidos de mundiales (~25% vs ~40% victoria local y ~35% victoria visitante). Un KFold simple podría generar folds donde un fold tiene muy pocos empates, distorsionando la evaluación. StratifiedKFold garantiza que cada fold mantenga la proporción original $W/D/L$.

Cross-validation se usa para:
1. Estimar el desempeño esperado del modelo sin tocar el test set
2. Comparar modelos (LR vs RF vs XGB) de forma justa
3. Tunear hiperparámetros sin contaminar el test

### 5.3. Metricas y por que cada una

**Log-loss (metrica principal):**

$$\text{Log-loss} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{3} y_{ik} \log(\hat{p}_{ik})$$

Penaliza no solo si aciertas la clase, sino *cuánta confianza* le asignas. Predecir $\hat{p}(W)=0.51$ y acertar es mucho mejor que predecir $\hat{p}(W)=0.99$ y fallar (la penalización es $-\log(0.01) = 4.6$ vs $-\log(0.51) = 0.67$).

Referencia teórica: un modelo que asigna probabilidad uniforme $1/3$ a las 3 clases tiene $\text{log-loss} = \ln(3) \approx 1.0986$. Cualquier modelo que supere este umbral está aprendiendo señal; valores por debajo son mejores. Los valores concretos de los modelos entrenados están en los notebooks de evaluación.

**Por que log-loss y no accuracy:** necesitamos probabilidades calibradas para Monte Carlo. Un modelo con 55% de accuracy podría ser terrible si dice $\hat{p}(W)=0.90$ para todos los partidos (overconfident). Log-loss detecta esa falla; accuracy no.

**Accuracy:** se reporta como métrica complementaria intuitiva, pero no es la que guía las decisiones. Con clases desbalanceadas, un modelo que siempre predice "victoria local" tendría ~40% de accuracy sin haber aprendido nada.

**Brier score multiclass:**

$$\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{3} (\hat{p}_{ik} - y_{ik})^2$$

Mide el error cuadrático medio entre probabilidades predichas y las clases reales (codificadas como 0/1). Es complementario a log-loss: ambos miden calibración, pero Brier es menos sensible a predicciones extremas (no tiene el $\log$ que explota cerca de 0).

**Matriz de confusion:** permite diagnóstico cualitativo. Si el modelo confunde sistemáticamente empates con victorias, podría ser necesario ajustar el umbral o agregar features que diferencien partidos cerrados.

### 5.4. Minimizacion del error — que hacemos

**Hyperparameter tuning:**

Se puede usar Optuna o GridSearchCV para buscar la mejor combinación de hiperparámetros, optimizando log-loss en cross-validation.

Optuna usa Tree-Structured Parzen Estimator (TPE): en lugar de probar una grilla exhaustiva, modela la distribución de hiperparámetros que producen buenos resultados y muestrea de ahí. Más eficiente que grid search, especialmente con muchos hiperparámetros (XGBoost tiene 7+).

```python
# Pseudocodigo Optuna
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        # ... otros hiperparametros
    }
    model = XGBClassifier(**params)
    score = cross_val_score(model, X_train, y_train, scoring='neg_log_loss', cv=5)
    return -score.mean()  # Optuna minimiza

study = optuna.create_study()
study.optimize(objective, n_trials=100)
```

**Early stopping en XGBoost:** se entrena con un set de validación (dentro del train). Si el log-loss en validación no mejora en $k$ rondas consecutivas, se detiene. Esto evita overfitting sin necesidad de fijar `n_estimators` manualmente.

**Regularizacion en XGBoost:**
- $\gamma$: penaliza cada hoja adicional → árboles más simples
- $\lambda$ (L2): suaviza los pesos de las hojas → predicciones menos extremas
- $\alpha$ (L1): empuja pesos hacia cero → sparsity
- `min_child_weight`: exige masa mínima de Hessiano en cada hoja → evita hojas basadas en pocas observaciones

**Feature selection previa:** menos features implica un espacio de hipótesis más chico, lo que reduce la varianza del modelo (sesgo-varianza tradeoff). Con $N$ chico, preferimos modelos de baja varianza aunque tengan un poco más de sesgo.

### 5.5. Por que los momios NO entran al modelo ML

Los momios de casas de apuestas (betting odds) **no** son features del modelo de ML. Solo entran en la etapa de Monte Carlo como ponderación.

**Razon 1 — Data leakage:** los momios del mundial 2026 incorporan información contemporánea (lesiones, forma reciente, noticias) que el modelo ML no debería ver si queremos validar honestamente con el split temporal. Meter momios al ML haría que el modelo "hiciera trampa" — no estaríamos evaluando si nuestras features predicen, sino si las casas de apuestas predicen (spoiler: sí).

**Razon 2 — Hueco historico:** no existen momios estandarizados para partidos de 1930–2018 en formato comparable. Solo tenemos momios para el mundial 2026. No se puede entrenar un modelo con una feature que solo existe para el test set.

---

## 6. Monte Carlo (resumen breve)

**Conexion con el ML:**

1. Se construyen todas las ~1,128 combinaciones posibles de enfrentamientos entre los 48 equipos clasificados ($\binom{48}{2} = 1{,}128$)
2. Se llama a `predict_proba` una sola vez para generar una tabla de lookup: para cada par $(A, B)$, las probabilidades $[P(W), P(D), P(L)]$
3. Opcionalmente, estas probabilidades se ponderan con los momios de apuestas para incorporar información de mercado

**Simulacion del torneo:**

Cada iteración de Monte Carlo simula el torneo completo:
- Fase de grupos: para cada partido, se muestrea un resultado según las probabilidades del lookup
- Fase eliminatoria: se avanza al ganador; en caso de empate, se resuelve con penaltis (probabilidad uniforme o ajustada)
- Se registra el campeón de esa iteración

Tras miles de iteraciones, se obtiene la distribución empírica: "Argentina ganó en el 18% de las simulaciones, Francia en el 15%, Brasil en el 12%...". Esta distribución es el producto final del proyecto.

**El modelo ML no se re-invoca:** la tabla de lookup se genera una vez. Cada simulación solo consulta esa tabla, lo que hace factible correr decenas de miles de iteraciones en segundos.

---

## 7. Q&A anticipado — preguntas dificiles del profesor

**P1: ¿Por que no usaste una red neuronal?**

Las redes neuronales necesitan miles a millones de observaciones para aprender representaciones útiles. Con ~700 filas, una red neuronal overfittearía severamente: memorizaría los datos de entrenamiento sin generalizar. Los modelos de gradient boosting (XGBoost, LightGBM) son consistentemente superiores a redes neuronales en datos tabulares con $N < 10{,}000$, como lo demuestran benchmarks recientes (Grinsztajn et al., 2022). Además, las features son todas numéricas/categóricas estructuradas — no hay texto, imágenes ni secuencias que justifiquen la capacidad representacional de una red.

**P2: ¿Tu dataset de 600–800 filas no es muy pequeño para XGBoost?**

XGBoost funciona bien con datasets chicos precisamente por su regularización integrada ($\gamma$, $\lambda$, `min_child_weight`). Con early stopping, learning rate bajo y profundidad limitada (`max_depth` 3–5), cada árbol es un modelo débil y simple. El riesgo real con datasets chicos es sobreoptimizar hiperparámetros, lo cual mitigamos con cross-validation y un test set intocable (2022).

**P3: ¿Como evitas data leakage entre entrenamiento y test?**

Tres mecanismos: (1) split estrictamente temporal (1930–2018 train / 2022 test) — ningún dato del futuro contamina el entrenamiento; (2) las features se calculan solo con información disponible *antes* del partido; (3) los momios no entran al modelo ML, solo a Monte Carlo. El StandardScaler se fitea solo en train y se transforma sobre test, para no filtrar media/varianza del futuro.

**P4: ¿Por que log-loss y no accuracy?**

Porque el output del modelo alimenta Monte Carlo, que necesita probabilidades calibradas, no etiquetas. Un modelo con 55% de accuracy pero overconfident ($\hat{p}=0.95$ para cada predicción) sería desastroso para Monte Carlo: produciría simulaciones irrealmente deterministas. Log-loss penaliza la descalibración, accuracy no. Además, con 3 clases desbalanceadas, accuracy es engañosa — un modelo trivial que siempre predice "victoria local" tendría ~40% de accuracy.

**P5: ¿Que pasa con los equipos que nunca han ido al mundial — como los predices?**

Se imputan sus features históricas con la mediana de su confederación y se marca `wc_debut_flag=1`. El modelo aprende el "efecto debutante" como una señal en sí misma. Además, estos equipos sí tienen datos de eliminatorias, ranking FIFA y valor de mercado — no están completamente ciegos. Las features de eliminatorias y ranking son las más informativas para equipos sin historial mundialista.

**P6: Si el modelo nunca vio al equipo X jugar contra Y, ¿como predice ese partido?**

El modelo no memoriza enfrentamientos específicos. Aprende relaciones entre features: "cuando la diferencia de ranking es X, la diferencia de valor de mercado es Y, y el historial mundialista es Z, la probabilidad de victoria local es P". Para un enfrentamiento nuevo, calcula las features de ambos equipos, computa las diferencias/ratios, y aplica esas relaciones aprendidas. Es generalización, no memorización.

**P7: ¿No es trampa usar momios de casas de apuestas?**

No, porque los momios NO entran al modelo ML. El modelo se entrena y evalúa exclusivamente con datos históricos. Los momios solo se usan en la etapa de Monte Carlo como ponderación opcional, y eso es transparente. Además, si usáramos momios como feature de entrenamiento, tendríamos el problema de que no existen momios históricos comparables para partidos de 1930–2018.

**P8: ¿Por que tres modelos y no solo el mejor?**

Tres razones: (1) Logistic Regression como baseline verifica que las features tienen señal — si LR no supera predicción uniforme, hay un problema en el feature engineering, no en el modelo. (2) Comparar LR (lineal) vs RF/XGB (no lineal) cuantifica cuánta no linealidad hay en los datos. (3) Si los tres modelos coinciden en las predicciones, hay más confianza en los resultados. Si divergen, eso informa sobre la incertidumbre.

**P9: ¿Como manejas la dependencia temporal? Un partido de 1950 no es comparable a uno de 2022.**

Este es un punto válido. El fútbol de 1950 era otro deporte. Hay dos mitigaciones: (1) las features son *relativas* al contexto de cada partido (ranking en ese momento, historial acumulado hasta esa fecha), no absolutas; (2) se pueden asignar pesos temporales (más peso a partidos recientes) — la feature `win_pct_vs_top10` ya implementa esto con ponderación ×3 para mundiales recientes, ×2 para intermedios, ×1 para antiguos. Una limitación honesta es que la definición de "equipo fuerte" ha cambiado con el tiempo.

**P10: ¿Que tan calibradas estan tus probabilidades?**

La calibración se puede verificar con un reliability diagram (calibration plot): si el modelo dice $P(W)=0.4$ para un grupo de partidos, debería haber ~40% de victorias locales en ese grupo. Log-loss como métrica principal incentiva la calibración, pero no la garantiza. Si se detecta descalibración, se puede aplicar Platt scaling o isotonic regression post-entrenamiento. Los valores concretos de calibración están en los notebooks de evaluación.

**P11: ¿Que pasa si XGBoost predice un empate con probabilidad 0.4 — que hace Monte Carlo con eso?**

Monte Carlo recibe el vector completo, por ejemplo $[P(W)=0.35, P(D)=0.40, P(L)=0.25]$. En cada simulación, genera un número aleatorio $u \sim U(0,1)$ y asigna el resultado según las probabilidades acumuladas: si $u < 0.35 \to W$, si $u < 0.75 \to D$, si $u < 1.0 \to L$. Así, en ~40% de las simulaciones ese partido termina en empate, en ~35% gana el local, y en ~25% gana el visitante. La incertidumbre del modelo se propaga naturalmente a la simulación.

**P12: ¿Como decides cuantas iteraciones de Monte Carlo correr?**

Se monitorea la convergencia de la distribución empírica. Si tras 5,000 simulaciones la probabilidad estimada del campeón cambia menos de 0.1% al agregar 1,000 más, la simulación ha convergido. Formalmente, el error estándar de una proporción estimada con $N$ simulaciones es $\sqrt{p(1-p)/N}$. Para $p=0.15$ (15% de probabilidad de ser campeón) y $N=10{,}000$, el error estándar es $\approx 0.36\%$, lo cual es suficiente para nuestros fines.

**P13: ¿Que pasaria si entrenaras con datos de todos los partidos internacionales (amistosos, eliminatorias, torneos continentales) y no solo mundiales?**

Sería una extensión válida que aumentaría $N$ de ~700 a ~15,000+. La ventaja: más datos, más señal, el modelo podría aprender patrones más finos. La desventaja: los amistosos son partidos de baja intensidad donde los equipos experimentan con alineaciones — mezclar amistosos con partidos de mundial introduce ruido. Si se hiciera, habría que ponderar los partidos por importancia (mundial ×3, eliminatorias ×2, amistoso ×1) para que los partidos competitivos dominen el aprendizaje.

**P14: ¿Tu modelo captura el efecto "equipo revelacion" (dark horse)?**

Parcialmente. El flag `wc_debut_flag` y las features de eliminatorias capturan si un equipo llega en forma pero sin historial. Sin embargo, los fenómenos de dark horse suelen depender de factores intangibles (cohesión grupal, momento psicológico, sorteo favorable) que no están en ningún dataset cuantitativo. Esta es una limitación inherente de cualquier modelo de predicción deportiva.

**P15: ¿Que validacion hiciste sobre la simulacion de Monte Carlo en si misma (no solo el modelo ML)?**

Se puede validar retroactivamente: simular el mundial 2022 usando el modelo entrenado con datos 1930–2018 y comparar la distribución resultante con lo que realmente pasó. Si el campeón real (Argentina) aparece consistentemente en el top-5 de la distribución simulada, la simulación es coherente. Esto no prueba que el modelo "acertó" — prueba que asignó probabilidad no trivial al resultado real, que es lo mejor que puede hacer un modelo probabilístico.

---

## 8. Cheatsheet final

### Las 3 decisiones de diseno mas defendibles

1. **Split temporal** (1930–2018 train / 2022 test): respeta la causalidad y evita data leakage. Es la única forma honesta de evaluar un modelo que predice eventos futuros.

2. **Log-loss como métrica principal** + `predict_proba`: alinea la optimización del modelo con su uso real. Necesitamos probabilidades calibradas para Monte Carlo, no etiquetas. Optimizar accuracy produciría un modelo "seguro" que siempre predice al favorito con alta confianza.

3. **Separación ML / momios**: el modelo se entrena y evalúa sin momios (honestidad experimental). Los momios solo entran en Monte Carlo como información complementaria del mercado, no como sustituto del modelo.

### Las 3 limitaciones honestas (admitir antes de que las saquen)

1. **Dataset chico** (~700 filas): limita la complejidad de los modelos que podemos usar y la confianza estadística de las métricas. Un intervalo de confianza del 95% para accuracy con $N=64$ (partidos del mundial 2022) tiene un ancho de $\pm 12\%$.

2. **Features estáticas**: el modelo no captura dinámica de partido (lesiones de última hora, condiciones climáticas, efecto del sorteo de grupos). Es una foto fija del estado de cada selección antes del torneo.

3. **Sesgo histórico**: entrenar con partidos desde 1930 asume que los patrones del fútbol pasado aplican al futuro. La globalización del fútbol, el VAR, el cambio a 48 equipos y la evolución táctica son rupturas que el modelo no puede anticipar.

### Las 3 formulas clave

**Softmax (prediccion de probabilidades por clase):**

$$P(Y = k \mid \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \quad z_k = \mathbf{w}_k^T \mathbf{x} + b_k$$

**Log-loss multiclass (funcion de costo principal):**

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik} \log(\hat{p}_{ik})$$

Referencia: predicción uniforme $\to \ln(3) \approx 1.0986$. Menor es mejor.

**Objetivo regularizado de XGBoost:**

$$\text{Obj}^{(t)} = \sum_{i=1}^{N} L(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)) + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

Con peso óptimo de hoja: $w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$, donde $g_i$ es el gradiente y $h_i$ el Hessiano de la pérdida.

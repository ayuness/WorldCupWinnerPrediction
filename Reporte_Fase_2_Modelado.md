# Reporte de la Fase 2: Modelado y Simulación del Mundial FIFA 2026

## 1. Introducción

La segunda fase del proyecto se concentró en el corazón predictivo del sistema. Mientras que la primera etapa estuvo más orientada a la exploración de datos, la limpieza y la construcción del pipeline base, en esta fase el trabajo principal consistió en diseñar, comparar y validar modelos capaces de estimar resultados de partidos y convertir esas probabilidades en escenarios completos de torneo.

El objetivo central no fue entrenar un modelo que predijera directamente al campeón, sino construir un sistema de dos niveles. En el primer nivel, un clasificador multiclase estima la probabilidad de victoria local, empate o victoria visitante para un partido individual. En el segundo nivel, esas probabilidades alimentan una simulación Monte Carlo que recrea el Mundial 2026 decenas de miles de veces, permitiendo estimar probabilidades de campeonato para cada selección.

Esta fase también sirvió para depurar errores metodológicos detectados en versiones previas y para decidir cuál arquitectura era la más sólida entre varias alternativas. Por eso, además de mejorar métricas, la Fase 2 estuvo enfocada en garantizar consistencia temporal, cobertura histórica amplia y un simulador de torneo estructuralmente válido.

## 2. Objetivo de la fase

El propósito específico de esta etapa fue responder cuatro preguntas:

1. ¿Qué variables describen mejor la fortaleza relativa entre dos selecciones en un momento determinado?
2. ¿Qué arquitectura produce probabilidades de partido mejor calibradas para luego simular un torneo completo?
3. ¿Cómo validar el desempeño sin introducir fuga de información del futuro hacia el pasado?
4. ¿Qué selección aparece como favorita una vez que el modelo se integra con la simulación del formato 2026?

En términos prácticos, la Fase 2 transformó el proyecto desde una idea analítica en un sistema predictivo completo y evaluable.

## 3. Datos utilizados para el modelado

El entrenamiento final se apoyó en un corpus histórico unificado de `1,560` partidos, construido cronológicamente a partir de competiciones comparables de selecciones nacionales:

| Fuente | Filas | Rol en el modelo |
| --- | ---: | --- |
| World Cup matches (1930-2022) | 964 | Resultados históricos de mundiales |
| Euro matches | 388 | Señal competitiva adicional para selecciones UEFA |
| Copa América matches | 212 | Señal competitiva adicional para selecciones CONMEBOL |
| FIFA Rankings (1993-2026) | 2,176 | Priors anuales de fuerza y volatilidad |
| Transfermarkt players | 39,903 | Valor, edad y microestructura de plantillas |
| Master dataset (64 equipos) | 64 | Priors de inferencia para 2026 |
| Momios del Mundial 2026 | 48 | Benchmark externo contra mercado |
| Eliminatorias conjuntas | 64 | Forma reciente para la inferencia 2026 |

La decisión de mezclar resultados históricos, rankings, estructura de plantillas y desempeño en eliminatorias permitió modelar tanto la fortaleza acumulada como el estado competitivo más reciente de cada selección.

## 4. Arquitectura del sistema

La arquitectura final quedó organizada en cuatro bloques:

1. Unificación histórica de datos. Se normalizaron partidos de Mundial, Euro y Copa América en una sola tabla cronológica.
2. Ingeniería de variables. Cada fila representa un partido y cada feature se define como una diferencia entre el equipo A y el equipo B.
3. Clasificación multiclase con XGBoost. El modelo estima probabilidades de `win/draw/loss` bajo validación temporal estricta.
4. Simulación Monte Carlo. Las probabilidades predichas se convierten en miles de torneos simulados hasta obtener porcentajes de campeonato.

Además, se asignaron pesos por torneo para reflejar que no todas las competencias aportan la misma señal:

- Mundial: `1.00`
- Euro: `0.85`
- Copa América: `0.80`

## 5. Ingeniería de variables

La versión final del modelo utiliza `22` variables. Más que usar atributos aislados, el enfoque fue construir diferencias pareadas entre selecciones para que el clasificador aprenda ventajas relativas de un equipo sobre otro.

Las variables pueden agruparse en seis familias:

| Familia | Variables representativas | Intención |
| --- | --- | --- |
| Fuerza dinámica | `delta_elo` | Capturar fortaleza competitiva actualizada cronológicamente |
| Forma reciente | `delta_form_wp`, `delta_form_gf`, `delta_form_gc` | Resumir rendimiento reciente en una ventana de 10 partidos |
| Historial mundialista | `delta_wc_win_pct`, `delta_wc_titles`, `delta_wc_experience` | Incorporar experiencia y tradición en Copa del Mundo |
| Priors temporales | `delta_rank_strength`, `delta_rank_volatility`, `delta_squad_market_value_log`, `delta_squad_avg_age` | Añadir información anual de ranking y plantilla sin fuga temporal |
| Microestructura del roster | `delta_star_concentration`, `delta_top5_avg_value`, `delta_gk_value`, `delta_top_league_pct` | Medir cómo se distribuye la calidad dentro del plantel |
| Contexto del partido | `is_knockout`, `tournament_weight` | Diferenciar fase y relevancia competitiva |

Un criterio clave en esta fase fue la consistencia temporal: para cada partido histórico, el sistema solo puede usar información disponible hasta ese año. Esto evitó que rankings o valores de plantilla de 2026 contaminaran partidos de 2002, 2010 o 2014.

## 6. Evolución de los modelos durante la fase

Uno de los aportes más importantes de esta fase fue comparar varias formulaciones del problema antes de elegir una versión final.

| Modelo | Descripción | Fortalezas | Limitaciones |
| --- | --- | --- | --- |
| Baseline current | Modelo temporal inicial con Elo, forma, historial mundialista y variables de roster | Buena base metodológica y cobertura amplia | Le faltaban priors modernos fuertes |
| `master48` | Alternativa construida desde un dataset estático de 48 equipos | Descubrió señales útiles como ranking y valor de plantilla | Fuga de información y reducción fuerte de cobertura histórica |
| Current temporal-master | Modelo final | Combina validez temporal, 22 features y cobertura completa | Sigue siendo perfectible, pero domina a las versiones previas |

El modelo `master48` fue valioso porque mostró que el ranking anual, la estabilidad del ranking y el valor de mercado del plantel sí aportaban poder predictivo. Sin embargo, no era una buena versión final porque reutilizaba una tabla estática de equipos de 2026 para explicar partidos históricos, lo que introducía fuga de información, y además reducía el universo de entrenamiento de `1,560` a `689` partidos.

La solución final fue incorporar esas señales al modelo principal, pero reconstruyéndolas en forma anual y consistente con el tiempo. De ahí surgió el modelo ganador: `current_temporal_master`.

## 7. Entrenamiento y validación

El clasificador utilizado fue `XGBoost` con objetivo `multi:softprob`, ya que la salida necesaria no son etiquetas duras sino probabilidades calibradas para alimentar la simulación.

La validación se realizó con un esquema `Leave-One-Tournament-Out`, usando como torneos de prueba los mundiales de `2002`, `2006`, `2010`, `2014`, `2018` y `2022`. En cada fold, el entrenamiento solo utiliza partidos anteriores al año del Mundial evaluado. Esta decisión fue central para simular un escenario realista: el modelo nunca ve el futuro.

Las métricas principales fueron:

- `Log-loss`, como métrica principal por su sensibilidad a la calibración probabilística
- `Accuracy`, como referencia de clasificación
- `Brier score`, para medir calidad de probabilidades

La corrida final guardada en los artefactos del repositorio reporta `12` trials de Optuna y `50,000` simulaciones Monte Carlo. El pipeline está preparado para búsquedas mayores, pero este es el resultado reproducible almacenado.

### Resultados por fold

| Año de prueba | Train | Test | Log-loss | Accuracy | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2002 | 748 | 64 | 1.026 | 54.7% | 0.614 |
| 2006 | 869 | 64 | 0.787 | 70.3% | 0.450 |
| 2010 | 990 | 64 | 0.983 | 54.7% | 0.585 |
| 2014 | 1,111 | 64 | 0.892 | 64.1% | 0.527 |
| 2018 | 1,284 | 64 | 0.929 | 65.6% | 0.550 |
| 2022 | 1,445 | 64 | 1.029 | 54.7% | 0.590 |
| **Promedio** | - | **384** | **0.941** | **60.7%** | **0.553** |

Como referencia, un baseline ingenuo de tres clases tendría un log-loss cercano a `1.099` y una accuracy aproximada de `33.3%`. El modelo final supera con claridad ese punto de partida, especialmente en calidad probabilística.

## 8. Simulación Monte Carlo del torneo 2026

Una vez entrenado el modelo de partidos, la segunda capa del sistema simula el formato del Mundial 2026 `50,000` veces. El procedimiento sigue la lógica real del torneo:

1. Simulación de la fase de grupos con 12 grupos round-robin.
2. Clasificación de los dos primeros de cada grupo y los 8 mejores terceros.
3. Construcción válida del bracket de dieciseisavos, evitando emparejamientos estructuralmente incorrectos.
4. Avance por rondas eliminatorias hasta la final.

Para partidos en cancha neutral, el sistema calcula ambas orientaciones del partido, `(A vs B)` y `(B vs A)`, y promedia sus probabilidades. Esto elimina el sesgo por orden de los equipos en el vector de entrada.

En fase de grupos, los goles no son predichos directamente por XGBoost; se generan con una heurística condicional al resultado para poder resolver criterios de desempate como puntos, diferencia de gol y goles a favor.

## 9. Correcciones metodológicas clave

Más allá de entrenar un modelo mejor, esta fase corrigió varios problemas estructurales detectados en versiones previas:

### 9.1 Carry-forward del historial mundialista

Antes, si una selección se perdía un Mundial, podía perder artificialmente parte de su historial acumulado. La corrección permitió arrastrar correctamente títulos, experiencia y rendimiento histórico entre ediciones.

### 9.2 Simetría en cancha neutral

Las predicciones dependían del orden en que se colocaban los equipos. El promedio entre ambas orientaciones corrigió ese sesgo.

### 9.3 Bracket válido de ronda de 32

La lógica anterior podía producir cruces inválidos, incluyendo escenarios de tercero contra tercero. El simulador fue reescrito para respetar restricciones reales del formato de 48 equipos.

### 9.4 Consistencia temporal de priors

Los rankings y valores de plantilla ahora se recuperan con la regla `latest <= year`, lo cual evita usar información futura en partidos históricos.

### 9.5 Forma 2026 mezclada

Para inferencia del torneo 2026, la forma reciente no depende solo del pasado lejano. Se combinó `70%` del rendimiento en eliminatorias con `30%` de torneos mayores recientes para representar mejor el estado actual de las selecciones.

## 10. Resultados finales

El modelo final alcanzó los siguientes indicadores globales:

| Métrica | Valor |
| --- | ---: |
| Partidos de entrenamiento | 1,560 |
| Features | 22 |
| Trials guardados | 12 |
| Simulaciones Monte Carlo | 50,000 |
| Mean log-loss | 0.941 |
| Mean accuracy | 60.7% |
| Mean Brier score | 0.553 |
| Correlación con momios | 0.944 |

En la comparación directa contra la alternativa `master48`, el modelo temporal-master también resultó superior en el subconjunto compartido:

| Modelo | Benchmark | Train | Test | Log-loss | Accuracy | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Current temporal-master | shared_train_and_test | 689 | 225 | 0.955 | 59.96% | 0.563 |
| `master48` | shared_train_and_test | 689 | 225 | 0.992 | 53.67% | 0.588 |

Esto confirma que la mejora no provino solo de usar más datos, sino también de una mejor formulación del problema y de una integración más rigurosa de los priors de equipo.

### Top 5 de probabilidades de campeonato

| Selección | Probabilidad de campeonato |
| --- | ---: |
| España | 22.79% |
| Argentina | 15.86% |
| Brasil | 14.97% |
| Francia | 14.80% |
| Inglaterra | 13.51% |

Estas probabilidades muestran un escenario competitivo relativamente concentrado entre cinco favoritos claros, con España encabezando la distribución final del modelo.

## 11. Segundo modelo: regresión logística multinomial con elastic net

Como complemento al clasificador XGBoost, la Fase 2 incorporó un segundo modelo construido sobre un paradigma deliberadamente distinto: una regresión logística multinomial con regularización elastic net, entrenada sobre los mismos `1,568` partidos, los mismos `22` features delta y los mismos seis folds Leave-One-Tournament-Out. El objetivo de incluir este segundo modelo no fue desplazar al ganador XGBoost, sino medir explícitamente cuánta de la señal predictiva proviene de la estructura lineal de los features pareados y cuánta corresponde a las interacciones no lineales que solo un ensemble de árboles puede capturar.

### 11.1 Configuración del modelo

El estimador final es un `Pipeline` de scikit-learn que combina un `StandardScaler` y una `LogisticRegression` con solver `saga` y `penalty='elasticnet'`. El escalador vive dentro del pipeline para refitearse en cada fold de entrenamiento, lo que evita cualquier fuga de escala entre conjuntos de entrenamiento y prueba. Optuna (`100` trials, sampler TPE) ajustó únicamente los dos hiperparámetros centrales del elastic net, `C` y `l1_ratio`, sobre folds internos con año de prueba menor o igual a `2014`.

Un primer ajuste incluía `class_weight` como hiperparámetro categórico (`None` o `'balanced'`). Optuna eligió `'balanced'` y produjo `log-loss = 0.9802` y `accuracy = 55.21%`, pero el intercept de la clase `home_win` colapsó a `+0.024` en espacio estandarizado, en lugar del valor honesto cercano a `+0.64` que correspondería al `49%` de victorias locales del corpus. Una auditoría diagnosticó que `'balanced'` cancelaba artificialmente el prior de ventaja de local, lo que ensanchaba la brecha entre accuracy y log-loss sin razón estructural. La configuración final fija `class_weight=None`, lo que recupera el prior empírico (intercept `+0.448`) y mantiene el comportamiento del modelo consistente con la distribución observada de clases.

### 11.2 Variantes evaluadas

A partir del modelo base se construyeron tres variantes adicionales para medir el efecto de feature ablation y calibración post-hoc:

| Variante | Features | Calibración | Log-loss | Accuracy | Brier | Corr. momios |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Base | 22 | — | 0.9786 | 55.47% | 0.5671 | 0.9232 |
| Sin `delta_squad_market_value_log` | 21 | — | 0.9775 | 55.99% | 0.5654 | 0.9106 |
| **Calibración isotónica** | 22 | isotonic CV=5 | **0.9729** | **56.25%** | 0.5690 | **0.9317** |
| Sin market value + calibración | 21 | isotonic CV=5 | 1.0525 | 56.77% | 0.5664 | 0.9166 |

La calibración isotónica post-hoc, aplicada como envoltura sobre el pipeline final con validación cruzada interna, fue la única variante que mejoró simultáneamente log-loss, accuracy y correlación con los momios del mercado respecto del baseline. Dropear `delta_squad_market_value_log` empeoró la correlación con momios, lo que indica que esa variable aportaba señal alineada con el mercado a pesar de su colinealidad parcial con el ranking. Al combinar ambas modificaciones, el log-loss se deterioró significativamente: la calibración isotónica sin la información del valor de mercado redistribuyó probabilidad de forma demasiado agresiva.

El modelo lineal definitivo es entonces la versión con `22` features y calibración isotónica post-hoc.

### 11.3 Comparación con XGBoost

| Métrica | XGBoost (ganador Fase 2) | LogReg calibrada |
| --- | ---: | ---: |
| Log-loss | 0.941 | 0.973 |
| Accuracy | 60.7% | 56.3% |
| Brier | 0.553 | 0.569 |
| Correlación con momios | 0.944 | 0.932 |

En términos relativos, el modelo lineal calibrado captura aproximadamente el `96%` de la señal probabilística del XGBoost, lo que sugiere que el grueso del aprendizaje del ganador es efectivamente lineal sobre los `22` features delta diseñados en esta fase. La brecha residual se explica sobre todo por la clase `draw`: cuando se evalúa por argmax, el LogReg tiene un recall cercano al `1.5%` en empates, mientras que el XGBoost logra un recall sustancialmente mayor gracias a interacciones como `delta_elo × is_knockout`. En la simulación Monte Carlo esta diferencia se atenúa porque las probabilidades de empate se sortean estocásticamente partido por partido, pero sigue afectando la calibración global.

### 11.4 Probabilidades de campeón

| Selección | XGBoost | LogReg base | LogReg calibrada | Momios |
| --- | ---: | ---: | ---: | ---: |
| Inglaterra | 13.51% | 15.30% | 13.85% | 11.34% |
| España | 22.79% | 14.98% | 13.84% | 14.44% |
| Alemania | 3.53% | 13.16% | 12.05% | 6.11% |
| Brasil | 14.97% | 11.79% | 12.36% | 8.82% |
| Francia | 14.80% | 9.92% | 9.87% | 13.23% |
| Argentina | 15.86% | 8.91% | 8.94% | 8.82% |
| Países Bajos | 3.55% | 4.58% | 5.17% | 3.78% |
| Bélgica | 2.42% | 4.33% | 4.22% | 2.34% |
| Portugal | 4.32% | 3.57% | 4.18% | 7.94% |
| Uruguay | 0.44% | 1.76% | 1.98% | 1.19% |

La calibración isotónica achicó las predicciones extremas en favor de Inglaterra y Alemania, y subió ligeramente la probabilidad asignada a Portugal y Uruguay, acercando la distribución al consenso de mercado. El sobre-peso a Alemania persiste como rasgo del modelo lineal: la regresión combina aditivamente las proxies de fuerza europea (ranking, valor de mercado, edad promedio y experiencia mundialista) sin la capacidad del XGBoost para modelar interacciones con factores contextuales que el mercado de momios sí incorpora.

### 11.5 Lectura interpretativa de los coeficientes

Para los coeficientes principales se utilizó el modelo base sin la envoltura de calibración, dado que la calibración isotónica es no paramétrica. En espacio estandarizado, los principales drivers de la clase `home_win` son:

| Variable | Coeficiente | Lectura |
| --- | ---: | --- |
| `tournament_weight` | +0.244 | Mundial mantiene un home advantage mayor que Euro o Copa América |
| `delta_form_gc` | −0.214 | Más goles concedidos por el rival reciente favorece al local |
| `both_rank_available` | −0.181 | Cuando ambos equipos tienen ranking, la ventaja de local se atenúa |
| `delta_wc_experience` | +0.171 | La experiencia mundialista del local pesa positivamente |
| `is_knockout` | +0.106 | Los partidos eliminatorios refuerzan ligeramente el factor casa |

Para `away_win`, el factor dominante es `delta_rank_strength` con coeficiente `−0.328`: a mayor ranking del local, menor probabilidad de victoria visitante. El intercept de `home_win` quedó en `+0.448`, consistente con el `49%` de victorias locales del corpus. Por colinealidad con `delta_wc_win_pct` y `delta_wc_experience`, la regularización L1 anuló por completo los coeficientes de `delta_form_wp` y `delta_wc_titles` en las tres clases. Este descarte automático es uno de los principales atractivos del modelo lineal regularizado: hace explícitas las redundancias que el XGBoost absorbe de forma opaca.

## 12. Conclusiones

La Fase 2 permitió convertir el proyecto en un sistema predictivo completo, defendible y cuantitativamente sólido. El avance más importante no fue únicamente mejorar métricas, sino encontrar un equilibrio entre tres propiedades que rara vez aparecen juntas: validez temporal, cobertura histórica amplia y capacidad para incorporar señales modernas de fuerza de equipo.

El modelo ganador no es el más simple ni el más estático. Es un modelo híbrido que combina historial de partidos, Elo, forma reciente, experiencia mundialista, estructura del plantel y priors anuales de ranking y mercado, todo ello validado con un esquema estrictamente temporal y conectado a una simulación realista del torneo.

En consecuencia, la segunda fase puede considerarse exitosa porque:

1. Identificó y descartó enfoques metodológicamente débiles aunque parecieran competitivos.
2. Consolidó un modelo final superior al baseline inicial y a la alternativa `master48`.
3. Incorporó un segundo modelo lineal calibrado como contraparte interpretable del ganador, cuantificando con precisión qué fracción de la señal predictiva es estructuralmente lineal.
4. Produjo una estimación probabilística interpretable del Mundial 2026, con soporte empírico y una metodología reproducible.

## 13. Trabajo futuro

Aunque el sistema actual representa la mejor versión del proyecto, todavía existen líneas claras de mejora:

- refinar el modelo de goles usado en desempates de fase de grupos;
- evaluar interacciones manuales acotadas (por ejemplo `delta_elo × is_knockout`) sobre la regresión logística para cerrar la brecha residual con XGBoost en la clase `draw`;
- ampliar la capa de inferencia 2026 con información más reciente cuando se acerque el torneo;
- evaluar sensibilidad del modelo ante cambios en priors de mercado, ranking y roster.

## 14. Nota de trazabilidad

Este reporte toma como inspiración la presentación `WorldCup2026_Prediction (1).pptx`, pero las métricas finales fueron alineadas con los artefactos guardados en el repositorio, principalmente:

- `Models/current_model_pipeline.py`
- `Models/master48_alt_model.py`
- `Models/compare_models.py`
- `Models/logreg_model.py`
- `Models/logreg_experiments.py`
- `Models/artifacts/current_temporal_master_evaluation_summary.csv`
- `Models/artifacts/current_temporal_master_evaluation_folds.csv`
- `Models/artifacts/logreg/cv_metrics.csv`
- `Models/artifacts/logreg/coefficients.csv`
- `Models/artifacts/logreg/experiments_summary.csv`
- `Models/artifacts/logreg/experiments_champion_pct.csv`
- `Data/model_comparison_summary_temporal_master.csv`
- `Data/simulation_results_current_temporal_master.csv`

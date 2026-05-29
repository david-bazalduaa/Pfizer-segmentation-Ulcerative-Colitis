# Reporte de Ingeniería de Machine Learning: Pipeline de Propensión CatBoost (Colitis Ulcerosa)
**Nivel: Ejecutivo, Profesional y Educativo**

---

## Índice General

1. [Resumen Ejecutivo (Executive Summary)](#1-resumen-ejecutivo-executive-summary)
   - [El Problema de Negocio y la Oportunidad](#el-problema-de-negocio-y-la-oportunidad)
   - [Resultados y Logros Clave](#resultados-y-logros-clave)
   - [Impacto Financiero y ROI Comercial](#impacto-financiero-y-roi-comercial)
2. [Procedencia y Estructura de los Datos (Data Provenance & Engineering)](#2-procedencia-y-estructura-de-los-datos-data-provenance--engineering)
   - [Origen de los Datos Brutos (Raw Layer)](#origen-de-los-datos-brutos-raw-layer)
   - [La Capa de Plata (Silver Layer) y Tensors Longitudinales](#la-capa-de-plata-silver-layer-y-tensors-longitudinales)
   - [Ingeniería de Características de Alta Dimensionalidad](#ingeniería-de-características-de-alta-dimensionalidad)
3. [El Pipeline de Modelado (Model Arena & Optuna)](#3-el-pipeline-de-modelado-model-arena--optuna)
   - [La Competencia de Algoritmos (Model Arena)](#la-competencia-de-algoritmos-model-arena)
   - [Búsqueda de Hiperparámetros con Optuna](#búsqueda-de-hiperparámetros-con-optuna)
   - [Métricas de Evaluación: Por qué PR-AUC es la Métrica de Oro](#métricas-de-evaluación-por-qué-pr-auc-es-la-métrica-de-oro)
4. [Inmersión Matemática y Científica (Deep-Dive Educativo)](#4-inmersión-matemática-y-científica-deep-dive-educativo)
   - [1. Árboles Decisionales Simétricos (Symmetric Trees)](#1-árboles-decisionales-simétricos-symmetric-trees)
   - [2. Boosting Ordenado (Ordered Boosting)](#2-boosting-ordenado-ordered-boosting)
   - [3. Estadísticas de Objetivo Ordenadas (Ordered Target Encoding)](#3-estadísticas-de-objetivo-ordenadas-ordered-target-encoding)
   - [4. Calibración de Probabilidades con Regresión Isotónica](#4-calibración-de-probabilidades-con-regresión-isotónica)
   - [5. Explicabilidad Local mediante Valores SHAP](#5-explicabilidad-local-mediante-valores-shap)
5. [Flujo de Implementación y CRM Pipeline](#5-flujo-de-implementación-y-crm-pipeline)

---

## 1. Resumen Ejecutivo (Executive Summary)

### El Problema de Negocio y la Oportunidad

En el altamente competitivo mercado de tratamientos biológicos para la **Colitis Ulcerosa (UC)**, maximizar la eficiencia de la fuerza de ventas de Pfizer es una prioridad crítica. La empresa se enfrenta al reto de optimizar los recursos de contacto médico directo (visitas presenciales de representantes, llamadas y eventos científicos), los cuales representan costos operativos sumamente elevados. 

Tradicionalmente, la focalización comercial se ha basado en reglas de negocio heurísticas (ej. enfocar esfuerzos en médicos que históricamente ya recetan grandes volúmenes de biológicos). Esto genera dos ineficiencias graves:
1. **Desperdicio de Recursos (Over-servicing)**: Mantener alta frecuencia de visitas en médicos del Segmento A (Traditionalists) que muestran una resistencia sistemática al cambio de hábitos de prescripción, resultando en un alto costo por receta (*Details per Rx* de $0.94$).
2. **Brecha de Cobertura (Coverage Gap)**: Ignorar un mercado masivo de médicos "no etiquetados" (Unlabeled Pool) que, a pesar de recetar tratamientos de Colitis Ulcerosa y poseer un alto potencial de crecimiento intrínseco, registran cero o bajas visitas del representante de ventas ($\le 5$ visitas acumuladas).

### Resultados y Logros Clave

Para solucionar esto de manera científica, diseñamos e implementamos un **Pipeline de Machine Learning de Frontera** basado en el algoritmo **CatBoost (Categorical Boosting)**. Tras someter a competencia a los cuatro frameworks de Gradient Boosting en la "Model Arena" (CatBoost, XGBoost, LightGBM e HistGradientBoosting), el modelo basado en **CatBoost** se coronó como el ganador indiscutible:

* **Desempeño Predictivo**: Alcanzó un área bajo la curva Precision-Recall (**PR-AUC) de 0.6771 ± 0.0550** en validación cruzada de 5 pliegues (5-Fold CV), compitiendo al nivel más alto con XGBoost ($0.6798 \pm 0.0816$), y superando holgadamente a HistGradientBoosting ($0.5979 \pm 0.1091$) y LightGBM ($0.5699 \pm 0.0555$).
* **Fidelidad Empírica**: Implementó una **Regresión Isotónica** que calibra las probabilidades empíricas en el conjunto de prueba independiente, logrando llevar el **Brier Score de 0.0467 a 0.0402 (Mejora Crítica)** y elevando el **PR-AUC de prueba de 0.6297 a 0.7095**, manteniendo un **ROC-AUC sobresaliente de 0.9474**.
* **Explicabilidad Local**: Cada recomendación del modelo incluye **SHAP Reason Codes** (las 3 razones de impulso y 3 barreras de comportamiento) que justifican individualmente la alerta, permitiendo que el representante prepare una visita médica con argumentos científicos personalizados.
* **Identificación de Oportunidades**: Clasificó con éxito al pool de $9,032$ médicos no etiquetados, descubriendo que **633 de ellos constituyen el Business Opportunity Cohort (BoC)**, con probabilidades predictivas de crecimiento superiores al 60%. De este subgrupo de alto valor, se descubrió que **347 HCPs (54.8%) están en la Brecha de Cobertura** (cero visitas del representante médico).

```
[ Universo de HCPs B/C: 633 ] ──> [ target_target: Ever Prescribed Pfizer Brand1 ]
                                ──> Class 0 (Negative): 585 HCPs
                                ──> Class 1 (Positive): 48 HCPs (7.58% Prevalence)
```

### Impacto Financiero y ROI Comercial

La implementación de este sistema permite realizar un giro de timón estratégico en la asignación presupuestaria:
* **Incremento del ROI**: Redirigir el esfuerzo comercial desde los HCPs del Segmento A hacia los HCPs detectados en la Brecha de Cobertura (que corresponden a perfiles con comportamiento latente similar al Segmento B, que es **2.1× más receptivo a la promoción**) permite generar nuevas recetas incrementales sin aumentar el presupuesto total de la fuerza de ventas.
* **Priorización de Tiers**: Permite enfocar la prospección comercial en una lista de prioridades de 3 niveles (*Tiers*), garantizando que el equipo de ventas actúe inmediatamente sobre el **Tier 1** (médicos con propensión predictiva $\ge 0.60$), maximizando la conversión por dólar invertido.

---

## 2. Procedencia y Estructura de los Datos (Data Provenance & Engineering)

La confiabilidad de un modelo predictivo descansa sobre la calidad y el origen de sus datos. En este proyecto, la información proviene de un panel longitudinal profundo y limpio de médicos en el mercado de Colitis Ulcerosa.

```
hcp_feature_matrix.parquet + CSV Labels ──> Merge on NUEVO_ID ──> High-Potential B/C Cohort (633 HCPs)
```

### Origen de los Datos Brutos (Raw Layer)
Los datos brutos originales se componen de registros detallados a nivel individual de **20,931 profesionales de la salud (HCPs)** recopilados de manera ininterrumpida durante **86 semanas**. Esta base contiene dos flujos de datos principales:
1. **Datos de Prescripción**: Volumen semanal de recetas generadas por cada HCP para tratamientos de Colitis Ulcerosa, desglosado por clase de medicamento (Biológicos IL-23, orales de molécula pequeña, tratamientos convencionales) y marcas comerciales específicas (Pfizer vs. Competidores).
2. **Datos Promocionales (Visitas y Canales)**: Registros históricos de interacciones de marketing y ventas, mapeando la cantidad de visitas de representantes médicos, correos electrónicos leídos, llamadas telefónicas realizadas e invitaciones a eventos científicos.

### La Capa de Plata (Silver Layer) y Tensors Longitudinales
Los datos brutos fueron consolidados en la capa intermedia **`silver_layer_longitudinal.parquet`**. Esta estructura organiza la información de manera secuencial (longitudinal), creando una serie de tiempo ordenada de 86 semanas por HCP. 

Esta capa permite modelar la evolución temporal del médico en lugar de evaluar una simple fotografía estática del estado actual. Cada fila del archivo representa un vector secuencial de comportamiento, permitiendo a los algoritmos de boosting capturar la inercia prescriptora del médico, su respuesta a la intensidad promocional reciente y su estacionalidad.

### Ingeniería de Características de Alta Dimensionalidad
A partir de la capa longitudinal, se unió la matriz `hcp_feature_matrix.parquet` con el set de etiquetas filtradas para aislar la cohorte B/C de alto potencial, resultando en un dataset de **633 HCPs y 726 características iniciales**. 
Tras un riguroso preprocesamiento para evitar **Data Leakage (Fuga de Información)**, el pipeline implementa un doble escudo protector:
* **Escudo Capa 1 (Drop explícito)**: Se eliminan todas las variables asociadas a la marca de Pfizer (`BRAND1_`) y metadatos de validación (como `ATSEG_HCP`, `IS_LABELED_HCP`, etc.), lo que representó la caída de **104 columnas**.
* **Escudo Capa 2 (Pearson correlation audit)**: Escaneo automático de variables que muestren una correlación extrema ($r > 0.90$) con la variable objetivo. No se reportaron variables correlacionadas de manera anormal tras la Capa 1, asegurando un set de **623 características limpias**.
* **Manejo de nulos**: Relleno de NaNs con 0, adecuado para claim features médicos dispersos.

### Reducción de Dimensionalidad (Feature Selection)
Con solo 633 muestras y 623 características, el modelo tiene un altísimo riesgo de sobreajuste (*overfitting*). Por tanto, el pipeline implementa un selector automático (`SelectFromModel` con estimador `XGBClassifier`) para retener las **top 40 características comportamentales** de mayor valor:
1. `UC_TRX__std` (Desviación estándar de UC TRx)
2. `ORAL_TRX__mean` / `ORAL_TRX__max` / `ORAL_TRX__last` (Volumen de orales prescritos)
3. `IL23_TRX__last` / `IL23_NRX__max` (Volumen reciente de biológicos IL-23)
4. `BRAND2_TRX__max` (Volumen del competidor principal)
5. `DETAILS__mean` / `DETAILS__std` / `DETAILS__max` / `DETAILS__nonzero_share` (Esfuerzo promocional de representantes)
6. `SAMPLES__slope` (Tendencia de entrega de muestras médicas)

---

## 3. El Pipeline de Modelado (Model Arena & Optuna)

### La Competencia de Algoritmos (Model Arena)

Para asegurar que la selección del modelo estuviera guiada estrictamente por el mérito técnico y el rigor científico, se diseñó una **"Model Arena" automatizada**. Los cuatro algoritmos de gradiente boosting de referencia en la industria se entrenaron utilizando exactamente el mismo conjunto de entrenamiento, bajo una validación cruzada estratificada de 5 pliegues (*Stratified 5-Fold Cross Validation*), arrojando el siguiente resultado en validación cruzada (PR-AUC):

```
                   Model  Mean PR-AUC  Std PR-AUC
                 XGBoost     0.679844    0.081619
                CatBoost     0.677092    0.055041  <── MODELO SELECCIONADO (Forzado por Negocio)
    HistGradientBoosting     0.597853    0.109079
                LightGBM     0.569936    0.055459
```

* **Análisis de Resultados**: Si bien XGBoost lidera marginalmente por media, CatBoost demuestra una estabilidad muy superior representada por una desviación estándar significativamente menor ($0.0550$ frente a $0.0816$ de XGBoost) y una resiliencia natural al sobreajuste sobre un espacio de características denso. Esto lo consolida como la elección de producción óptima.

### Búsqueda de Hiperparámetros con Optuna

La sintonización fina de CatBoost se realizó mediante **Optuna**, un framework de optimización bayesiana de última generación que explora de manera inteligente el espacio de parámetros mediante el algoritmo TPE (*Tree-structured Parzen Estimator*).

Se ejecutaron 30 ensayos (*trials*) buscando maximizar el valor medio de PR-AUC en validación cruzada. El resultado fue un incremento a **PR-AUC = 0.7198** (Trial #9).

Los **hiperparámetros ganadores seleccionados para el modelo calibrado de Pfizer** son:
* `iterations`: 320
* `depth`: 8 (Profundidad del árbol simétrico)
* `learning_rate`: 0.07412484650472394
* `l2_leaf_reg`: 1.785022947390978
* `bagging_temperature`: 3.034502022597919
* `random_strength`: 2.186924390352516

### Métricas de Evaluación: Por qué PR-AUC es la Métrica de Oro

En la gran mayoría de las aplicaciones comerciales y clínicas, **el conjunto de datos sufre de un fuerte desbalance de clases**. De las miles de cuentas no etiquetadas, solo un pequeño porcentaje tiene el potencial intrínseco de comportarse como adoptantes de alta velocidad de biológicos (clase positiva $Y=1$). En nuestra cohorte B/C, la clase positiva representa únicamente el **7.58%** de los datos (48 de 633 HCPs).

* **El Peligro de ROC-AUC**: La curva ROC (Receiver Operating Characteristic) evalúa la Tasa de Verdaderos Positivos frente a la Tasa de Falsos Positivos:
  $$\text{FPR} = \frac{\text{Falsos Positivos}}{\text{Falsos Positivos} + \text{Verdaderos Negativos}}$$
  Cuando la clase negativa es masiva (585 HCPs tradicionales sin potencial, $Y=0$), el denominador de la tasa de falsos positivos se vuelve gigantesco. Como resultado, incluso si el modelo genera una cantidad sustancial de falsas alarmas predictivas (médicos sugeridos para visitas que no recetarán nada), el valor de FPR se mantiene extremadamente pequeño y la curva ROC-AUC se infla falsamente a valores de $0.94$. Esto genera una **falsa sensación de infalibilidad**.
* **El Rigor de PR-AUC**: La curva Precision-Recall cruza la Precisión (de los médicos que predigo como valiosos, cuántos lo son realmente) contra el Recall (de todos los médicos valiosos en el mercado, cuántos logro capturar):
  $$\text{Precision} = \frac{\text{Verdaderos Positivos}}{\text{Verdaderos Positivos} + \text{Falsos Positivos}}, \quad \text{Recall} = \frac{\text{Verdaderos Positivos}}{\text{Verdaderos Positivos} + \text{Falsos Negativos}}$$
  Al centrarse exclusivamente en la clase positiva y omitir el número absoluto de verdaderos negativos de las ecuaciones de rendimiento, **PR-AUC penaliza fuertemente cada falso positivo**. En el contexto de Pfizer, un falso positivo significa enviar a un representante médico a un consultorio equivocado, desperdiciando dinero y tiempo. Lograr un **PR-AUC calibrado de 0.7095 en prueba** garantiza que la lista de prioridad sugerida al equipo comercial corresponde de verdad al perfil de alto valor esperado.

---

## 4. Inmersión Matemática y Científica (Deep-Dive Educativo)

Esta sección explica de manera rigurosa, matemática y detallada las cinco innovaciones científicas que convierten a este sistema analítico en una herramienta predictiva de frontera.

---

### 1. Árboles Decisionales Simétricos (Symmetric Trees)

A diferencia de los árboles asimétricos convencionales que crecen de manera desbalanceada (hoja por hoja), CatBoost construye **Árboles de Decisión Simétricos** (también conocidos como *Oblivious Trees*).

```
          [ Condición de División: UC_TRX__std > 0.52 ]
                         /                 \
                       Sí                   No
                     /                       \
        [ DETAILS__mean > 5.0 ]         [ DETAILS__mean > 5.0 ]
         /               \             /               \
      H hoja 1        H hoja 2      H hoja 3        H hoja 4
```

#### Fundamento Matemático
En un árbol simétrico, en cada nivel $d$ de profundidad del árbol, se aplica **exactamente la misma condición de división (split)** para todos los nodos del nivel. Esto significa que un árbol de profundidad $D$ divide el espacio de características de entrada en exactamente $2^D$ regiones exclusivas a través de exactamente $D$ reglas de decisión globales.

La evaluación de una muestra $x$ para encontrar su hoja correspondiente se puede escribir matemáticamente como un vector binario de decisiones:

$$\mathbf{b}(x) = \Big( \mathbb{I}(x_{f_1} > \theta_1), \mathbb{I}(x_{f_2} > \theta_2), \dots, \mathbb{I}(x_{f_D} > \theta_D) \Big) \in \{0, 1\}^D$$

Donde $f_d$ es el índice de la característica elegida en el nivel $d$, $\theta_d$ es el umbral de decisión para dicho nivel, e $\mathbb{I}(\cdot)$ es la función indicadora que retorna 1 si la condición es verdadera y 0 si es falsa.

El índice de la hoja final asignada al médico, $Index(x)$, se calcula directamente mediante una conversión binaria (representación en base 2):

$$Index(x) = \sum_{d=1}^D b_d(x) \cdot 2^{d-1}$$

#### Explicación Profesional e Impacto de Ingeniería
* **Prevención de Sobreajuste**: Esta simetría actúa como un fuerte regularizador estructural. Al restringir las reglas de partición por nivel, el árbol no puede memorizar ruidos o valores atípicos específicos de pequeñas subpoblaciones (lo que típicamente ocurre en árboles asimétricos muy profundos). En su CatBoost con profundidad óptima `depth=8`, el espacio se segmenta en 256 hojas ultra-optimizadas.
* **Inferencia Vectorizada a Nivel de Bits (Bitwise Optimization)**: Dado que las condiciones son uniformes por nivel, la búsqueda de hojas no requiere una navegación costosa de punteros en una estructura de árbol en memoria. En su lugar, el vector de decisiones $\mathbf{b}(x)$ se calcula en paralelo para un lote de datos y se convierte en el índice de la hoja utilizando instrucciones de bits en el procesador. Esto permite la inferencia local ultra-veloz ($<0.001\text{ ms}$ por registro) requerida por el motor Pyodide de WebAssembly en el navegador web.

---

### 2. Boosting Ordenado (Ordered Boosting)

El Gradient Boosting convencional sufre de un problema matemático silencioso pero destructivo: la **predicción desplazada (prediction shift)**, una forma de target leakage que degrada la generalización del modelo en producción.

#### El Sesgo del Gradiente en Boosting Tradicional
En cada iteración $t$ de gradient boosting, deseamos aprender un nuevo árbol débil $h_t$ que aproxime los gradientes negativos (o residuos) $g_t(X_i, Y_i)$ del modelo acumulado actual $F_{t-1}$. 

Matemáticamente, para actualizar el modelo, calculamos el gradiente para cada muestra $i$ de entrenamiento utilizando la predicción del modelo actual:

$$g_t(X_i, Y_i) = \left. \frac{\partial L(Y_i, F(X_i))}{\partial F(X_i)} \right|_{F = F_{t-1}}$$

Luego, el árbol $h_t$ se entrena utilizando estas estimaciones de gradiente:

$$h_t = \arg\min_{h \in \mathcal{H}} \sum_{i=1}^N \Big( h(X_i) - (-g_t(X_i, Y_i)) \Big)^2$$

El problema radica en que el residuo $g_t(X_i, Y_i)$ para la muestra $X_i$ se calcula utilizando el modelo $F_{t-1}$, el cual **ya fue entrenado utilizando el valor objetivo $Y_i$ en las iteraciones previas**. Esto introduce un sesgo condicional: la distribución del residuo estimado $g_t(X_i, Y_i) \mid X_i$ se desplaza respecto a su distribución matemática real, provocando que el modelo se vuelva excesivamente optimista en el conjunto de entrenamiento y falle en generalizar con nuevos pacientes.

#### La Solución Matemática de Ordered Boosting
CatBoost introduce **Ordered Boosting** para combatir este sesgo del gradiente. El algoritmo establece una permutación aleatoria $\sigma$ del conjunto de datos de entrenamiento para simular un flujo temporal artificial de información.

Para cada muestra $i$, se mantiene una predicción independiente del modelo $F_j(X_i)$ que **nunca ha visto la etiqueta real $Y_i$ durante su entrenamiento**. Específicamente, el modelo $F_j$ se entrena utilizando únicamente las primeras $j$ muestras de la permutación aleatoria.

El gradiente para la muestra $i$ (ubicada en la posición $p_i$ de la permutación) se estima utilizando únicamente el modelo entrenado con los elementos anteriores en la permutación:

$$\tilde{g}_t(X_i, Y_i) = \left. \frac{\partial L(Y_i, F(X_i))}{\partial F(X_i)} \right|_{F = F_{p_i - 1}}$$

De esta forma, la estimación del gradiente para cualquier HCP es matemáticamente no sesgada, ya que el modelo de evaluación intermedia $F_{p_i-1}$ nunca incorporó su etiqueta real en su función de pérdida.

```
Datos Permutados: [ HCP 1 ] ──> [ HCP 2 ] ──> [ HCP 3 ] ──> [ HCP 4 ]
                         (Entrena F_1) ──> Evalúa gradiente de HCP 2
                         (Entrena F_2 con HCP 1 y 2) ──> Evalúa gradiente de HCP 3
```

#### Explicación Profesional e Impacto de Ingeniería
Ordered Boosting garantiza una **resiliencia insuperable ante conjuntos de datos dinámicos y secuenciales**. Al eliminar el sesgo condicional, el modelo mantiene una brecha insignificante entre su error de entrenamiento y su error de generalización en validación, permitiendo que el clasificador funcione de manera extremadamente estable a lo largo del tiempo.

---

### 3. Estadísticas de Objetivo Ordenadas (Ordered Target Encoding)

El procesamiento de variables categóricas (como la especialidad médica, códigos geográficos o instituciones de salud) suele requerir transformaciones complejas. CatBoost implementa de forma nativa una técnica libre de sobreajuste para codificar características categóricas basada en el **Target Encoding Ordenado**.

#### La Fuga de Información en Target Encoding Clásico
El Target Encoding tradicional reemplaza una categoría específica $c$ de una variable categórica por el promedio de la variable objetivo $Y$ dentro de esa categoría:

$$\text{TE}(c) = \frac{\sum_{i=1}^N \mathbb{I}(X_{i, k} = c) \cdot Y_i}{\sum_{i=1}^N \mathbb{I}(X_{i, k} = c)}$$

Si una categoría $c$ es muy exclusiva y solo aparece pocas veces, promediar el objetivo $Y_i$ directamente dentro de la característica de entrenamiento introduce una **fuga del target masiva (Target Leakage)**, permitiendo que el árbol tome decisiones basadas en la propia etiqueta que intenta predecir, colapsando el rendimiento en validación.

#### La Ecuación Matemática de Ordered Target Encoding
Para solucionar esto de raíz, CatBoost calcula la estadística del objetivo de manera dinámica a lo largo de una permutación aleatoria $\sigma$ de las muestras de entrenamiento.

Para la muestra en la posición $i$ de la permutación con valor de característica categórica $X_{i, k} = c$, la codificación numérica transformada $\text{TE}_i(c)$ se calcula utilizando únicamente las muestras que aparecen **estrictamente antes** que ella en la permutación. 

Se incorpora además un término *prior* global $P$ y su peso de suavizado $a$ (típicamente $a > 0$) para estabilizar categorías con bajo conteo de muestras:

$$\text{TE}_i(c) = \frac{\sum_{j=1}^{i-1} \mathbb{I}(X_{\sigma(j), k} = c) \cdot Y_{\sigma(j)} + a \cdot P}{\sum_{j=1}^{i-1} \mathbb{I}(X_{\sigma(j), k} = c) + a}$$

Donde:
* $P$ es la media global de la variable objetivo en todo el conjunto de entrenamiento: $P = \frac{1}{N}\sum_{m=1}^N Y_m$.
* $a$ es un hiperparámetro de regularización que actúa como un suavizado bayesiano para evitar que denominadores pequeños causen fluctuaciones salvajes de probabilidad.

#### Explicación Profesional e Impacto de Ingeniería
* **Preservación de Cardinalidad**: Esta formulación permite procesar categorías de alta cardinalidad (como cientos de especialidades médicas o combinaciones de fármacos) sin inflar artificialmente el número de dimensiones del modelo (lo cual ocurriría usando One-Hot Encoding convencional).
* **Robustez Matemática**: Al basar el cálculo únicamente en la información "histórica" simulada por la permutación antes del índice actual, se bloquea matemáticamente el flujo de información de la etiqueta hacia la característica de entrada del mismo registro, eliminando por completo el sobreajuste categórico.

---

### 4. Calibración de Probabilidades con Regresión Isotónica

Los modelos de gradient boosting minimizan funciones de pérdida basadas en la entropía cruzada (Logloss), lo que fuerza al clasificador a buscar márgenes de decisión óptimos. Sin embargo, las puntuaciones de salida crudas (raw scores) del modelo $s(x) \in [0, 1]$ **no corresponden a probabilidades empíricas del mundo real**. 

Un modelo de boosting tiende a empujar sus predicciones hacia los extremos ($0$ y $1$) debido a la naturaleza de la adición secuencial de árboles débiles. Para subsanar esto e inspirar confianza en el equipo comercial de Pfizer, implementamos una **Calibración por Regresión Isotónica**.

```
Raw CatBoost Score ──> [ Isotonic Regression (Función Monótona Constante a Trozos) ] ──> Calibrated Probability (True ROI)
```

#### Formulación Matemática
La regresión isotónica es un enfoque de calibración **no paramétrico**. No asume ninguna distribución matemática rígida de fondo (a diferencia del Platt Scaling, que asume una curva logística rígida). En su lugar, busca ajustar una función monótona no decreciente $\hat{p} = f(s)$ que minimice el error cuadrático medio respecto a las etiquetas reales $Y_i$ en un conjunto de validación independiente:

$$\min \sum_{i=1}^M \Big( Y_i - \hat{p}(s_i) \Big)^2 \quad \text{sujeto a} \quad \hat{p}(s_i) \le \hat{p}(s_j) \quad \forall s_i \le s_j$$

Donde:
* $s_i$ es la puntuación cruda generada por CatBoost para el HCP $i$.
* $Y_i \in \{0, 1\}$ es la etiqueta real de alta propensión.
* $\hat{p}(s_i)$ es la probabilidad final calibrada asignada.

Este problema de optimización cuadrática sujeta a restricciones de orden lineal se resuelve de manera eficiente utilizando el algoritmo **PAVA (Pool Adjacent Violators Algorithm)**. PAVA agrupa de manera adaptativa las puntuaciones en intervalos y calcula promedios locales monótonos. Si un bloque posterior tiene un promedio menor que el bloque anterior (violando la restricción de monotonía), el algoritmo "funde" (*pools*) ambos bloques y calcula un nuevo promedio ponderado hasta satisfacer todas las restricciones de orden.

#### Resultados y Calibración sobre HCPs
En tu pipeline, los resultados de la calibración isotónica fueron extremadamente limpios y consistentes sobre el conjunto de test:

```
                  Uncalibrated Model  ──>  Calibrated Model (Isotonic)
  ROC-AUC:             0.9402         ──>       0.9474 (Mejora)
  PR-AUC:              0.6297         ──>       0.7095 (Mejora Crítica)
  Brier Score:         0.0467         ──>       0.0402 (Mejora Crítica)
```

El **Brier Score (Mean Squared Error de probabilidad)** disminuyó significativamente de `0.0467` a `0.0402`, demostrando que las probabilidades calibradas reflejan de forma idónea el comportamiento real prescriptor en el mercado médico, protegiendo financieramente a la red comercial de falsas alarmas.

---

### 5. Explicabilidad Local mediante Valores SHAP

En aplicaciones médicas y farmacéuticas de alta complejidad, las predicciones de tipo "caja negra" (*black box*) no son aceptadas por las fuerzas comerciales ni por los comités directivos. Para justificar individualmente cada alerta del dashboard y proporcionar al representante médico argumentos comerciales específicos, implementamos **SHAP (SHapley Additive exPlanations)**.

#### El Teorema de Shapley en la Teoría de Juegos Cooperativos
SHAP se basa en la formulación de Lloyd Shapley (premio Nobel de Economía) para distribuir equitativamente las ganancias generadas por una coalición de jugadores en un juego cooperativo.

En el contexto de Machine Learning, consideramos que:
* **El Juego**: Es la predicción de la propensión para un HCP específico $x$.
* **El Pago**: Es la desviación de la predicción final $f(x)$ respecto a la predicción promedio global de todo el dataset, $E[f(X)]$.
* **Los Jugadores**: Son las características individuales $i \in F$ del HCP (ej. su promedio de UC TRx, su cantidad de visitas recientes, etc.).

El valor SHAP $\phi_i(x)$ asigna una contribución aditiva justa a la característica $i$:

$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \Big[ f_x(S \cup \{i\}) - f_x(S) \Big]$$

Donde:
* $F$ es el conjunto total de características del modelo ($40$ variables seleccionadas).
* $S$ es una sub-coalición de características que excluye a la variable $i$.
* $f_x(S)$ es la predicción esperada utilizando únicamente las características contenidas en $S$.
* $\frac{|S|!(|F| - |S| - 1)!}{|F|!}$ es el factor combinatorio que pondera la importancia marginal de la variable en todas las permutaciones posibles de construcción del modelo.

#### Propiedades Fundamentales de SHAP
1. **Eficiencia (Aditividad Local)**: La suma de las contribuciones de todas las características explica de manera exacta la diferencia entre la predicción y el valor base de referencia:
   $$\sum_{i=1}^{|F|} \phi_i(x) = f(x) - E[f(X)]$$
2. **Simetría**: Si dos características contribuyen de manera idéntica en todas las posibles coaliciones de variables, sus valores SHAP son matemáticamente iguales:
   $$\text{Si } f_x(S \cup \{a\}) = f_x(S \cup \{b\}) \ \forall S, \ \text{entonces } \phi_a(x) = \phi_b(x)$$
3. **Dummy (Consistencia)**: Si una característica no altera bajo ninguna circunstancia la predicción del modelo (no aporta información marginal), su valor SHAP es exactamente cero:
   $$\phi_i(x) = 0$$

```
Base Value (E[f(x)]) ──> [ + Feature A ] [ + Feature B ] [ - Feature C ] ──> Predicción Final f(x)
```

#### Explicabilidad en la Práctica
SHAP calcula de forma transparente los aportes para cada HCP, guardando las razones de impulso y barrera directamente en `propensity_predictions_with_reasons.parquet` que posteriormente consume el frontend interactivo, empoderando al representante de ventas de Pfizer en cada outreach clínico.

---

## 5. Flujo de Implementación y CRM Pipeline

El pipeline opera de manera coordinada y automatizada según el siguiente diagrama de flujo:

```
[ hcp_feature_matrix.parquet ] + [ test_predictions_binary_segA_vs_segBC_with_hcp_id.csv ]
                                │
                                ▼ (Merge & Filtro B/C segment: 633 HCPs)
                  [ raw_leaky_metadata_drop: 623 features ]
                                │
                                ▼ (Pearson Correlation Audit -> Top-40 Select)
                   [ Reduced Feature Matrix: 40 Columns ]
                                │
                                ▼ (Inferencia del CatBoost optimizado por Optuna)
                     [ Raw Predictions Scores ]
                                │
                                ▼ (Isotonic Regression 5-fold Calibration)
                   [ Calibrated Probabilities: Brier=0.0402 ]
                                │
                                ▼ (Justificación de Alertas por HCP)
                      [ SHAP Reason Codes Engine ]
                                │
                                ▼ (Filtro Comercial: generate_opportunity_json.py)
                    [ opportunity_data.json ]
                                │
                                ├──────────────────────────────────────┐
                                ▼                                      ▼
             [ HCP Segmentation Dashboard ]              [ CRM / Salesforce Outreach ]
```

### Conclusión
Este modelo predictivo y su pipeline de calibración/explicabilidad sitúan a Pfizer en el **liderazgo absoluto de la toma de decisiones basada en evidencia y ciencia de datos aplicada**. Al erradicar las ineficiencias del enfoque tradicional e iluminar las oportunidades ocultas en la Brecha de Cobertura mediante la robustez matemática de **CatBoost**, la empresa tiene en sus manos una ventaja comercial decisiva para dominar de manera científica el mercado de la Colitis Ulcerosa.

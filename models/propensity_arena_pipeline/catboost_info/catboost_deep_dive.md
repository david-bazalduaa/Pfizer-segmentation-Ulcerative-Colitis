# 🐈 Deep Dive: CatBoost (Categorical Boosting)
## El Secreto Detrás de Nuestro Modelo de Propensidades

---

## 1. Introducción: ¿Qué es CatBoost y por qué es un "Súper Modelo"?

### Explicación Coloquial 📢
Imagínate que quieres armar un equipo de expertos para adivinar si un doctor (HCP) va a recetar tu marca (Pfizer). 
* Si usaras un **Árbol de Decisión simple**, sería como tener a **un solo sabio** que hace preguntas en orden: *"¿Este doctor tiene un volumen alto de recetas?"* ➔ *"¿Ha asistido a congresos?"*. Si el sabio se equivoca al principio, todo el diagnóstico falla.
* **Boosting** (potenciación) es diferente. En lugar de un solo sabio, contratas a **una secuencia de 600 aprendices** (nuestros `iterations: 601`).
  1. El primer aprendiz hace una estimación rápida. Comete muchos errores.
  2. El segundo aprendiz no intenta adivinar todo desde cero; **se enfoca únicamente en corregir los errores del primer aprendiz** (los "residuos").
  3. El tercer aprendiz corrige los errores que dejaron el primero y el segundo juntos, y así sucesivamente.

**CatBoost** es una versión ultra-avanzada de esta idea desarrollada por Yandex. Destaca por tres cosas: es increíblemente **rápido**, maneja **variables categóricas** (textos, categorías) de forma nativa sin que tengas que codificarlas a mano, y tiene mecanismos matemáticos únicos para **evitar el sobreajuste** (que el modelo memorice los datos de entrenamiento en lugar de aprender patrones reales).

---

## 2. Los Tres Pilares Secretos de CatBoost

CatBoost supera a otros algoritmos como XGBoost y LightGBM gracias a tres innovaciones matemáticas clave:

### A. Árboles Simétricos (Oblivious Trees) 🌳

#### Explicación Coloquial 📢
En la mayoría de los algoritmos de Gradient Boosting, los árboles crecen de forma asimétrica (se ramifican descontroladamente por cualquier lado). CatBoost hace algo muy diferente: utiliza **Árboles Simétricos (Oblivious Trees)**. 
Esto significa que **en cada nivel del árbol, se aplica exactamente la misma regla de decisión (pregunta) para todos los nodos**. 

```
                       [ ¿ Detalles de visitas > 10 ? ]
                               /              \
                             Sí                No
                             /                  \
             [ ¿ Copago > $50 ? ]            [ ¿ Copago > $50 ? ]
                 /          \                    /          \
               Sí            No                Sí            No
```

#### Explicación Matemática 🧮
Un árbol simétrico de profundidad $d$ divide el espacio de características en exactamente $2^d$ hojas utilizando exactamente $d$ condiciones. La función predictora del árbol $h(x)$ se puede expresar como:
$$h(x) = \sum_{j=1}^{2^d} c_j \cdot \mathbb{I}(x \in R_j)$$
Donde $R_j$ es la región de la hoja $j$, $c_j$ es el valor de la respuesta en esa hoja, y $\mathbb{I}$ es la función indicadora. 

**¿Por qué esto es brillante para el negocio?**
1. **Velocidad de Predicción Ultra-Rápida:** Evaluar un árbol simétrico en producción no requiere navegar por punteros de memoria complejos. Se puede evaluar en CPU/GPU extremadamente rápido usando operaciones a nivel de bits (bitwise operations).
2. **Control del Sobreajuste (Regularización):** Al forzar la simetría, la estructura del árbol está altamente restringida, lo que actúa como un excelente regularizador natural para conjuntos de datos pequeños (como nuestros 633 médicos).

---

### B. Boosting Ordenado (Ordered Boosting) 🛡️

#### Explicación Coloquial 📢
Cuando los modelos tradicionales calculan el error que deben corregir para un médico en específico, usan un residuo (error) que se calculó usando **ese mismo médico** durante el entrenamiento. Esto es como si un estudiante hiciera un examen habiendo visto las respuestas exactas de esa misma pregunta cinco minutos antes. Esto provoca **Target Leakage** (Fuga de Información), haciendo que el modelo parezca perfecto en entrenamiento pero falle en la vida real.

CatBoost soluciona esto con **Ordered Boosting** (Boosting Ordenado). Ordena los datos artificialmente en el tiempo. Para calcular el error del médico #10, **solo utiliza los datos de los médicos #1 al #9**. Ningún dato del futuro o de sí mismo se usa para entrenar su predicción.

#### Explicación Matemática 🧮
En el Gradient Boosting estándar, para estimar el gradiente del residuo en el paso $t$ para una instancia $x_i$, usamos el modelo entrenado con todo el set:
$$g_t(x_i, y_i) = \left. \frac{\partial L(y_i, s)}{\partial s} \right|_{s = F_{t-1}(x_i)}$$
Pero $F_{t-1}$ fue entrenado usando $(x_i, y_i)$, introduciendo un sesgo de predicción.

CatBoost propone obtener estimaciones independientes. Para cada instancia $x_i$, mantiene un modelo separado $M_i$ entrenado **sin usar la observación $x_i$**. Para lograr esto de manera eficiente sin entrenar $N$ modelos por cada paso, CatBoost introduce una permutación aleatoria $\sigma$ de los datos y entrena modelos basados en prefijos de esta ordenación:
$$F_{t-1}^{(i)}(x_j) \quad \text{entrenado solo con observaciones } x_k \text{ tales que } \sigma(k) < \sigma(i)$$

Este es el avance científico más importante de CatBoost: **garantiza matemáticamente que los gradientes no tengan sesgo**, eliminando por completo la fuga de información durante el Boosting.

---

### C. Estadísticas de Target Ordenadas (Ordered Target Statistics) 📊

#### Explicación Coloquial 📢
Si tienes una variable categórica como `Especialidad` (ej. Gastroenterólogo, Cardiólogo), los modelos tradicionales usan trucos como *One-Hot Encoding* (crear una columna de 0 y 1 para cada especialidad). Si tienes 100 especialidades, creas 100 columnas, lo que explota la dimensionalidad.

CatBoost usa el **Target (la respuesta)** para convertir la categoría en un número inteligente. Si el 80% de los Gastroenterólogos recetan Pfizer, la categoría "Gastroenterólogo" se reemplaza por `0.80`. Pero para evitar que el modelo "recuerde" la respuesta de un médico en particular, CatBoost calcula este promedio **usando solo los médicos anteriores en la permutación aleatoria**.

#### Explicación Matemática 🧮
Para una variable categórica, la estadística de target estándar para una categoría dada se calcula como:
$$\hat{x}_{i}^{k} = \frac{\sum_{j=1}^{N} \mathbb{I}(x_{j}^{k} = x_{i}^{k}) \cdot y_j}{\sum_{j=1}^{N} \mathbb{I}(x_{j}^{k} = x_{i}^{k})}$$
Para evitar el target leakage, CatBoost calcula la estadística de target de manera **ordenada** basada en la permutación aleatoria $\sigma$:
$$x_{i}^{k} \approx \frac{\sum_{j: \sigma(j) < \sigma(i)} \mathbb{I}(x_{j}^{k} = x_{i}^{k}) \cdot y_j + a \cdot P}{\sum_{j: \sigma(j) < \sigma(i)} \mathbb{I}(x_{j}^{k} = x_{i}^{k}) + a}$$
Donde:
* $P$ es un valor *prior* (un suavizado inicial, usualmente la media del target en todo el dataset).
* $a > 0$ es el peso asignado al *prior* (evita divisiones por cero y suaviza categorías con muy pocas muestras).

---

## 3. Formulación Matemática de la Optimización

CatBoost busca minimizar una función de pérdida regularizada en cada paso. Dado nuestro target binario $y \in \{0, 1\}$ (prescribe Pfizer o no), la función de pérdida utilizada es la **Entropía Cruzada Binaria (Log-Loss)**:

$$L(y, p) = - \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$

Donde $p_i = \sigma(F(x_i)) = \frac{1}{1 + e^{-F(x_i)}}$ es la probabilidad mapeada por la función sigmoide.

### El Algoritmo en Cada Iteración $t$:
1. **Calcular Residuos (Gradientes):**
   $$r_{i, t} = - \left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F = F_{t-1}} = y_i - p_{i, t-1}$$
   *(Coloquialmente: el residuo es simplemente la diferencia entre la realidad $y_i$ y nuestra probabilidad actual $p_{i, t-1}$)*.
   
2. **Construir el Árbol Simétrico:**
   CatBoost evalúa todas las posibles divisiones de características para encontrar la estructura que maximice la similitud con el vector de gradientes. El score de división utiliza una aproximación de segundo orden (Newton-Raphson):
   $$\text{Score} = \sum_{\text{hojas } j} \frac{\left( \sum_{i \in R_j} r_{i,t} \right)^2}{\sum_{i \in R_j} p_{i, t-1}(1 - p_{i, t-1}) + \lambda}$$
   Donde $\lambda$ es nuestro parámetro `l2_leaf_reg` (Regularización L2). Esta penalización frena los coeficientes de las hojas si hay muy pocas muestras en ellas, **evitando que el modelo memorice médicos atípicos**.

3. **Actualizar el Modelo:**
   $$F_t(x) = F_{t-1}(x) + \eta \cdot h_t(x)$$
   Donde $\eta$ es el `learning_rate` (tasa de aprendizaje, configurada por Optuna en nuestro modelo final en `0.1475`).

---

## 4. ¿Por qué CatBoost Ganó Nuestra Arena de Modelos?

En nuestro pipeline, evaluamos XGBoost, LightGBM, CatBoost e HistGradientBoosting. CatBoost destacó por dos motivos principales:

1. **Robustez ante Datos Pequeños y Altamente Desbalanceados:**
   Nuestra muestra contiene 633 médicos con solo 48 positivos (~7.5% de prevalencia). XGBoost y LightGBM tienden a sobreajustar (overfit) rápidamente bajo estas condiciones. Gracias a los **Árboles Simétricos** y el **Ordered Boosting**, CatBoost mantuvo una excelente generalización.
   
2. **Pesos de Clase Dinámicos (`auto_class_weights='Balanced'`):**
   Al entrenar, CatBoost asigna automáticamente un peso mayor al gradiente de la clase positiva (los médicos que sí prescriben):
   $$W_{\text{positivo}} = \frac{N_{\text{total}}}{2 \cdot N_{\text{positivo}}} = \frac{633}{2 \cdot 48} \approx 6.59$$
   Esto obliga a los árboles a enfocarse en los patrones de los prescriptores minoritarios en lugar de ignorarlos para minimizar el error global.

---

## 5. Explicabilidad con SHAP (Shapley Additive exPlanations)

CatBoost es una "caja negra" compleja, pero la abrimos usando **SHAP**. 

### Explicación Coloquial 📢
Imagina que el modelo predice que el Dr. Pérez tiene un **80%** de propensión a recetar. La propensión promedio global es **7.5%**. 
¿Cómo pasamos de 7.5% a 80%?
Los valores SHAP actúan como **fuerzas en un tira y afloja**:
* El hecho de que el Dr. Pérez tenga una desviación estándar de visitas alta (`DETAILS__std`) empuja la predicción hacia arriba en $+25\%$.
* El hecho de que su copago promedio (`COPAY__mean`) sea bajo empuja la predicción hacia abajo en $-5\%$.
* La suma de todas las contribuciones individuales de sus variables nos da exactamente el $80\%$ final.

### Explicación Matemática 🧮
La contribución SHAP de una característica $i$ para una predicción individual se basa en la teoría de juegos cooperativos. Se define como:
$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]$$
Donde:
* $F$ es el conjunto de todas las características.
* $S$ es un subconjunto de características que excluye a la variable $i$.
* $f_x(S)$ es la predicción esperada del modelo dado el subconjunto de variables $S$.

CatBoost implementa **TreeSHAP**, un algoritmo optimizado de tiempo lineal $O(T L D^2)$ (donde $T$ es árboles, $L$ hojas y $D$ profundidad) que calcula estas expectativas matemáticas de manera exacta navegando por las rutas del árbol sin necesidad de simular infinitas combinaciones de datos, permitiéndonos obtener los **Top 5 Reason Codes** para todos los 633 médicos en segundos.

---

## 6. Calibración de Probabilidades (Isotonic Regression)

### Explicación Coloquial 📢
Un modelo de Machine Learning clasifica muy bien quién es más propenso que otro, pero sus puntuaciones crudas están distorsionadas. Si CatBoost escupe un score de `0.90` a 10 médicos, no necesariamente significa que 9 de ellos vayan a recetar. 

Para corregir esto, aplicamos **Regresión Isotónica**. Esto es como llevar el modelo al calibrador: agrupamos los scores en rangos y los ajustamos con la frecuencia real observada en el set de validación, garantizando que **si el modelo calibrado dice 30%, realmente el 30% de ese grupo de médicos recete en la práctica**.

### Explicación Matemática 🧮
Dado el conjunto de predicciones crudas del modelo $\hat{y}_i$ y sus verdaderas etiquetas $y_i$, la Regresión Isotónica busca una función no decreciente (monótona) $g$ que minimice el error cuadrático medio:
$$\min_{g} \sum_{i=1}^{N} \left( y_i - g(\hat{y}_i) \right)^2 \quad \text{sujeto a } g(\hat{y}_a) \le g(\hat{y}_b) \text{ si } \hat{y}_a \le \hat{y}_b$$

Esto se resuelve de manera exacta utilizando el algoritmo **PAVA (Pool Adjacent Violators Algorithm)**.

Evaluamos el éxito de esta calibración con el **Brier Score**, que es equivalente al Error Cuadrático Medio de las probabilidades estimadas:
$$\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} \left( p_i - y_i \right)^2$$
En nuestro pipeline, la calibración **redujo el Brier Score de 0.0585 a 0.0465**, lo que significa que tus probabilidades ahora son **extremadamente honestas y listas para la toma de decisiones financieras y comerciales**.
